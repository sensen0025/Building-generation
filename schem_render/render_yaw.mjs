// 按 manifest 指定的 yaw 渲染 litematic; 用于验证旋转几何映射 + 批量补渲染.
// 用法: node render_yaw.mjs <manifest.json>
// manifest: { "out_dir": "...", "targets": [ {"key","file","out","yaw"} ] }
// 兼容: 若 target 带 "slot": [start,end) 形式则忽略 (由上层生成具体 target).
import { createServer } from 'http';
import { existsSync, mkdirSync, writeFileSync, readFileSync, createReadStream } from 'fs';
import { join } from 'path';
import puppeteer from 'puppeteer';
import { createServer as createVite } from 'vite';

const ROOT = '/home/linux/schem_render';
const CHROME = '/usr/bin/google-chrome';
const LIT_PORT = parseInt(process.env.LIT_PORT || '8960', 10);
const VITE_PORT = parseInt(process.env.VITE_PORT || '8961', 10);
const PITCH = 35.264;

const manifest = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const OUT_DIR = manifest.out_dir;
const TARGETS = manifest.targets.filter(t => t.file && t.out);
mkdirSync(OUT_DIR, { recursive: true });

const fileByKey = {};
for (const t of TARGETS) fileByKey[t.key] = t.file;

const litServer = createServer((req, res) => {
  const key = decodeURIComponent(req.url.split('?')[0].replace(/^\//, ''));
  const fp = fileByKey[key];
  const cors = {
    'Access-Control-Allow-Origin': '*',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  };
  if (!fp || !existsSync(fp)) { res.writeHead(404, cors); res.end('nf'); return; }
  res.writeHead(200, { 'Content-Type': 'application/octet-stream', ...cors });
  createReadStream(fp).pipe(res);
});

function log(m) { console.log(`[${new Date().toISOString().slice(11, 19)}] ${m}`); }

async function main() {
  await new Promise(r => litServer.listen(LIT_PORT, r));
  const vite = await createVite({ root: ROOT, server: { port: VITE_PORT, strictPort: true, host: '127.0.0.1' }, logLevel: 'error' });
  await vite.listen();
  const browser = await puppeteer.launch({
    headless: 'new', executablePath: CHROME,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-unsafe-swiftshader',
           '--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist',
           '--disable-dev-shm-usage', '--enable-features=SharedArrayBuffer'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 800, height: 800, deviceScaleFactor: 1 });
  page.on('console', m => {
    const t = m.text();
    if (t.includes('GPU stall') || t.includes('GL Driver') || t.includes('[vite]') || t.includes('High memory')) return;
    if (m.type() === 'error') log(`[page:error] ${t.slice(0, 160)}`);
  });
  page.on('pageerror', e => log('PAGEERR ' + String(e).slice(0, 200)));
  await page.goto(`http://127.0.0.1:${VITE_PORT}/index.html`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction('window.__renderReady === true', { timeout: 120000 });
  log('renderer ready');

  let done = 0, fail = 0, skip = 0;
  const t0 = Date.now();
  for (const t of TARGETS) {
    const outFile = join(OUT_DIR, t.out);
    if (existsSync(outFile)) {
      try { if (readFileSync(outFile).length > 5000) { skip++; continue; } } catch (e) {}
    }
    try {
      const ok = await page.evaluate(async ({ key, port, yaw, pitch }) => {
        const r = window.__renderer;
        if (!r) return { ok: false, err: 'no renderer' };
        try {
          const resp = await fetch(`http://127.0.0.1:${port}/${encodeURIComponent(key)}`);
          const buf = await resp.arrayBuffer();
          if (buf.byteLength < 50) return { ok: false, err: 'bad size ' + buf.byteLength };
          try { await r.schematicManager.removeAllSchematics(); } catch (e) {}
          await r.schematicManager.loadSchematic(key, buf, {});
          await new Promise(res => setTimeout(res, 1500));
          try { r.setIsometricAngles(pitch, yaw, true); } catch (e) {}
          await new Promise(res => setTimeout(res, 1200));
          const blob = await r.takeScreenshot({ format: 'image/png', width: 768, height: 768 });
          const ab = await blob.arrayBuffer();
          return { ok: true, bytes: Array.from(new Uint8Array(ab)) };
        } catch (e) { return { ok: false, err: String((e && e.stack) || e).slice(0, 300) }; }
      }, { key: t.key, port: LIT_PORT, yaw: t.yaw, pitch: PITCH });
      if (ok && ok.ok) {
        writeFileSync(outFile, Buffer.from(ok.bytes));
        done++;
      } else {
        fail++;
        log(`FAIL ${t.key} yaw=${t.yaw}: ${((ok && ok.err) || '').slice(0, 200)}`);
      }
    } catch (e) { fail++; log(`EXC ${t.key}: ${String(e).slice(0, 200)}`); }
    if ((done + fail) % 200 === 0 && (done + fail) > 0) {
      const el = (Date.now() - t0) / 1000;
      const rate = (done + fail) / Math.max(el, 1);
      log(`done=${done} fail=${fail} skip=${skip} rate=${rate.toFixed(2)}/s remain=${((TARGETS.length - done - fail - skip) / Math.max(rate, 0.01) / 60).toFixed(1)}m`);
    }
  }
  log(`BATCH DONE done=${done} fail=${fail} skip=${skip} total=${TARGETS.length}`);
  await browser.close();
  await vite.close();
  litServer.close();
  process.exit(0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
