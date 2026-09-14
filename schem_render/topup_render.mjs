// 定点补渲缺失 idx: 读取缺失列表(参数: 起始 结束 或单idx), 小批跑防内存
import { createServer } from 'http';
import { readFileSync, existsSync, mkdirSync, writeFileSync, createReadStream, statSync } from 'fs';
import { join, basename } from 'path';
import puppeteer from 'puppeteer';
import { createServer as createVite } from 'vite';

const ROOT = '/home/linux/schem_render';
const LIST = '/home/linux/redstone_ai/dataset/valid_files_list.txt';
const LIT_DIR = '/home/linux/redstone_ai/dataset/litematics';
const OUT_DIR = '/home/linux/redstone_ai/dataset/render_schem';
const CHROME = '/usr/bin/google-chrome';
const LIT_PORT = 8940;
const VITE_PORT = 8941;
mkdirSync(OUT_DIR, { recursive: true });

const litServer = createServer((req, res) => {
  let raw = req.url.split('?')[0];
  let name;
  try { name = decodeURIComponent(raw.replace(/^\//, '')); } catch (e) { name = basename(raw); }
  const fp = join(LIT_DIR, name);
  const cors = { 'Access-Control-Allow-Origin': '*', 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };
  if (!existsSync(fp)) { res.writeHead(404, cors); res.end('nf:' + name); return; }
  res.writeHead(200, { 'Content-Type': 'application/octet-stream', ...cors });
  createReadStream(fp).pipe(res);
});

function log(m) { console.log(`[${new Date().toISOString().slice(11,19)}] ${m}`); }

async function main() {
  const only = (process.argv[2] || '').split(',').map(s => parseInt(s, 10)).filter(n => !isNaN(n));
  await new Promise(r => litServer.listen(LIT_PORT, r));
  const names = readFileSync(LIST, 'utf-8').split('\n').map(s => s.trim()).filter(Boolean);
  log(`server up; targets: ${only.length}`);
  const vite = await createVite({ root: ROOT, server: { port: VITE_PORT, strictPort: true, host: '127.0.0.1' }, logLevel: 'error' });
  await vite.listen();
  const browser = await puppeteer.launch({
    headless: 'new', executablePath: CHROME,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-unsafe-swiftshader',
           '--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist',
           '--disable-dev-shm-usage', '--enable-features=SharedArrayBuffer'],
  });
  let page = await browser.newPage();
  await page.setViewport({ width: 800, height: 800, deviceScaleFactor: 1 });
  page.on('console', m => { const t = m.text(); if (t.includes('GPU stall') || t.includes('GL Driver') || t.includes('[vite]') || t.includes('High memory')) return; if (m.type() === 'error') log(`[pe] ${t.slice(0,150)}`); });
  page.on('pageerror', e => log('PAGEERR ' + String(e).slice(0, 200)));
  await page.goto(`http://127.0.0.1:${VITE_PORT}/index.html`, { waitUntil: 'networkidle0', timeout: 120000 });
  try { await page.waitForFunction('window.__renderReady === true', { timeout: 120000 }); }
  catch (e) { log('ready fail ' + (await page.evaluate('window.__renderErr') || '')); process.exit(1); }
  log('renderer ready');

  let done = 0, fail = 0, t0 = Date.now();
  for (let k = 0; k < only.length; k++) {
    const idx = only[k];
    if (idx < 0 || idx >= names.length) { log(`skip idx${idx} out of range`); continue; }
    const name = names[idx];
    const outFile = join(OUT_DIR, `bld_${String(idx).padStart(4, '0')}.png`);
    try { if (existsSync(outFile) && statSync(outFile).size > 20000) { done++; continue; } } catch (e) {}
    try {
      const ok = await page.evaluate(async ({ name, port }) => {
        const r = window.__renderer;
        try {
          const resp = await fetch(`http://127.0.0.1:${port}/${encodeURIComponent(name)}`);
          const buf = await resp.arrayBuffer();
          if (buf.byteLength < 50) return { ok: false, err: 'bad size ' + buf.byteLength };
          try { await r.schematicManager.removeAllSchematics(); } catch (e) {}
          await r.schematicManager.loadSchematic(name, buf, {});
          await new Promise(res => setTimeout(res, 1500));
          try { r.setIsometricAngles(35.264, 45, true); } catch (e) {}
          await new Promise(res => setTimeout(res, 1200));
          const blob = await r.takeScreenshot({ format: 'image/png', width: 768, height: 768 });
          const ab = await blob.arrayBuffer();
          return { ok: true, bytes: Array.from(new Uint8Array(ab)) };
        } catch (e) { return { ok: false, err: String((e && e.stack) || e).slice(0, 250) }; }
      }, { name, port: LIT_PORT });
      if (ok && ok.ok) { writeFileSync(outFile, Buffer.from(ok.bytes)); done++; }
      else { fail++; log(`idx${idx} FAIL ${name}: ${((ok && ok.err) || '').slice(0, 150)}`); }
    } catch (e) { fail++; log(`idx${idx} EXC: ${String(e).slice(0, 150)}`); }
    if ((k + 1) % 10 === 0 || k === only.length - 1) {
      const el = (Date.now() - t0) / 1000;
      log(`[${k + 1}/${only.length}] done=${done} fail=${fail} rate=${((done + fail) / Math.max(el, 1)).toFixed(2)}/s`);
    }
    // 每 10 张重启页防内存(仅剩 28 张, 保守)
    if ((k + 1) % 10 === 0 && k < only.length - 1) {
      log('page restart guard');
      await page.close();
      page = await browser.newPage();
      await page.setViewport({ width: 800, height: 800, deviceScaleFactor: 1 });
      page.on('console', m => { const t = m.text(); if (t.includes('GPU stall') || t.includes('GL Driver') || t.includes('[vite]') || t.includes('High memory')) return; if (m.type() === 'error') log(`[pe] ${t.slice(0,150)}`); });
      await page.goto(`http://127.0.0.1:${VITE_PORT}/index.html`, { waitUntil: 'networkidle0', timeout: 120000 });
      await page.waitForFunction('window.__renderReady === true', { timeout: 120000 });
    }
  }
  log(`TOPUP DONE done=${done} fail=${fail}`);
  await browser.close();
  await vite.close();
  litServer.close();
  process.exit(0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
