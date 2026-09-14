// Batch render all rotated litematics (90/180/270) -> bld_{i*4+k:04d}.png at 768x768.
// Usage: node rot_batch_render.mjs [startIdx] [endIdx]  (range over sorted latent-index list, resumable)
import { createServer } from 'http';
import { existsSync, mkdirSync, writeFileSync, readFileSync, readdirSync, createReadStream } from 'fs';
import { join } from 'path';
import puppeteer from 'puppeteer';
import { createServer as createVite } from 'vite';

const ROOT = '/home/linux/schem_render';
const LIT_DIR = '/home/linux/redstone_ai/render_mc/rot_litematics';
const OUT_DIR = '/home/linux/redstone_ai/render_output_rot';
const CHROME = '/usr/bin/google-chrome';
const LIT_PORT = 8940;
const VITE_PORT = 8941;
mkdirSync(OUT_DIR, { recursive: true });

const START = parseInt(process.argv[2] || '0', 10);
const END = parseInt(process.argv[3] || '999999', 10);
const WAIT_LOAD = parseInt(process.argv[4] || '1200', 10);
const WAIT_ANGLE = parseInt(process.argv[5] || '900', 10);

// collect all rotated litematics, map to latent index
const files = readdirSync(LIT_DIR).filter(f => /^\d{4}_r\d{3}\.litematic$/.test(f));
const items = files.map(f => {
  const m = f.match(/^(\d{4})_r(\d{3})\.litematic$/);
  const i = parseInt(m[1], 10);
  const ang = parseInt(m[2], 10);
  return { key: f, i, ang, latidx: i * 4 + ang / 90, file: join(LIT_DIR, f) };
}).sort((a, b) => a.latidx - b.latidx || a.ang - b.ang);
console.log('total rot litematics:', items.length, 'range', START, '..', Math.min(END, items.length));

const fileByKey = {};
for (const it of items) fileByKey[it.key] = it.file;

const litServer = createServer((req, res) => {
  const key = decodeURIComponent(req.url.split('?')[0].replace(/^\//, ''));
  const fp = fileByKey[key];
  if (!fp || !existsSync(fp)) { res.writeHead(404); res.end('nf'); return; }
  res.writeHead(200, {
    'Content-Type': 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Access-Control-Allow-Origin': '*',
  });
  createReadStream(fp).pipe(res);
});

function log(m) { console.log(`[${new Date().toISOString().slice(11,19)}] ${m}`); }

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
  page.on('console', m => { const t = m.text(); if (m.type() === 'error' || m.type() === 'warn') log(`[page:${m.type()}] ${t.slice(0,160)}`); });
  page.on('pageerror', e => log('PAGEERR ' + String(e).slice(0, 200)));
  await page.goto(`http://127.0.0.1:${VITE_PORT}/index.html`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction('window.__renderReady === true', { timeout: 120000 });
  log('renderer ready');

  let done = 0, fail = 0, skip = 0, t0 = Date.now();
  const range = items.slice(START, END);
  for (let idx = 0; idx < range.length; idx++) {
    const it = range[idx];
    const outFile = join(OUT_DIR, `bld_${String(it.latidx).padStart(4, '0')}.png`);
    if (existsSync(outFile)) {
      try { if (readFileSync(outFile).length > 5000) { skip++; log(`skip idx=${it.latidx} (${it.key})`); continue; } } catch (e) {}
    }
    try {
      const ok = await page.evaluate(async ({ key, port, wa, wb }) => {
        const r = window.__renderer;
        if (!r) return { ok:false, err:'no renderer' };
        try {
          const resp = await fetch(`http://127.0.0.1:${port}/${encodeURIComponent(key)}`);
          const buf = await resp.arrayBuffer();
          if (buf.byteLength < 50) return { ok:false, err:'bad size ' + buf.byteLength };
          try { await r.schematicManager.removeAllSchematics(); } catch (e) {}
          await r.schematicManager.loadSchematic(key, buf, {});
          await new Promise(res => setTimeout(res, wa));
          try { r.setIsometricAngles(35.264, 45, true); } catch (e) {}
          await new Promise(res => setTimeout(res, wb));
          const blob = await r.takeScreenshot({ format:'image/png', width:768, height:768 });
          const ab = await blob.arrayBuffer();
          return { ok:true, bytes:Array.from(new Uint8Array(ab)) };
        } catch (e) { return { ok:false, err:String((e && e.stack)||e).slice(0,300) }; }
      }, { key: it.key, port: LIT_PORT, wa: WAIT_LOAD, wb: WAIT_ANGLE });
      if (ok && ok.ok) {
        writeFileSync(outFile, Buffer.from(ok.bytes));
        done++;
      } else {
        fail++;
        log(`FAIL idx=${it.latidx} (${it.key}): ${((ok && ok.err)||'').slice(0,160)}`);
      }
    } catch (e) {
      fail++;
      log(`EXC idx=${it.latidx}: ${String(e).slice(0,160)}`);
    }
    if ((idx + 1) % 50 === 0 || idx === range.length - 1) {
      const el = (Date.now() - t0) / 1000;
      const rate = done / Math.max(el, 0.001);
      log(`[${START + idx + 1}/${START + range.length}] done=${done} skip=${skip} fail=${fail} rate=${rate.toFixed(2)}/s elapsed=${el.toFixed(0)}s`);
    }
  }
  log(`CHUNK done done=${done} skip=${skip} fail=${fail}`);
  await browser.close();
  await vite.close();
  litServer.close();
  process.exit(0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
