// Batch render NEW slice litematics -> condition images (same style as old render_output).
// Adapted from schem_render/batch_render.mjs; reads /data/redstone_v4/litematics_new/NNNNN.litematic
// outputs /data/redstone_v4/render_new/bld_00000.png (768x768, isometric 35.264/45).
import { createServer } from 'http';
import { readFileSync, existsSync, mkdirSync, writeFileSync, createReadStream } from 'fs';
import { join, basename } from 'path';
import puppeteer from 'puppeteer';
import { createServer as createVite } from 'vite';

const ROOT = '/home/linux/schem_render';
const LIT_DIR = '/data/redstone_v4/litematics_new';
const OUT_DIR = '/data/redstone_v4/render_new';
const CHROME = '/usr/bin/google-chrome';
const START = parseInt(process.argv[2] || '0', 10);
const END = parseInt(process.argv[3] || '999999', 10);
const RESTART_EVERY = parseInt(process.argv[4] || '400', 10);
const LIT_PORT = parseInt(process.argv[5] || "8942", 10);
const VITE_PORT = parseInt(process.argv[6] || "8943", 10);
mkdirSync(OUT_DIR, { recursive: true });

const litServer = createServer((req, res) => {
  let raw = req.url.split('?')[0];
  let name;
  try { name = decodeURIComponent(raw.replace(/^\//, '')); } catch (e) { name = basename(raw); }
  const fp = join(LIT_DIR, name);
  const cors = {
    'Access-Control-Allow-Origin': '*',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  };
  if (!existsSync(fp)) { res.writeHead(404, cors); res.end('nf:' + name); return; }
  res.writeHead(200, { 'Content-Type': 'application/octet-stream', ...cors });
  createReadStream(fp).pipe(res);
});

function log(msg) {
  console.log(`[${new Date().toISOString().slice(11, 19)}] ${msg}`);
}

// build name list from files on disk
import { readdirSync } from 'fs';
const all = readdirSync(LIT_DIR).filter(f => f.endsWith('.litematic')).sort();
const names = all;

async function main() {
  await new Promise(r => litServer.listen(LIT_PORT, r));
  log(`lit server on ${LIT_PORT}`);
  log(`total ${names.length}, range [${START}, ${Math.min(END, names.length)})`);
  const vite = await createVite({
    root: ROOT,
    server: { port: VITE_PORT, strictPort: true, host: '127.0.0.1' },
    logLevel: 'error',
  });
  await vite.listen();
  log(`vite on ${VITE_PORT}`);

  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: CHROME,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-unsafe-swiftshader',
           '--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist',
           '--disable-dev-shm-usage', '--enable-features=SharedArrayBuffer'],
  });
  let page = await browser.newPage();
  await page.setViewport({ width: 800, height: 800, deviceScaleFactor: 1 });
  page.on('console', m => {
    const t = m.text();
    if (t.includes('GPU stall') || t.includes('GL Driver') || t.includes('[vite]') || t.includes('High memory')) return;
    if (m.type() === 'error' || m.type() === 'warn') log(`[page:${m.type()}] ${t.slice(0, 200)}`);
  });
  page.on('pageerror', e => log('PAGEERR ' + String(e).slice(0, 250)));
  await page.goto(`http://127.0.0.1:${VITE_PORT}/index.html`, { waitUntil: 'networkidle0', timeout: 120000 });
  try {
    await page.waitForFunction('window.__renderReady === true', { timeout: 120000 });
  } catch (e) {
    log('ready timeout err=' + (await page.evaluate('window.__renderErr') || ''));
    process.exit(1);
  }
  log('renderer ready');

  let done = 0, fail = 0, t0 = Date.now();
  const range = names.slice(START, END);
  let sinceRestart = 0;
  for (let i = 0; i < range.length; i++) {
    if (RESTART_EVERY > 0 && sinceRestart >= RESTART_EVERY) {
      log(`restarting browser at idx ${START + i} (mem guard)`);
      await page.close();
      const p2 = await browser.newPage();
      await p2.setViewport({ width: 800, height: 800, deviceScaleFactor: 1 });
      p2.on('console', m => {
        const t = m.text();
        if (t.includes('GPU stall') || t.includes('GL Driver') || t.includes('[vite]') || t.includes('High memory')) return;
        if (m.type() === 'error' || m.type() === 'warn') log(`[page:${m.type()}] ${t.slice(0, 200)}`);
      });
      p2.on('pageerror', e => log('PAGEERR ' + String(e).slice(0, 250)));
      await p2.goto(`http://127.0.0.1:${VITE_PORT}/index.html`, { waitUntil: 'networkidle0', timeout: 120000 });
      try {
        await p2.waitForFunction('window.__renderReady === true', { timeout: 120000 });
      } catch (e) {
        log('re-ready timeout err=' + (await p2.evaluate('window.__renderErr') || ''));
        process.exit(1);
      }
      page = p2;
      sinceRestart = 0;
    }
    const name = range[i];
    const idx = START + i;
    const outFile = join(OUT_DIR, `bld_${String(idx).padStart(5, '0')}.png`);
    if (existsSync(outFile)) {
      try { if (readFileSync(outFile).length > 5000) { done++; continue; } } catch (e) {}
    }
    try {
      const ok = await page.evaluate(async ({ name, port }) => {
        const r = window.__renderer;
        if (!r) return { ok: false, err: 'no renderer' };
        try {
          const resp = await fetch(`http://127.0.0.1:${port}/${encodeURIComponent(name)}`);
          const buf = await resp.arrayBuffer();
          if (buf.byteLength < 50) return { ok: false, err: 'bad size ' + buf.byteLength };
          try { await r.schematicManager.removeAllSchematics(); } catch (e) {}
          const info = await r.schematicManager.loadSchematic(name, buf, {});
          await new Promise(res => setTimeout(res, 1500));
          try { r.setIsometricAngles(35.264, 45, true); } catch (e) {}
          await new Promise(res => setTimeout(res, 1200));
          const blob = await r.takeScreenshot({ format: 'image/png', width: 768, height: 768 });
          const ab = await blob.arrayBuffer();
          return { ok: true, bytes: Array.from(new Uint8Array(ab)) };
        } catch (e) {
          return { ok: false, err: String((e && e.stack) || e).slice(0, 300) };
        }
      }, { name, port: LIT_PORT });
      if (ok && ok.ok) {
        writeFileSync(outFile, Buffer.from(ok.bytes));
        done++;
        sinceRestart++;
      } else {
        fail++;
        log(`idx${idx} FAIL ${name}: ${((ok && ok.err) || '').slice(0, 200)}`);
      }
    } catch (e) {
      fail++;
      log(`idx${idx} EXC ${name}: ${String(e).slice(0, 200)}`);
    }
    if ((i + 1) % 25 === 0 || i === range.length - 1) {
      const el = (Date.now() - t0) / 1000;
      const rate = (done + fail) / Math.max(el, 1);
      const remain = (range.length - i - 1) / Math.max(rate, 0.01) / 60;
      log(`[${idx}/${START + range.length - 1}] done=${done} fail=${fail} rate=${rate.toFixed(2)}/s remain=${remain.toFixed(1)}m`);
    }
  }
  log(`BATCH DONE done=${done} fail=${fail} total=${range.length}`);
  await browser.close();
  await vite.close();
  litServer.close();
  process.exit(0);
}

main().catch(e => { console.error('FATAL', e); process.exit(1); });
