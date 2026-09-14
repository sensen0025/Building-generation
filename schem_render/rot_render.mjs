// Render N buildings x 4 rotations (0/90/180/270) via schem_render (Three.js + Chromium).
// Mirrors batch_render.mjs: loadSchematic -> setIsometricAngles(35.264,45,true) -> screenshot 768x768.
import { createServer } from 'http';
import { existsSync, mkdirSync, writeFileSync, readFileSync, createReadStream } from 'fs';
import { join } from 'path';
import puppeteer from 'puppeteer';
import { createServer as createVite } from 'vite';

const ROOT = '/home/linux/schem_render';
const OUT_DIR = '/home/linux/redstone_ai/rot_preview';
const CHROME = '/usr/bin/google-chrome';
const LIT_PORT = 8940;
const VITE_PORT = 8941;
mkdirSync(OUT_DIR, { recursive: true });

const TARGETS = [
  { key: 'k0030_r000', out: 'bld_0030_r000.png', file: '/home/linux/redstone_ai/dataset/litematics/16453.litematic' },
  { key: 'k0030_r090', out: 'bld_0030_r090.png', file: '/home/linux/redstone_ai/render_mc/rot_litematics/0030_r090.litematic' },
  { key: 'k0030_r180', out: 'bld_0030_r180.png', file: '/home/linux/redstone_ai/render_mc/rot_litematics/0030_r180.litematic' },
  { key: 'k0030_r270', out: 'bld_0030_r270.png', file: '/home/linux/redstone_ai/render_mc/rot_litematics/0030_r270.litematic' },
  { key: 'k0038_r000', out: 'bld_0038_r000.png', file: '/home/linux/redstone_ai/dataset/litematics/16463.litematic' },
  { key: 'k0038_r090', out: 'bld_0038_r090.png', file: '/home/linux/redstone_ai/render_mc/rot_litematics/0038_r090.litematic' },
  { key: 'k0038_r180', out: 'bld_0038_r180.png', file: '/home/linux/redstone_ai/render_mc/rot_litematics/0038_r180.litematic' },
  { key: 'k0038_r270', out: 'bld_0038_r270.png', file: '/home/linux/redstone_ai/render_mc/rot_litematics/0038_r270.litematic' },
];
const fileByKey = {};
for (const t of TARGETS) fileByKey[t.key] = t.file;

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
  log('lit server on ' + LIT_PORT);
  const vite = await createVite({ root: ROOT, server: { port: VITE_PORT, strictPort: true, host: '127.0.0.1' }, logLevel: 'error' });
  await vite.listen();
  log('vite on ' + VITE_PORT);
  const browser = await puppeteer.launch({
    headless: 'new', executablePath: CHROME,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-unsafe-swiftshader',
           '--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist',
           '--disable-dev-shm-usage', '--enable-features=SharedArrayBuffer'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 800, height: 800, deviceScaleFactor: 1 });
  page.on('console', m => { const t = m.text(); if (m.type() === 'error' || m.type() === 'warn') log(`[page:${m.type()}] ${t.slice(0,200)}`); });
  page.on('pageerror', e => log('PAGEERR ' + String(e).slice(0, 250)));
  await page.goto(`http://127.0.0.1:${VITE_PORT}/index.html`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction('window.__renderReady === true', { timeout: 120000 });
  log('renderer ready');

  for (const t of TARGETS) {
    const outFile = join(OUT_DIR, t.out);
    if (existsSync(outFile) && readFileSync(outFile).length > 5000) { log('skip ' + t.key); continue; }
    try {
      const ok = await page.evaluate(async ({ key, port }) => {
        const r = window.__renderer;
        if (!r) return { ok:false, err:'no renderer' };
        try {
          const resp = await fetch(`http://127.0.0.1:${port}/${encodeURIComponent(key)}`);
          const buf = await resp.arrayBuffer();
          if (buf.byteLength < 50) return { ok:false, err:'bad size ' + buf.byteLength };
          try { await r.schematicManager.removeAllSchematics(); } catch (e) {}
          await r.schematicManager.loadSchematic(key, buf, {});
          await new Promise(res => setTimeout(res, 1500));
          try { r.setIsometricAngles(35.264, 45, true); } catch (e) {}
          await new Promise(res => setTimeout(res, 1200));
          const blob = await r.takeScreenshot({ format:'image/png', width:768, height:768 });
          const ab = await blob.arrayBuffer();
          return { ok:true, bytes:Array.from(new Uint8Array(ab)) };
        } catch (e) { return { ok:false, err:String((e && e.stack)||e).slice(0,300) }; }
      }, { key: t.key, port: LIT_PORT });
      if (ok && ok.ok) {
        writeFileSync(outFile, Buffer.from(ok.bytes));
        log('WROTE ' + t.out);
      } else {
        log('FAIL ' + t.key + ': ' + ((ok && ok.err)||'').slice(0,200));
      }
    } catch (e) { log('EXC ' + t.key + ': ' + String(e).slice(0,200)); }
  }
  log('DONE');
  await browser.close();
  await vite.close();
  litServer.close();
  process.exit(0);
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
