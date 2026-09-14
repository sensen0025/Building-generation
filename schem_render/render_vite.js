// vite 版 puppeteer 驱动
const http = require('http');
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');
const { createServer: createVite } = require('vite');

const ROOT = __dirname;
const PORT = 8931;
const SAMPLE_DIR = path.join(ROOT, '..', '_litematic_test');

const MIME = { '.litematic': 'application/octet-stream', '.png': 'image/png' };
const sampleServer = http.createServer((req, res) => {
  const name = path.basename(req.url.split('?')[0]);
  const fp = path.join(SAMPLE_DIR, name);
  if (!fs.existsSync(fp)) { res.writeHead(404); res.end('nf'); return; }
  res.writeHead(200, {
    'Content-Type': 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Access-Control-Allow-Origin': '*',
  });
  fs.createReadStream(fp).pipe(res);
});

async function main() {
  const sample = process.argv[2] || 's250';
  await new Promise(r => sampleServer.listen(PORT, r));
  console.log('sample server on', PORT);

  const vite = await createVite({
    root: ROOT,
    server: { port: 8932, strictPort: true, host: '127.0.0.1' },
    logLevel: 'error',
  });
  await vite.listen();
  console.log('vite on 8932');

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-unsafe-swiftshader',
           '--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist',
           '--enable-features=SharedArrayBuffer'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 800, height: 800, deviceScaleFactor: 1 });
  page.on('console', m => console.log('[page]', m.type(), m.text().slice(0, 250)));
  page.on('pageerror', e => console.log('[pageerror]', String(e).slice(0, 300)));
  await page.goto('http://127.0.0.1:8932/', { waitUntil: 'networkidle0', timeout: 90000 });
  try {
    await page.waitForFunction('window.__renderReady === true', { timeout: 60000 });
  } catch (e) {
    console.log('ready timeout; err=', await page.evaluate('window.__renderErr'));
  }
  const err0 = await page.evaluate('window.__renderErr');
  if (err0) console.log('INIT ERR:', String(err0).slice(0, 600));

  const res = await page.evaluate(async (sample, base) => {
    const resp = await fetch(`${base}/${sample}.litematic`);
    const buf = await resp.arrayBuffer();
    console.log('fetched', buf.byteLength);
    // import 模块里的 renderLitematic
    const mod = await import('/main.js');
    const bytes = await mod.renderLitematic(sample, buf);
    return { ok: true, bytes };
  }, sample, `http://127.0.0.1:${PORT}`);

  if (res && res.ok) {
    const out = path.join(__dirname, `${sample}_render.png`);
    fs.writeFileSync(out, Buffer.from(res.bytes));
    console.log('SAVED', out, res.bytes.length);
  } else {
    console.log('FAIL', JSON.stringify(res).slice(0, 800));
  }
  await browser.close();
  await vite.close();
  sampleServer.close();
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
