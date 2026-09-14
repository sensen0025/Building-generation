// 远程单张调试: 完整输出 console, 检查 schematicManager 状态
import { createServer } from 'http';
import { readFileSync, existsSync, mkdirSync, writeFileSync, createReadStream } from 'fs';
import { join, basename } from 'path';
import puppeteer from 'puppeteer';
import { createServer as createVite } from 'vite';

const ROOT = '/home/linux/schem_render';
const LIT_DIR = '/home/linux/redstone_ai/dataset/litematics';
const CHROME = '/usr/bin/google-chrome';
const LIT_PORT = 8940;
const VITE_PORT = 8941;

const litServer = createServer((req, res) => {
  const name = basename(req.url.split('?')[0]);
  const fp = join(LIT_DIR, name);
  if (!existsSync(fp)) { res.writeHead(404); res.end('nf'); return; }
  res.writeHead(200, { 'Content-Type': 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp',
    'Access-Control-Allow-Origin': '*' });
  createReadStream(fp).pipe(res);
});

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
  page.on('console', m => console.log('[page:' + m.type() + ']', m.text().slice(0, 300)));
  page.on('pageerror', e => console.log('[PAGEERR]', String(e).slice(0, 400)));
  await page.goto(`http://127.0.0.1:${VITE_PORT}/index.html`, { waitUntil: 'networkidle0', timeout: 120000 });
  try {
    await page.waitForFunction('window.__renderReady === true', { timeout: 120000 });
    console.log('READY true');
  } catch (e) {
    console.log('READY false, err=', await page.evaluate('window.__renderErr'));
  }
  // 检查关键组件
  const state = await page.evaluate(() => {
    const r = window.__renderer;
    if (!r) return { noRenderer: true };
    return {
      hasSchematicManager: !!r.schematicManager,
      hasCameraManager: !!r.cameraManager,
      hasWorldMeshBuilder: !!r.worldMeshBuilder,
      keys: Object.keys(r).filter(k => k.includes('Manager') || k.includes('Builder')).slice(0, 30),
    };
  });
  console.log('STATE', JSON.stringify(state, null, 1));
  await browser.close();
  await vite.close();
  litServer.close();
}
main().catch(e => { console.error('FATAL', e); process.exit(1); });
