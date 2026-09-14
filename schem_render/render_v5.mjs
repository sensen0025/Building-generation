// 渲染旋转槽位 litematic -> 条件图. 渲染代码块与生产 render_new_batch.mjs 完全一致
// (isometric 35.264/45, 768x768, 同等待时长), 仅改输入/输出目录与命名.
// 输入: /data/redstone_v4/litematics_rot/{slot:05d}.litematic
// 输出: /data/redstone_v4/render_rot/bld_{slot:05d}.png
// 用法: node render_rot_batch.mjs <SLOT_START> <SLOT_END> [RESTART_EVERY] [LIT_PORT] [VITE_PORT]
// 按 slot 区间分片 (不是数组位置), 因此某区间建完即可独立开渲.
import { createServer } from 'http';
import { readFileSync, existsSync, mkdirSync, writeFileSync, createReadStream, readdirSync } from 'fs';
import { join, basename } from 'path';
import { createHash } from 'crypto';
import { renameSync } from 'fs';
import puppeteer from 'puppeteer';
import { createServer as createVite } from 'vite';

const ROOT = '/home/linux/schem_render';
const LIT_DIR = process.env.MCFIX_LIT_DIR;
if (!LIT_DIR) throw new Error('MCFIX_LIT_DIR required');
const OUT_DIR = process.env.MCFIX_RENDER_DIR;
if (!OUT_DIR) throw new Error('MCFIX_RENDER_DIR required');
const sha = p => createHash('sha256').update(readFileSync(p)).digest('hex');
const rendererHash = sha(new URL(import.meta.url));
const packHash = sha('/home/linux/schem_render/node_modules/schematic-renderer/dist/pack.zip');
const CHROME = '/usr/bin/google-chrome';
const START = parseInt(process.argv[2] || '0', 10);
const END = parseInt(process.argv[3] || '9999999', 10);
const RESTART_EVERY = parseInt(process.argv[4] || '400', 10);
const LIT_PORT = parseInt(process.argv[5] || '8990', 10);
const VITE_PORT = parseInt(process.argv[6] || '8991', 10);
mkdirSync(OUT_DIR, { recursive: true });

const litServer = createServer((req, res) => {
  let raw = req.url.split('?')[0];
  let name;
  try { name = decodeURIComponent(raw.replace(/^\//, '')); } catch (e) { name = basename(raw); }
  if (name !== basename(name) || !/^\d+\.litematic$/.test(name)) { res.writeHead(400); res.end(); return; }
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

const all = readdirSync(LIT_DIR).filter(f => f.endsWith('.litematic')).sort();
const slotOf = f => parseInt(basename(f).replace('.litematic', ''), 10);
// 按 slot 区间筛选 (build 与 render 可重叠, 不依赖数组位置)
const shardCount = parseInt(process.env.MCFIX_SHARDS || '1',10);
const shardId = parseInt(process.env.MCFIX_SHARD_ID || '0',10);
const names = all.filter(f => { const s = slotOf(f); return s >= START && s < END && s % shardCount === shardId; });

async function main() {
  await new Promise(r => litServer.listen(LIT_PORT, '127.0.0.1', r));
  log(`lit server on ${LIT_PORT}`);
  log(`total ${names.length} files in slot range [${START}, ${END})`);
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

  let done = 0, fail = 0, skip = 0, sinceRestart = 0, attempts = 0;
  const t0 = Date.now();
  const range = names;
  for (let i = 0; i < range.length; i++) {
    // 关键: 用「尝试次数」而非「成功次数」触发页面重启 —
    // 否则一旦 Chrome 卡死进入失败循环, 永远等不到重启 (实测踩过)
    if (attempts >= RESTART_EVERY) {
      try { await page.close(); } catch (e) {}
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
      attempts = 0;
    }
    const name = range[i];
    const slot = parseInt(basename(name).replace('.litematic', ''), 10);
    const outFile = join(OUT_DIR, `bld_${String(slot).padStart(5, '0')}.png`);
    const key = { input: sha(join(LIT_DIR,name)), renderer: rendererHash, pack: packHash, camera: [35.264,45], size: 768 };
    if (existsSync(outFile) && existsSync(outFile+'.json')) {
      try { const old=JSON.parse(readFileSync(outFile+'.json','utf8')); if (JSON.stringify(old.key)===JSON.stringify(key) && old.output===sha(outFile)) {skip++;continue;} } catch(e) {}
    }
    try {
      let timer;
      const ok = await Promise.race([page.evaluate(async ({ name, port }) => {
        const r = window.__renderer;
        if (!r) return { ok: false, err: 'no renderer' };
        try {
          const resp = await fetch(`http://127.0.0.1:${port}/${encodeURIComponent(name)}`);
          const buf = await resp.arrayBuffer();
          if (!resp.ok || buf.byteLength < 50) return { ok: false, err: 'bad size ' + buf.byteLength };
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
      }, { name, port: LIT_PORT }), new Promise((_,reject)=> {timer=setTimeout(()=>reject(new Error("render timeout")),90000);})]).finally(()=>clearTimeout(timer));
      attempts++;
      if (ok && ok.ok) {
        const bytes=Buffer.from(ok.bytes);
        if (bytes.length<1000 || bytes.subarray(1,4).toString()!=='PNG') throw new Error('Invalid screenshot');
        writeFileSync(outFile+'.tmp', bytes); renameSync(outFile+'.tmp',outFile);
        writeFileSync(outFile+'.json.tmp',JSON.stringify({key,output:sha(outFile)})); renameSync(outFile+'.json.tmp',outFile+'.json');
        done++;
        sinceRestart++;
      } else {
        fail++;
        log(`slot${slot} FAIL ${name}: ${((ok && ok.err) || '').slice(0, 200)}`);
      }
    } catch (e) {
      attempts++;
      fail++;
      log(`slot${slot} EXC ${name}: ${String(e).slice(0, 200)}`);
      attempts = RESTART_EVERY;
    }
    if ((i + 1) % 25 === 0 || i === range.length - 1) {
      const el = (Date.now() - t0) / 1000;
      const rate = (done + fail) / Math.max(el, 1);
      const remain = (range.length - i - 1) / Math.max(rate, 0.01) / 60;
      log(`slot${slot} done=${done} fail=${fail} skip=${skip} rate=${rate.toFixed(2)}/s remain=${remain.toFixed(1)}m`);
    }
  }
  log(`BATCH DONE done=${done} fail=${fail} skip=${skip} total=${range.length}`);
  await browser.close();
  await vite.close();
  litServer.close();
  process.exit(fail ? 1 : 0);
}

main().catch(e => { console.error('FATAL', e); process.exit(1); });
