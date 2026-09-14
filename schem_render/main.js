import { SchematicRenderer } from 'schematic-renderer';

window.__renderReady = false;
window.__renderErr = null;
window.__renderer = null;

export async function initRenderer() {
  try {
    const canvas = document.getElementById('c');
    // 内置资源包(来自 dist/pack.zip), 通过 vite 静态服务提供
    const packUrl = '/node_modules/schematic-renderer/dist/pack.zip';
    let resolveReady;
    const readyPromise = new Promise(r => { resolveReady = r; });
    const renderer = new SchematicRenderer(
      canvas,
      {},
      {
        'vanilla': async () => {
          const r = await fetch(packUrl);
          return r.blob();
        },
      },
      {
        backgroundColor: 0x87ceeb,
        showGrid: false,
        showAxes: false,
        enableInteraction: false,
        enableDragAndDrop: false,
        enableGizmos: false,
        enableProgressBar: false,
        sidebarOptions: { enabled: false },
        postProcessingOptions: { enabled: false },
        cameraOptions: { position: [40, 40, 40] },
        callbacks: {
          onRendererInitialized: (r) => { resolveReady(r); },
        },
      }
    );
    // 等官方初始化回调(manager 在内部 async 初始化完成后赋值)
    const done = await Promise.race([
      readyPromise,
      new Promise(r => setTimeout(r, 120000)),
    ]);
    // 轮询兜底
    const t0 = Date.now();
    while (Date.now() - t0 < 60000) {
      if (renderer.schematicManager && renderer.worldMeshBuilder && renderer.renderManager) break;
      await new Promise(r => setTimeout(r, 500));
    }
    await new Promise(r => setTimeout(r, 2000));
    window.__renderer = renderer;
    window.__renderReady = true;
    console.log('renderer ready via=' + (done ? 'callback' : 'timeout') + ' sm=' + !!renderer.schematicManager + ' wmb=' + !!renderer.worldMeshBuilder + ' rm=' + !!renderer.renderManager);
  } catch (e) {
    window.__renderErr = String(e && e.stack || e);
    console.error(e);
  }
}

export async function renderLitematic(name, arrayBuffer) {
  const r = window.__renderer;
  if (!r) throw new Error('renderer not ready');
  try { await r.schematicManager.removeAllSchematics(); } catch (e) {}
  const info = await r.schematicManager.loadSchematic(name, arrayBuffer, {});
  await new Promise(res => setTimeout(res, 4000));
  try {
    r.setIsometricAngles(35.264, 45, true);
  } catch (e) { console.log('iso err', String(e)); }
  await new Promise(res => setTimeout(res, 3000));
  const blob = await r.takeScreenshot({ format: 'image/png', width: 768, height: 768 });
  const ab = await blob.arrayBuffer();
  return Array.from(new Uint8Array(ab));
}

initRenderer();
