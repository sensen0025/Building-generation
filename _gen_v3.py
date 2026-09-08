"""v3 生成验证: 图像 -> flow(v3) -> decoder(163ch) -> id + state -> 带state litematic.
统计: IoU, 台阶/楼梯/活板门保留率, state 字段准确率(id对时).
"""
import os, sys, glob
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
import numpy as np
import torch
from PIL import Image
from litemapy import Region, BlockState, Schematic

from trellis.models import SparseStructureFlowModel, SparseStructureDecoder
from trellis.pipelines.samplers import FlowEulerCfgSampler
from transformers import AutoModel

TRELLIS = '/home/linux/redstone_ai/TRELLIS'
FLOW_CKPT = f'{TRELLIS}/outputs/ss_flow_mc_img_v3/ckpts'
VAE_CKPT = f'{TRELLIS}/outputs/ss_vae_mc_v2/ckpts'
IMG_DIR = '/home/linux/redstone_ai/render_output'
SS64 = '/home/linux/redstone_ai/dataset/v2/ss64.npy'
BLOCK_MAP = '/home/linux/redstone_ai/dataset/v2/block_map.npy'
VOCAB = '/home/linux/redstone_ai/dataset/BLOCK_VOCAB_FULL.txt'
OUT = '/home/linux/redstone_ai/gen_outputs'
NUM_CLASSES = 128
STATE_DIMS = [7, 3, 6, 16, 3]
FIELD_MULT = [1, 7, 21, 126, 2016]
# 真实有状态类 (v2 数据映射实测: 1..62; 63 死槽; 64..71 为无状态 plain; 127=other 语义不可解释)
STATE_CLS = list(range(1, 63))

# ---- 归一化统计: 直接读 flow 训练配置, 保证与训练完全一致 ----
def load_norm_from_config():
    import json as _json
    cfg_path = os.path.join(TRELLIS, 'configs/flow/ss_flow_mc_v3.json')
    with open(cfg_path, encoding='utf-8') as _f:
        cfg = _json.load(_f)
    nm = cfg['dataset']['args']['normalize']
    return nm['mean'], nm['std']

NORM = {}
NORM['mean'], NORM['std'] = load_norm_from_config()
print('NORM from config:', flush=True)
print('  mean', NORM['mean'], flush=True)
print('  std ', NORM['std'], flush=True)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {dev}', flush=True)

def latest_ema(prefix, d):
    files = [f for f in glob.glob(os.path.join(d, f'{prefix}*ema*.pt'))]
    if not files:
        files = glob.glob(os.path.join(d, f'{prefix}*.pt'))
    return max(files, key=lambda f: int(os.path.basename(f).split('step')[-1].split('.')[0]))

flow_ckpt = latest_ema('denoiser', FLOW_CKPT)
dec_ckpt = latest_ema('decoder', VAE_CKPT)
print('flow ckpt:', os.path.basename(flow_ckpt), flush=True)
print('dec  ckpt:', os.path.basename(dec_ckpt), flush=True)

denoiser = SparseStructureFlowModel(resolution=16, in_channels=8, out_channels=8,
    model_channels=512, cond_channels=1024, num_blocks=8, num_heads=8, num_head_channels=64,
    mlp_ratio=4, patch_size=2, pe_mode='ape', use_fp16=False).to(dev)
denoiser.load_state_dict(torch.load(flow_ckpt, map_location='cpu', weights_only=True))
denoiser.eval()
decoder = SparseStructureDecoder(out_channels=NUM_CLASSES + sum(STATE_DIMS), latent_channels=8,
    num_res_blocks=2, num_res_blocks_middle=2, channels=[512, 128, 32], use_fp16=False).to(dev)
decoder.load_state_dict(torch.load(dec_ckpt, map_location='cpu', weights_only=True))
decoder.eval()
dino = AutoModel.from_pretrained('facebook/dinov2-large', local_files_only=True).to(dev).eval()
im_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
im_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

block_map = np.load(BLOCK_MAP)          # new_id -> old vocab(202) id
vocab = [l.strip() for l in open(VOCAB, encoding='utf-8') if l.strip()]
def block_name(nid):
    oi = int(block_map[nid])
    return vocab[oi] if 0 <= oi < len(vocab) else 'minecraft:stone'

def state_to_props(name, fields):
    o, v, sh, hz, ac = [int(f) for f in fields]
    props = {}
    facing = {1: 'north', 2: 'south', 3: 'east', 4: 'west'}
    if 'stair' in name:
        if o in facing: props['facing'] = facing[o]
        if v == 1: props['half'] = 'top'
        elif v == 2: props['half'] = 'bottom'
        if sh: props['shape'] = {1:'straight',2:'inner_left',3:'inner_right',4:'outer_left',5:'outer_right'}[sh]
    elif 'slab' in name:
        if v == 1: props['type'] = 'top'
        elif v == 2: props['type'] = 'bottom'
    elif 'trapdoor' in name:
        if o in facing: props['facing'] = facing[o]
        if v == 1: props['half'] = 'top'
        elif v == 2: props['half'] = 'bottom'
        if ac: props['open'] = 'true' if ac == 2 else 'false'
    elif any(k in name for k in ['log', '_wood', '_stem', 'hay', 'basalt', 'mushroom', 'froglight', 'quartz_pillar', 'purpur_pillar']):
        if o: props['axis'] = {3: 'x', 1: 'z', 5: 'y'}.get(o, 'y')
    elif any(k in name for k in ['fence', 'glass_pane', 'bars']):
        for i, d in enumerate(['north', 'south', 'east', 'west']):
            if hz & (1 << i): props[d] = 'true'
    elif 'chest' in name:
        if o in facing: props['facing'] = facing[o]
    elif 'torch' in name or 'lever' in name or 'button' in name:
        if o in facing: props['facing'] = facing[o]
    elif 'door' in name:
        if o in facing: props['facing'] = facing[o]
        if v == 1: props['half'] = 'upper'
        elif v == 2: props['half'] = 'lower'
        if ac: props['open'] = 'true' if ac == 2 else 'false'
    elif 'ladder' in name or 'end_rod' in name or 'chain' in name or 'lightning_rod' in name:
        if o in facing: props['facing'] = facing[o]
        if o == 5: props['facing'] = 'up'
        if o == 6: props['facing'] = 'down'
    return props

def gen_from_image(idx, cfg=1.5, steps=50, seed=0):
    torch.manual_seed(seed)
    img = Image.open(os.path.join(IMG_DIR, f'bld_{idx:04d}.png')).convert('RGB')
    w, h = img.size; s = min(w, h); l = (w - s) // 2; t = (h - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((518, 518), Image.LANCZOS)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
    with torch.no_grad():
        feats = dino((x - im_mean) / im_std).last_hidden_state
    sampler = FlowEulerCfgSampler(1e-5)
    noise = torch.randn(1, 8, 16, 16, 16, device=dev)
    with torch.no_grad():
        res = sampler.sample(denoiser, noise=noise, cond=feats, neg_cond=torch.zeros_like(feats),
                             steps=steps, cfg_strength=cfg, verbose=False)
    z = res.samples
    m = torch.tensor(NORM['mean'], device=dev).view(1, 8, 1, 1, 1)
    sd = torch.tensor(NORM['std'], device=dev).view(1, 8, 1, 1, 1)
    z = z * sd + m
    with torch.no_grad():
        logits = decoder(z)  # [1, 163, 64,64,64]
        id_logits = logits[:, :NUM_CLASSES]
        st_logits = logits[:, NUM_CLASSES:NUM_CLASSES + sum(STATE_DIMS)]
        vox = id_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int16)
        fields = []
        off = 0
        for dim in STATE_DIMS:
            fields.append(st_logits[:, off:off + dim].argmax(1).squeeze(0).cpu().numpy().astype(np.int16))
            off += dim
    return vox, fields

def bbox(v):
    nz = np.argwhere(v > 0)
    return (nz.max(0) - nz.min(0) + 1).tolist() if len(nz) else None

def to_litematic(vox, fields, name, path):
    G = vox.shape[0]
    reg = Region(0, 0, 0, G, G, G)
    n = 0
    for x in range(G):
        for y in range(G):
            for z in range(G):
                c = int(vox[x, y, z])
                if c > 0:
                    nm = block_name(c)
                    if nm == 'minecraft:air':
                        nm = 'minecraft:stone'
                    props = state_to_props(nm, [f[x, y, z] for f in fields])
                    reg[x, y, z] = BlockState(nm, **props)
                    n += 1
    sch = Schematic(name=name, author='linux', regions={'main': reg})
    sch.save(path)
    return n

def state_accuracy(gt, pred, gt_fields, pred_fields, state_cls=STATE_CLS):
    """只统计 'GT 与预测 ID 一致, 且该 ID 为可表达有状态类(1..62)' 的体素.
    分别给出 5 个字段的准确率; 不含空气/无状态类/other, 避免全零字段虚高.
    """
    correct_id = (gt == pred)
    in_state = np.isin(gt, state_cls)
    sel = correct_id & in_state
    n_id = int(sel.sum())
    if n_id == 0:
        return [0.0] * 5, 0
    accs = []
    for f in range(5):
        accs.append(float((sel & (gt_fields[f] == pred_fields[f])).sum()) / n_id)
    return accs, n_id

# --- 验证 (idx 0 小建筑, 6 seeds 取最佳 IoU) ---
ss = np.load(SS64, mmap_mode='r')
gt_st = np.load('/home/linux/redstone_ai/dataset/v2/ss64_state.npy', mmap_mode='r')
idx = 0
gt = np.asarray(ss[idx]).astype(np.int16)
gt_st_v = np.asarray(gt_st[idx]).astype(np.int64)
gt_fields = [(gt_st_v // FIELD_MULT[f]) % STATE_DIMS[f] for f in range(5)]
print(f'GT  idx={idx} occ={(gt>0).mean()*100:.2f}% bbox={bbox(gt)}', flush=True)

best = None
for seed in range(6):
    vox, fields = gen_from_image(idx, seed=seed)
    inter = ((gt > 0) & (vox > 0)).sum(); union = ((gt > 0) | (vox > 0)).sum()
    iou = inter / max(union, 1)
    st_accs, n_id = state_accuracy(gt, vox, gt_fields, fields)
    st_acc = float(np.mean(st_accs))
    # 可表达有状态类(1..62)占比
    stated_cls = np.isin(vox, STATE_CLS)
    print(f'seed {seed}: IoU={iou:.3f} occ={(vox>0).mean()*100:.2f}% bbox={bbox(vox)} state_acc={st_acc:.3f}(n={n_id}) stated_cls={(stated_cls).mean()*100:.2f}%', flush=True)
    if best is None or iou > best[0]:
        best = (iou, vox, fields)
iou, gen, fields = best
print(f'BEST: IoU={iou:.3f}', flush=True)

np.save(f'{OUT}/gen_v3_small.npy', gen)
np.savez(f'{OUT}/gen_v3_small_state.npz', f0=fields[0], f1=fields[1], f2=fields[2], f3=fields[3], f4=fields[4])
n_blocks = to_litematic(gen, fields, 'gen_v3_small', f'{OUT}/gen_v3_small.litematic')
print(f'litematic: {n_blocks} blocks -> {OUT}/gen_v3_small.litematic', flush=True)

# GT 带 state 导出对比
n_gt = to_litematic(gt, gt_fields, 'GT_v3_small', f'{OUT}/GT_v3_small.litematic')
print(f'GT litematic: {n_gt} blocks', flush=True)

# 台阶等保留率对比 (GT vs GEN): 只统计可表达有状态类 1..62
for tag, v in [('GT', gt), ('GEN', gen)]:
    stated = np.isin(v, STATE_CLS).sum()
    total = (v > 0).sum()
    print(f'{tag}: 有state类(1-62) cells={stated}/{total} = {100.0*stated/max(total,1):.2f}%', flush=True)
print('DONE', flush=True)
