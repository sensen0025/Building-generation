"""VAE v2 重建测试: GT voxel -> encoder(45k) -> latent -> decoder(ema 45k) -> id + 5 state 字段.
统计 IoU / id 准确率 / state 字段准确率, 导出带 state 的 litematic + GT 对比.
在训练进行中运行(仅 ~1.5GB 显存, 不干扰).
"""
import os, sys, glob
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
import numpy as np
import torch
import torch.nn.functional as F
from litemapy import Region, BlockState, Schematic

from trellis.models import SparseStructureEncoder, SparseStructureDecoder

BASE = '/home/linux/redstone_ai'
CKPT = f'{BASE}/TRELLIS/outputs/ss_vae_mc_v2/ckpts'
SS64 = f'{BASE}/dataset/v2/ss64.npy'
ST64 = f'{BASE}/dataset/v2/ss64_state.npy'
BLOCK_MAP = f'{BASE}/dataset/v2/block_map.npy'
VOCAB = f'{BASE}/dataset/BLOCK_VOCAB_FULL.txt'
OUT = f'{BASE}/gen_outputs'
NUM_CLASSES = 128
STATE_DIMS = [7, 3, 6, 16, 3]
FIELD_MULT = [1, 7, 21, 126, 2016]
# 真实有状态类 (v2 数据映射实测: 1..62; 63 死槽; 64..71 无状态 plain; 127=other 语义不可解释)
STATE_CLS = list(range(1, 63))

dev = 'cuda'
print(f'device: {dev}', flush=True)

def latest(prefix, ema=False):
    pat = f'{prefix}_ema*.pt' if ema else f'{prefix}_step*.pt'
    files = glob.glob(os.path.join(CKPT, pat))
    if ema:
        files = [f for f in files if 'ema' in os.path.basename(f)]
    files = [f for f in files if 'ema' not in os.path.basename(f)] if not ema else files
    return max(files, key=lambda f: int(os.path.basename(f).split('step')[-1].split('.')[0]))

enc_ckpt = latest('encoder', ema=False)
dec_ckpt = latest('decoder', ema=True)
print('enc ckpt:', os.path.basename(enc_ckpt), flush=True)
print('dec ckpt:', os.path.basename(dec_ckpt), flush=True)

enc = SparseStructureEncoder(in_channels=NUM_CLASSES, latent_channels=8, num_res_blocks=2,
                             num_res_blocks_middle=2, channels=[32, 128, 512], use_fp16=False).to(dev)
enc.load_state_dict(torch.load(enc_ckpt, map_location='cpu', weights_only=True))
enc.eval()
dec = SparseStructureDecoder(out_channels=NUM_CLASSES + sum(STATE_DIMS), latent_channels=8,
                             num_res_blocks=2, num_res_blocks_middle=2, channels=[512, 128, 32],
                             use_fp16=False).to(dev)
dec.load_state_dict(torch.load(dec_ckpt, map_location='cpu', weights_only=True))
dec.eval()

block_map = np.load(BLOCK_MAP)
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
    return {k: str(v) for k, v in props.items()}

def recon(idx):
    gt = np.asarray(np.load(SS64, mmap_mode='r')[idx]).astype(np.int64)
    t = torch.from_numpy(gt).unsqueeze(0).to(dev)
    x = F.one_hot(t, num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
    with torch.no_grad():
        z = enc(x, sample_posterior=False)
        logits = dec(z)  # [1,163,64,64,64]
        id_logits = logits[:, :NUM_CLASSES]
        st_logits = logits[:, NUM_CLASSES:NUM_CLASSES + sum(STATE_DIMS)]
        vox = id_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int16)
        fields = []
        off = 0
        for dim in STATE_DIMS:
            fields.append(st_logits[:, off:off + dim].argmax(1).squeeze(0).cpu().numpy().astype(np.int16))
            off += dim
    return z, vox, fields

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
    """只统计 'GT 与预测 ID 一致, 且该 ID 为可表达有状态类(1..62)' 的体素."""
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

# GT state 是打包 int, 解出 5 字段
ss = np.load(SS64, mmap_mode='r')
st = np.load(ST64, mmap_mode='r')

os.makedirs(OUT, exist_ok=True)
results = []
for idx in range(6):
    gt = np.asarray(ss[idx]).astype(np.int16)
    gt_st = np.asarray(st[idx]).astype(np.int64)
    gt_fields = [(gt_st // FIELD_MULT[f]) % STATE_DIMS[f] for f in range(5)]
    z, vox, fields = recon(idx)
    inter = ((gt > 0) & (vox > 0)).sum(); union = ((gt > 0) | (vox > 0)).sum()
    iou = inter / max(union, 1)
    id_acc = float(((gt > 0) & (gt == vox)).sum()) / max(int((gt > 0).sum()), 1)
    st_accs, n_id = state_accuracy(gt, vox, gt_fields, fields)
    st_acc = float(np.mean(st_accs))
    occ_gt = (gt > 0).mean() * 100
    occ_pd = (vox > 0).mean() * 100
    # 生成中有 state 的类占比 (1..62)
    stated_pd = np.isin(vox, STATE_CLS).mean() * 100
    z_abs = float(z.abs().mean())
    print(f'idx={idx}: IoU={iou:.3f} id_acc={id_acc:.3f} state_acc={st_acc:.3f}(n={n_id}) '
          f'occ {occ_gt:.1f}%->{occ_pd:.1f}% bbox_gt={bbox(gt)} bbox_pd={bbox(vox)} '
          f'stated_pd={stated_pd:.2f}% |z|_mean={z_abs:.4f}', flush=True)
    results.append({'idx': idx, 'iou': round(iou, 4), 'id_acc': round(id_acc, 4),
                    'state_acc': round(st_acc, 4), 'n_id': n_id})

# 导出 idx0 与 idx1 的 litematic + GT
for idx in [0, 1]:
    gt = np.asarray(ss[idx]).astype(np.int16)
    gt_st = np.asarray(st[idx]).astype(np.int64)
    gt_fields = [(gt_st // FIELD_MULT[f]) % STATE_DIMS[f] for f in range(5)]
    _, vox, fields = recon(idx)
    n_pd = to_litematic(vox, fields, f'recon_v2_{idx}', f'{OUT}/recon_v2_{idx}.litematic')
    n_gt = to_litematic(gt, gt_fields, f'GT_v2_{idx}', f'{OUT}/GT_v2_{idx}.litematic')
    print(f'litematic idx={idx}: pred {n_pd} blocks -> {OUT}/recon_v2_{idx}.litematic, GT {n_gt} blocks', flush=True)

print('RESULTS:', results, flush=True)
print('DONE', flush=True)
