from mcfix.codec import RENAMES
from mcfix.legacy import legacy_state_classes
# -*- coding: utf-8 -*-
"""v4 生成测试: 用最新 EMA ckpt 做
  (a) 纯图像条件生成 (known_mask 全 False)
  (b) bottom 掩码 inpaint 补全
并与 GT 对比, 导出 litematic 供渲染.
用法: python _gen_v4_test.py <ckpt> <out_dir> [nsamples]
"""
import os, sys, json, time
os.environ.setdefault('ATTN_BACKEND', 'sdpa')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')

import numpy as np
import torch
from PIL import Image
from litemapy import Region, BlockState, Schematic

from trellis.models import SparseStructureFlowModel, SparseStructureDecoder
from trellis.utils.mc_inpaint import sample_completion
from transformers import AutoModel

NUM_CLASSES = 128
STATE_DIMS = [7, 3, 6, 16, 3]
FIELD_MULT = [1, 7, 21, 126, 2016]
DEV = 'cuda'

CFG_PATH = '/data/redstone_v4/work/ss_flow_mc_v4_comb.json'
cfg = json.load(open(CFG_PATH))
MEAN = torch.tensor(cfg['dataset']['args']['normalize']['mean']).view(1, -1, 1, 1, 1)
STD = torch.tensor(cfg['dataset']['args']['normalize']['std']).view(1, -1, 1, 1, 1)
IMG_DIR = cfg['dataset']['args']['image_dir']

# ---- litematic 导出 (取自 _gen_v3.py, 与训练同轴序 [x,y,z]) ----
BLOCK_MAP = '/home/linux/redstone_ai/dataset/v2/block_map.npy'
VOCAB = '/home/linux/redstone_ai/dataset/BLOCK_VOCAB_FULL.txt'
block_map = np.load(BLOCK_MAP)
vocab = [l.strip() for l in open(VOCAB, encoding='utf-8') if l.strip()]


def block_name(nid):
    oi = int(block_map[nid])
    return vocab[oi] if 0 <= oi < len(vocab) else 'minecraft:stone'


def state_to_props(name, fields):
    from mcfix.legacy import legacy_properties
    packed = sum(int(v)*m for v,m in zip(fields, [1,7,21,126,2016]))
    return legacy_properties(name, packed)


def fields_from_packed(packed):
    return [(packed // FIELD_MULT[f]) % STATE_DIMS[f] for f in range(5)]


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
                    nm = RENAMES.get(nm, nm)
                    if nm == 'minecraft:air':
                        nm = 'minecraft:stone'
                    reg[x, y, z] = BlockState(nm, **state_to_props(nm, [f[x, y, z] for f in fields]))
                    n += 1
    Schematic(name=name, author='linux', regions={'main': reg}, mc_version=4189).save(path)
    return n


def img_path(idx):
    """与训练 MCFlowImageDataset 完全一致的取名 (%04d), 兼容 %05d."""
    for p in (f'{IMG_DIR}/bld_{idx:04d}.png', f'{IMG_DIR}/bld_{idx:05d}.png'):
        if os.path.exists(p):
            return p
    return None


def encode_image(p, dino):
    from mcfix.images import prepare_image, encode_image as shared_encode
    return shared_encode(dino, prepare_image(p)[None].to(DEV))


def bottom_mask(vox, frac=0.45):
    nz = np.argwhere(vox > 0)
    m = np.zeros_like(vox, dtype=bool)
    cut = int(nz[:, 1].min() + (nz[:, 1].max() - nz[:, 1].min()) * frac)
    m[:, cut:, :] = True
    return m, cut


def iou(a, b, sel=None):
    A = a > 0; B = b > 0
    if sel is not None:
        A = A & sel; B = B & sel
    return float((A & B).sum()) / max(float((A | B).sum()), 1)


def main():
    ckpt = sys.argv[1]
    out = sys.argv[2]
    nsamp = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    os.makedirs(out, exist_ok=True)

    args = cfg['models']['denoiser']['args']
    denoiser = SparseStructureFlowModel(**args).to(DEV)
    denoiser.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    denoiser.eval()
    print(f'denoiser: {os.path.basename(ckpt)}', flush=True)

    dec_ck = '/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts/decoder_step0050000.pt'
    decoder = SparseStructureDecoder(out_channels=NUM_CLASSES + sum(STATE_DIMS), latent_channels=8,
                                     num_res_blocks=2, num_res_blocks_middle=2,
                                     channels=[512, 128, 32], use_fp16=False).to(DEV)
    decoder.load_state_dict(torch.load(dec_ck, map_location='cpu', weights_only=True))
    decoder.eval()
    dino = AutoModel.from_pretrained('facebook/dinov2-large', local_files_only=True).to(DEV).eval()
    print('models loaded', flush=True)

    ss = np.load('/data/redstone_v4/train/ss64_all.npy', mmap_mode='r')
    st = np.load('/data/redstone_v4/train/ss64_state_all_rotfix.npy', mmap_mode='r')
    M = ss.shape[0]
    n_new = M - 21356
    rng = np.random.RandomState(123)
    holdout = rng.choice(np.arange(21356 + n_new // 2, M, 4), size=min(24, n_new // 4), replace=False)
    picks = list(holdout[:nsamp])
    print(f'holdout picks (same protocol as _eval_v4_inpaint): {picks}', flush=True)

    report = []
    t0 = time.time()
    for si, gi in enumerate(picks):
        ip = img_path(int(gi))
        if ip is None:
            print(f'idx {gi}: no cond image, skip', flush=True)
            continue
        gt = np.asarray(ss[gi]).astype(np.int64)
        gtp = np.asarray(st[gi]).astype(np.int64)
        cond = encode_image(ip, dino)
        nz = np.argwhere(gt > 0)
        bbox = (nz.max(0) - nz.min(0) + 1).tolist() if len(nz) else None
        print(f'--- idx {gi}  solid={len(nz)}  bbox={bbox}  occ={(gt>0).mean()*100:.2f}%', flush=True)

        # (a) 纯图像条件生成
        torch.manual_seed(1000 + si)
        gen, gen_st = sample_completion(
            denoiser, decoder, torch.randn(1, 8, 16, 16, 16, device=DEV), cond,
            torch.zeros(1, 64, 64, 64, dtype=torch.long, device=DEV),
            torch.zeros(1, 64, 64, 64, dtype=torch.bool, device=DEV),
            torch.zeros(1, 64, 64, 64, dtype=torch.long, device=DEV),
            MEAN.to(DEV), STD.to(DEV), state_classes=legacy_state_classes())
        gen = gen[0].cpu().numpy().astype(np.int64)
        gen_st = gen_st[0].cpu().numpy().astype(np.int64)

        # (b) bottom 掩码补全 (上半已知)
        mask, cut = bottom_mask(gt, 0.45)
        known = ~mask
        torch.manual_seed(2000 + si)
        inp, inp_st = sample_completion(
            denoiser, decoder, torch.randn(1, 8, 16, 16, 16, device=DEV), cond,
            torch.from_numpy(gt).long().unsqueeze(0).to(DEV),
            torch.from_numpy(known).unsqueeze(0).to(DEV),
            torch.from_numpy(gtp).long().unsqueeze(0).to(DEV),
            MEAN.to(DEV), STD.to(DEV), state_classes=legacy_state_classes())
        inp = inp[0].cpu().numpy().astype(np.int64)
        inp_st = inp_st[0].cpu().numpy().astype(np.int64)

        i_gen = iou(gt, gen)
        i_inp_masked = iou(gt, inp, sel=mask)
        i_inp_all = iou(gt, inp)
        known_kept = float(((gt > 0) == (inp > 0))[known].mean())
        print(f'    IoU full-gen={i_gen:.4f}  inpaint(masked:{mask.sum()})={i_inp_masked:.4f}  '
              f'inpaint(all)={i_inp_all:.4f}  known_kept={known_kept:.4f}', flush=True)

        known_only = np.where(known, gt, 0)
        files = {}
        for tag, v, s in (('GT', gt, gtp), ('GEN', gen, gen_st),
                          ('INP', inp, inp_st), ('KNOWN', known_only, np.where(known, gtp, 0))):
            p = f'{out}/{tag}_{gi}.litematic'
            n = to_litematic(v, fields_from_packed(s), f'{tag}_{gi}', p)
            files[tag] = {'path': p, 'blocks': n}
            print(f'    {tag}: {n} blocks -> {p}', flush=True)
        np.save(f'{out}/vox_{gi}.npy', np.stack([gt, gen, inp]).astype(np.int16))
        report.append({'idx': int(gi), 'solid_gt': int(len(nz)), 'bbox': bbox, 'cut_y': int(cut),
                       'iou_fullgen': round(i_gen, 4), 'iou_inpaint_masked': round(i_inp_masked, 4),
                       'iou_inpaint_all': round(i_inp_all, 4), 'known_kept': round(known_kept, 4),
                       'cond_img': ip, 'files': files})
        print(f'    elapsed {time.time()-t0:.0f}s', flush=True)

    summ = {
        'ckpt': ckpt,
        'n': len(report),
        'mean_iou_fullgen': round(float(np.mean([r['iou_fullgen'] for r in report])), 4),
        'mean_iou_inpaint_masked': round(float(np.mean([r['iou_inpaint_masked'] for r in report])), 4),
        'mean_iou_inpaint_all': round(float(np.mean([r['iou_inpaint_all'] for r in report])), 4),
    }
    json.dump({'summary': summ, 'samples': report}, open(f'{out}/gen_report.json', 'w'), indent=1)
    print('SUMMARY ' + json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
