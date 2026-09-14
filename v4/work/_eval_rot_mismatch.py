from mcfix.legacy import legacy_state_classes
# -*- coding: utf-8 -*-
"""量化「条件图与朝向错配」的影响 (配对实验).

设计: 同一栋楼的 4 个旋转槽位 k=0,1,2,3
  - 4 个槽位拿到的是**同一张**条件图 (render_all/bld_{slot}.png -> 都是 0deg 渲染)
  - 但 GT 体素分别是 rot_y_64^k 后的结果
  => 分离出「朝向错配」这一个变量, 排除建筑差异

对每个 (j,k) 跑 bottom/side/box 三种掩码补全, 报 masked IoU.
另算参考上界: 若模型完全听从图像(输出 k=0 几何), 与 rot^k GT 的 IoU 是多少.

用法: python _eval_rot_mismatch.py <ckpt> <out_json> [n_samples]
"""
import os
import sys
import json
import time

os.environ.setdefault('ATTN_BACKEND', 'sdpa')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')

import numpy as np
import torch
from PIL import Image
from trellis.models import SparseStructureFlowModel, SparseStructureDecoder
from trellis.utils.mc_inpaint import sample_completion
from transformers import AutoModel

NUM_CLASSES = 128
STATE_DIMS = [7, 3, 6, 16, 3]
N0 = 21356
DEV = 'cuda'

cfg = json.load(open('/data/redstone_v4/work/ss_flow_mc_v4_comb.json'))
MEAN = torch.tensor(cfg['dataset']['args']['normalize']['mean']).view(1, -1, 1, 1, 1)
STD = torch.tensor(cfg['dataset']['args']['normalize']['std']).view(1, -1, 1, 1, 1)
IMG_DIR = cfg['dataset']['args']['image_dir']


def encode_image(p, dino):
    from mcfix.images import prepare_image, encode_image as shared_encode
    return shared_encode(dino, prepare_image(p)[None].to(DEV))


def masks_for(vox):
    nz = np.argwhere(vox > 0)
    m = {}
    a = np.zeros_like(vox, dtype=bool)
    cut = int(nz[:, 1].min() + (nz[:, 1].max() - nz[:, 1].min()) * 0.45)
    a[:, cut:, :] = True
    m['bottom'] = a
    b = np.zeros_like(vox, dtype=bool)
    b[:, :, :32] = True
    m['side'] = b
    c = np.zeros_like(vox, dtype=bool)
    c[20:44, 20:44, 20:44] = True
    m['box'] = c
    return m


def iou_sel(a, b, sel):
    A = (a > 0) & sel
    B = (b > 0) & sel
    return float((A & B).sum()) / max(float((A | B).sum()), 1)


def main():
    ckpt = sys.argv[1]
    out_json = sys.argv[2]
    nj = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    denoiser = SparseStructureFlowModel(**cfg['models']['denoiser']['args']).to(DEV)
    denoiser.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    denoiser.eval()
    decoder = SparseStructureDecoder(out_channels=NUM_CLASSES + sum(STATE_DIMS), latent_channels=8,
                                     num_res_blocks=2, num_res_blocks_middle=2,
                                     channels=[512, 128, 32], use_fp16=False).to(DEV)
    decoder.load_state_dict(torch.load(
        '/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts/decoder_step0050000.pt',
        map_location='cpu', weights_only=True))
    decoder.eval()
    dino = AutoModel.from_pretrained('facebook/dinov2-large', local_files_only=True).to(DEV).eval()
    print('models loaded', flush=True)

    ss = np.load('/data/redstone_v4/train/ss64_all.npy', mmap_mode='r')
    st = np.load('/data/redstone_v4/train/ss64_state_all_rotfix.npy', mmap_mode='r')
    M = ss.shape[0]
    n_new = M - N0
    n_j = n_new // 4          # 新样本个数 (6062); 槽位 = N0 + j*4 + k
    rng = np.random.RandomState(123)
    js = rng.choice(np.arange(n_j // 2, n_j), size=nj, replace=False)
    print(f'n_new_slots={n_new} n_samples={n_j}  samples j = {list(map(int, js))}', flush=True)

    rows = []
    t0 = time.time()
    for ji, j in enumerate(js):
        gt0 = np.asarray(ss[N0 + j * 4 + 0]).astype(np.int64)
        cond = None
        img_used = None
        for k in range(4):
            slot = N0 + j * 4 + k
            ip = f'{IMG_DIR}/bld_{slot:05d}.png'
            if not os.path.exists(ip):
                continue
            gt = np.asarray(ss[slot]).astype(np.int64)
            gtp = np.asarray(st[slot]).astype(np.int64)
            if cond is None or True:
                cond = encode_image(ip, dino)   # 4 个槽位同一张图, 但仍各自编码(确认一致性)
                img_used = ip
            for mk, mask in masks_for(gt).items():
                known = ~mask
                try:
                    vox, _ = sample_completion(
                        denoiser, decoder, torch.randn(1, 8, 16, 16, 16, device=DEV), cond,
                        torch.from_numpy(gt).long().unsqueeze(0).to(DEV),
                        torch.from_numpy(known).unsqueeze(0).to(DEV),
                        torch.from_numpy(gtp).long().unsqueeze(0).to(DEV),
                        MEAN.to(DEV), STD.to(DEV), state_classes=legacy_state_classes())
                    vox = vox[0].cpu().numpy().astype(np.int64)
                    iou = iou_sel(gt, vox, mask)
                    # 参考上界: 若模型完全按图像给出 k=0 几何
                    ref = iou_sel(gt, gt0, mask) if k > 0 else 1.0
                    rows.append({'j': int(j), 'k': k, 'mask': mk, 'iou': round(iou, 4),
                                 'ref_image_following': round(ref, 4),
                                 'known_kept': round(float((((gt > 0) == (vox > 0))[known]).mean()), 4)})
                except Exception as e:
                    rows.append({'j': int(j), 'k': k, 'mask': mk, 'error': str(e)[:120]})
        if (ji + 1) % 2 == 0:
            print(f'{ji+1}/{len(js)} ({time.time()-t0:.0f}s)', flush=True)

    summary = {}
    for k in range(4):
        summary[f'k={k}'] = {}
        for mk in ('bottom', 'side', 'box'):
            v = [r['iou'] for r in rows if r.get('k') == k and r.get('mask') == mk and 'iou' in r]
            rv = [r['ref_image_following'] for r in rows if r.get('k') == k and r.get('mask') == mk and 'iou' in r]
            if v:
                summary[f'k={k}'][mk] = {'mean_iou': round(float(np.mean(v)), 4), 'n': len(v),
                                         'ref_if_follow_image': round(float(np.mean(rv)), 4)}
    for k in range(4):
        v = [r['iou'] for r in rows if r.get('k') == k and 'iou' in r]
        if v:
            summary[f'k={k}']['ALL'] = round(float(np.mean(v)), 4)
    json.dump({'summary': summary, 'ckpt': ckpt, 'rows': rows}, open(out_json, 'w'), indent=1)
    print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
