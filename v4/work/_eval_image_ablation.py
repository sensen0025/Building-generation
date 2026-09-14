from mcfix.legacy import legacy_state_classes
# -*- coding: utf-8 -*-
"""消融实验: 模型到底用了多少条件图?

在 k=0 (朝向一致) 槽位上跑同一套补全, 只改条件图:
  A. 正确图 (与训练/推理一致)
  B. 全零 tokens (相当于没有图像信息)
  C. 错图 (换另一栋楼的图)
若 A 与 B 差距很小 => 模型基本没用图像 (被 40% 矛盾监督教成"忽略图").
若 A >> B     => 图像确实在被使用, 错配影响仅限局部.

用法: python _eval_image_ablation.py <ckpt> <out_json> [n_samples]
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
    ckpt, out_json = sys.argv[1], sys.argv[2]
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
    n_j = (ss.shape[0] - N0) // 4
    rng = np.random.RandomState(123)
    js = rng.choice(np.arange(n_j // 2, n_j), size=nj, replace=False)

    rows = []
    conds = {}
    t0 = time.time()
    for ji, j in enumerate(js):
        slot = N0 + int(j) * 4          # k=0 槽位 (朝向一致)
        ip = f'{IMG_DIR}/bld_{slot:05d}.png'
        gt = np.asarray(ss[slot]).astype(np.int64)
        gtp = np.asarray(st[slot]).astype(np.int64)
        c_correct = encode_image(ip, dino)
        c_zero = torch.zeros_like(c_correct)
        # 错图: 用另一栋楼 (下一个样本) 的条件图
        j2 = int(js[(ji + 1) % len(js)])
        c_wrong = encode_image(f'{IMG_DIR}/bld_{N0 + j2 * 4:05d}.png', dino)
        c_zero = c_zero.float()
        for mk, mask in masks_for(gt).items():
            known = ~mask
            for tag, cond in (('correct_img', c_correct), ('zero_cond', c_zero), ('wrong_img', c_wrong)):
                try:
                    torch.manual_seed(4242)
                    vox, _ = sample_completion(
                        denoiser, decoder, torch.randn(1, 8, 16, 16, 16, device=DEV), cond,
                        torch.from_numpy(gt).long().unsqueeze(0).to(DEV),
                        torch.from_numpy(known).unsqueeze(0).to(DEV),
                        torch.from_numpy(gtp).long().unsqueeze(0).to(DEV),
                        MEAN.to(DEV), STD.to(DEV), state_classes=legacy_state_classes())
                    rows.append({'j': int(j), 'mask': mk, 'cond': tag,
                                 'iou': round(iou_sel(gt, vox[0].cpu().numpy().astype(np.int64), mask), 4)})
                except Exception as e:
                    rows.append({'j': int(j), 'mask': mk, 'cond': tag, 'error': str(e)[:120]})
        if (ji + 1) % 2 == 0:
            print(f'{ji+1}/{len(js)} ({time.time()-t0:.0f}s)', flush=True)

    summary = {}
    for tag in ('correct_img', 'zero_cond', 'wrong_img'):
        summary[tag] = {}
        for mk in ('bottom', 'side', 'box', 'ALL'):
            v = [r['iou'] for r in rows if r['cond'] == tag and (mk == 'ALL' or r['mask'] == mk) and 'iou' in r]
            if v:
                summary[tag][mk] = round(float(np.mean(v)), 4)
    json.dump({'summary': summary, 'ckpt': ckpt, 'rows': rows}, open(out_json, 'w'), indent=1)
    print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
