from mcfix.legacy import legacy_state_classes
# -*- coding: utf-8 -*-
"""纯生成模式下的条件图消融: 图像到底有没有用?

不提供任何已知体素 (known_mask 全 False), 只靠条件图生成整栋建筑:
  A. 正确图   B. 全零 tokens (无图像)   C. 错图 (别的楼)
这能直接回答: 朝向错配是否把模型的「图像条件能力」练废了.

用法: python _eval_gen_ablation.py <ckpt> <out_json> [n_samples]
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


def occ_iou(a, b):
    A = a > 0
    B = b > 0
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
    n_j = (ss.shape[0] - N0) // 4
    rng = np.random.RandomState(123)
    js = rng.choice(np.arange(n_j // 2, n_j), size=nj, replace=False)

    zero_blocks = torch.zeros(1, 64, 64, 64, dtype=torch.long, device=DEV)
    no_mask = torch.zeros(1, 64, 64, 64, dtype=torch.bool, device=DEV)  # 全 False => 全未知

    rows = []
    t0 = time.time()
    for ji, j in enumerate(js):
        slot = N0 + int(j) * 4
        gt = np.asarray(ss[slot]).astype(np.int64)
        conds = {
            'correct_img': encode_image(f'{IMG_DIR}/bld_{slot:05d}.png', dino),
            'wrong_img': encode_image(
                f'{IMG_DIR}/bld_{N0 + int(js[(ji + 1) % len(js)]) * 4:05d}.png', dino),
        }
        conds['zero_cond'] = torch.zeros_like(conds['correct_img'])
        for tag, cond in conds.items():
            try:
                torch.manual_seed(777)
                vox, _ = sample_completion(
                    denoiser, decoder, torch.randn(1, 8, 16, 16, 16, device=DEV), cond,
                    zero_blocks, no_mask, zero_blocks, MEAN.to(DEV), STD.to(DEV), state_classes=legacy_state_classes())
                v = vox[0].cpu().numpy().astype(np.int64)
                rows.append({'j': int(j), 'cond': tag, 'iou': round(occ_iou(gt, v), 4),
                             'solid': int((v > 0).sum()), 'gt_solid': int((gt > 0).sum())})
            except Exception as e:
                rows.append({'j': int(j), 'cond': tag, 'error': str(e)[:120]})
        if (ji + 1) % 3 == 0:
            print(f'{ji+1}/{len(js)} ({time.time()-t0:.0f}s)', flush=True)

    summary = {}
    for tag in ('correct_img', 'zero_cond', 'wrong_img'):
        v = [r['iou'] for r in rows if r['cond'] == tag and 'iou' in r]
        s = [r['solid'] for r in rows if r['cond'] == tag and 'iou' in r]
        g = [r['gt_solid'] for r in rows if r['cond'] == tag and 'iou' in r]
        if v:
            summary[tag] = {'mean_iou': round(float(np.mean(v)), 4), 'n': len(v),
                            'mean_solid': int(np.mean(s)), 'mean_gt_solid': int(np.mean(g))}
    json.dump({'summary': summary, 'ckpt': ckpt, 'rows': rows}, open(out_json, 'w'), indent=1)
    print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
