from mcfix.legacy import legacy_state_classes
# -*- coding: utf-8 -*-
"""Stage F: v4 inpaint completion evaluation.
For held-out samples: build 3 mask types (bottom/side/box), sample_completion,
measure IoU of completed-unknown-region vs GT + boundary continuity.
Usage: python3 _eval_v4_inpaint.py <ckpt_path> <out_json> [nsamples]
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault('ATTN_BACKEND', 'sdpa')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')

from trellis.models import SparseStructureFlowModel, SparseStructureDecoder
from trellis.pipelines.samplers import FlowEulerCfgSampler
from trellis.utils.mc_inpaint import sample_known_mask, preserve_known, sample_completion
from transformers import AutoModel
from PIL import Image

NUM_CLASSES = 128
STATE_DIMS = [7, 3, 6, 16, 3]
FIELD_MULT = [1, 7, 21, 126, 2016]
TRELLIS = '/home/linux/redstone_ai/TRELLIS'
COMB = '/data/redstone_v4'
DEVICE = 'cuda'

import json as _json
CFG_PATH = '/data/redstone_v4/work/ss_flow_mc_v4_comb.json'
cfg = _json.load(open(CFG_PATH))


def load_norm():
    nm = cfg['dataset']['args']['normalize']
    return (torch.tensor(nm['mean']).reshape(-1, 1, 1, 1).float(),
            torch.tensor(nm['std']).reshape(-1, 1, 1, 1).float())


MEAN, STD = load_norm()


def latest_ema(prefix, d):
    files = [f for f in os.listdir(d) if f.startswith(prefix)]
    if not files:
        return None
    files.sort(key=lambda f: int(f.split('step')[-1].split('.')[0]))
    return os.path.join(d, files[-1])


def encode_image(img_path, dino):
    from mcfix.images import prepare_image, encode_image as shared_encode
    return shared_encode(dino, prepare_image(img_path)[None].to(DEVICE))


def masks_for(vox):
    """three completion masks (True = unknown to complete)."""
    nz = np.argwhere(vox > 0)
    masks = {}
    m1 = np.zeros_like(vox, dtype=bool)
    cut = int(nz[:, 1].min() + (nz[:, 1].max() - nz[:, 1].min()) * 0.45)
    m1[:, cut:, :] = True
    masks['bottom'] = m1
    m2 = np.zeros_like(vox, dtype=bool)
    m2[:, :, :32] = True
    masks['side'] = m2
    m3 = np.zeros_like(vox, dtype=bool)
    x0, x1 = 20, 44
    m3[x0:x1, x0:x1, x0:x1] = True
    masks['box'] = m3
    return masks


def fields_from_packed(stv):
    return [(stv // FIELD_MULT[f]) % STATE_DIMS[f] for f in range(5)]


def main():
    if '--allow-training-diagnostic' not in sys.argv:
        raise RuntimeError('This historical array was used in training. Use python -m mcfix.evaluate for a held-out evaluation, or explicitly pass --allow-training-diagnostic')
    sys.argv.remove('--allow-training-diagnostic')
    ckpt_path = sys.argv[1]
    out_json = sys.argv[2]
    nsamp = int(sys.argv[3]) if len(sys.argv) > 3 else 24

    ss = np.load(f'{COMB}/train/ss64_all.npy', mmap_mode='r')
    stv = np.load(f'{COMB}/train/ss64_state_all_rotfix.npy', mmap_mode='r')

    # held-out: last 3% (not used in training shuffles? trainer sees all; approximate with tail slices)
    # pick N_new tail samples (these belong to new dataset tail)
    M = ss.shape[0]
    n_new = M - 21356
    assert n_new > 0
    rng = np.random.RandomState(123)
    holdout = rng.choice(np.arange(21356 + n_new // 2, M, 4), size=min(nsamp, n_new // 4), replace=False)

    dev = DEVICE
    args = cfg['models']['denoiser']['args']
    denoiser = SparseStructureFlowModel(**args).to(dev)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    denoiser.load_state_dict(sd)
    denoiser.eval()
    vae_ck = '/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts'
    dec_ckpt = latest_ema('decoder', vae_ck) or os.path.join(vae_ck, 'decoder_step0050000.pt')
    decoder = SparseStructureDecoder(out_channels=NUM_CLASSES + sum(STATE_DIMS), latent_channels=8,
                                     num_res_blocks=2, num_res_blocks_middle=2, channels=[512, 128, 32],
                                     use_fp16=False).to(dev)
    decoder.load_state_dict(torch.load(dec_ckpt, map_location='cpu', weights_only=True))
    decoder.eval()
    dino = AutoModel.from_pretrained('facebook/dinov2-large', local_files_only=True).to(dev).eval()

    img_dir = cfg['dataset']['args']['image_dir']
    results = []
    t0 = time.time()
    for hi, gi in enumerate(holdout):
        gt = np.asarray(ss[gi])
        gtp = np.asarray(stv[gi])
        # render image path: images named by ORIGINAL render idx -> build mapping:
        # idx in combined = position; images stored render_all/bld_%05d.png with SAME combined idx
        img_path = f'{img_dir}/bld_{gi:05d}.png'
        if not os.path.exists(img_path):
            continue
        cond = encode_image(img_path, dino)
        masks = masks_for(gt)
        for mk, mask in masks.items():
            known = ~mask
            try:
                vox, fields = sample_completion(
                    denoiser, decoder, torch.randn(1, 8, 16, 16, 16, device=dev),
                    cond, torch.from_numpy(gt).long().unsqueeze(0).to(dev),
                    torch.from_numpy(known).unsqueeze(0).to(dev),
                    torch.from_numpy(gtp).long().unsqueeze(0).to(dev),
                    MEAN.to(dev), STD.to(dev), state_classes=legacy_state_classes())
                vox = vox[0].cpu().numpy()
                # preserve known exactly (helper does it, double-check)
                inter = ((gt > 0) & (vox > 0) & mask).sum()
                union = (((gt > 0) & mask) | ((vox > 0) & mask)).sum()
                iou = float(inter) / max(float(union), 1)
                # boundary continuity: fraction of mask-face cells adjacent to known solids that are solid
                from mcfix.metrics import completion_metrics
                metrics = completion_metrics(gt, vox, gtp, fields[0].cpu().numpy(), known)
                results.append({'idx': int(gi), 'mask': mk, 'iou': round(iou, 4), **metrics})
            except Exception as e:
                results.append({'idx': int(gi), 'mask': mk, 'error': str(e)[:120]})
        if (hi + 1) % 4 == 0:
            print(f'{hi+1}/{len(holdout)} ({time.time()-t0:.0f}s)', flush=True)
    json.dump(results, open(out_json, 'w'), indent=1)
    # summary
    by = {}
    for r in results:
        if 'iou' in r:
            by.setdefault(r['mask'], []).append(r['iou'])
    summary = {k: {'mean_iou': round(sum(v) / len(v), 4), 'n': len(v)} for k, v in by.items()}
    json.dump({'summary': summary, 'ckpt': ckpt_path, 'evaluation_kind': 'training_diagnostic_NOT_holdout'}, open(out_json + '.summary', 'w'), indent=1)
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
