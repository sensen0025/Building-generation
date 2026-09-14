# -*- coding: utf-8 -*-
"""Generate final v4 config for combined training, fixing normalize + paths + finetune_ckpt."""
import json

SRC_CFG = '/home/linux/redstone_ai/TRELLIS/configs/flow/ss_flow_mc_inpaint_v4.json'
OUT_CFG = '/data/redstone_v4/work/ss_flow_mc_v4_comb.json'

cfg = json.load(open(SRC_CFG))
ds = cfg['dataset']['args']
ds['latent_npy'] = '/data/redstone_v4/train/ss_latent_all.npy'
ds['blocks_npy'] = '/data/redstone_v4/train/ss64_all.npy'
ds['states_npy'] = '/data/redstone_v4/train/ss64_state_all.npy'
ds['image_dir'] = '/data/redstone_v4/render_all'

import numpy as np
norm = np.load('/data/redstone_v4/train/latent_norm_all.npz')
ds['normalize'] = {'mean': [float(x) for x in norm['mean']],
                   'std': [float(x) for x in norm['std']]}

tr = cfg['trainer']['args']
tr['max_steps'] = 200000
tr['lr_scheduler']['args']['T_max'] = 200000
tr['finetune_ckpt'] = {'denoiser': '/data/redstone_v4/work/v4_init.pt'}
tr['i_save'] = 2000
tr['i_sample'] = 1000000

json.dump(cfg, open(OUT_CFG, 'w'), indent=2)
print('config written:', OUT_CFG)
print('normalize mean[:3]:', ds['normalize']['mean'][:3])
