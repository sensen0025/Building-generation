# -*- coding: utf-8 -*-
"""核对 flow v3 config 的 normalize 与 latent_norm.npz 一致; 旧输出目录状态."""
import json, os
import numpy as np

cfg = json.load(open('/home/linux/redstone_ai/TRELLIS/configs/flow/ss_flow_mc_v3.json'))
norm = cfg['dataset']['args']['normalize']
npz = np.load('/home/linux/redstone_ai/dataset/v2/latent_norm.npz')
print('config mean:', norm['mean'])
print('npz   mean:', npz['mean'].tolist())
print('config std :', norm['std'])
print('npz   std :', npz['std'].tolist())
print('match:', np.allclose(norm['mean'], npz['mean']) and np.allclose(norm['std'], npz['std']))
print('image_dir:', cfg['dataset']['args'].get('image_dir'))
print('latent_npy:', cfg['dataset']['args'].get('latent_npy'))
o = '/home/linux/redstone_ai/TRELLIS/outputs'
print('outputs:', os.listdir(o))
