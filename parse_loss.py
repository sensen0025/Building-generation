#!/usr/bin/env python3
"""Parse flow training logs -> compact npz for loss-trend plotting."""
import json
import numpy as np


def parse(path):
    steps, mse, lr = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            try:
                step_s, js = line.split(': ', 1)
                d = json.loads(js)
            except Exception:
                continue
            steps.append(int(step_s))
            loss = d.get('loss', {})
            mse.append(loss.get('mse', loss.get('loss', float('nan'))))
            st = d.get('status', {})
            lr.append(st.get('lr', float('nan')))
    return np.array(steps), np.array(mse, dtype=float), np.array(lr, dtype=float)


old = parse('/home/linux/redstone_ai/TRELLIS/outputs/ss_flow_mc_img_v3_lr5e5_165k/log.txt')
new = parse('/home/linux/redstone_ai/TRELLIS/outputs/ss_flow_mc_rot_400k/log.txt')
print('old steps:', old[0][0], '..', old[0][-1], 'n=', len(old[0]))
print('new steps:', new[0][0], '..', new[0][-1], 'n=', len(new[0]))
np.savez('/home/linux/redstone_ai/loss_data.npz',
         old_step=old[0], old_mse=old[1], old_lr=old[2],
         new_step=new[0], new_mse=new[1], new_lr=new[2])
print('saved loss_data.npz')
