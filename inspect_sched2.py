import json, glob, torch, os
d = '/home/linux/redstone_ai/TRELLIS/outputs/ss_flow_mc_rot_400k/ckpts'
fs = sorted(glob.glob(os.path.join(d, 'misc_*.pt')), key=lambda p: int(os.path.basename(p).split('step')[-1].split('.')[0]))
for latest in fs[-3:]:
    ck = torch.load(latest, map_location='cpu', weights_only=False)
    s = ck.get('lr_scheduler', {})
    print(os.path.basename(latest), 'step', ck.get('step'),
          'T_max', s.get('T_max'), 'base_lrs', s.get('base_lrs'),
          'last_epoch', s.get('last_epoch'), '_last_lr', s.get('_last_lr'))
