import json, glob, torch, os
d = '/home/linux/redstone_ai/TRELLIS/outputs/ss_flow_mc_rot_400k/ckpts'
fs = sorted(glob.glob(os.path.join(d, 'misc_*.pt')), key=lambda p: int(os.path.basename(p).split('step')[-1].split('.')[0]))
latest = fs[-1]
print('latest misc:', os.path.basename(latest))
ck = torch.load(latest, map_location='cpu', weights_only=False)
sched = ck.get('lr_scheduler', {})
print('lr_scheduler keys:', list(sched.keys()) if isinstance(sched, dict) else type(sched))
if isinstance(sched, dict):
    for k in ('T_max', 'eta_min', 'last_epoch'):
        if k in sched:
            print(f'  {k} = {sched[k]}')
    opt = ck.get('optimizer', {})
    pgs = opt.get('param_groups', []) if isinstance(opt, dict) else []
    if pgs:
        print('  optimizer param_group[0] lr =', pgs[0].get('lr'), 'initial_lr =', pgs[0].get('initial_lr'))
print('ckpt step:', ck.get('step'))
