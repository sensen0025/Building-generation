
import json, numpy as np
V2 = '/home/linux/redstone_ai/dataset/v2'
CFG = '/home/linux/redstone_ai/TRELLIS/configs/flow/ss_flow_mc_v3.json'
ln = np.load(f'{V2}/latent_norm.npz')
print('NEW mean:', ln['mean'].tolist(), flush=True)
print('NEW std :', ln['std'].tolist(), flush=True)
cfg = json.load(open(CFG, encoding='utf-8'))
nm = cfg['dataset']['args']['normalize']
print('CFG before mean:', nm['mean'][:4], '...', flush=True)
with open(CFG + '.bak_badnorm', 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)
nm['mean'] = [float(x) for x in ln['mean']]
nm['std'] = [float(x) for x in ln['std']]
with open(CFG, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)
print('CFG normalize updated', flush=True)
print('DONE', flush=True)
