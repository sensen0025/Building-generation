"""按 train.py 方式构建 trainer, 复现 snapshot 崩溃, 打印完整 traceback."""
import os, sys, json, traceback
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
import torch
import trellis.models as models
import trellis.datasets as datasets
import trellis.trainers as trainers
from easydict import EasyDict as edict

BASE = '/home/linux/redstone_ai/TRELLIS'
cfg = json.load(open(f'{BASE}/configs/flow/ss_flow_mc_v3.json'))

data_dir = '/home/linux/redstone_ai/TRELLIS/data/mc'
dataset = getattr(datasets, cfg['dataset']['name'])(data_dir, **cfg['dataset']['args'])
model_dict = {
    name: getattr(models, m['name'])(**m['args']).cuda()
    for name, m in cfg['models'].items()
}
targs = dict(cfg['trainer']['args'])
trainer = getattr(trainers, cfg['trainer']['name'])(
    model_dict, dataset,
    **targs,
    output_dir=f'{BASE}/outputs/ss_flow_mc_img_v3',
    load_dir=None, step=None,
)
print('trainer ready, snapshot...', flush=True)
try:
    trainer.snapshot(num_samples=4, batch_size=2)
    print('SNAPSHOT OK', flush=True)
except Exception:
    traceback.print_exc()
    print('SNAPSHOT FAILED', flush=True)
