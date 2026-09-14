# -*- coding: utf-8 -*-
import os, sys, time, subprocess, glob

TRELLIS = '/home/linux/redstone_ai/TRELLIS'
CFG = '/data/redstone_v5/work/config_30k.json'
OUT_DIR = '/data/redstone_v5/geometry_ema_fixed'
LOG = '/data/redstone_v5/train_flow_30k.log'
PY = '/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python'
MAX_STEPS = 30000
MARKER = 'config_30k.json'

def last_step():
    p = os.path.join(OUT_DIR, 'log.txt')
    if not os.path.exists(p):
        return 1000
    n = 1000
    try:
        with open(p) as f:
            for line in f:
                if line.strip() and ':' in line:
                    prefix = line.split(':')[0].strip()
                    if prefix.isdigit():
                        n = int(prefix)
    except Exception:
        pass
    return n

def prune_ckpts():
    ck = os.path.join(OUT_DIR, 'ckpts')
    if not os.path.isdir(ck):
        return
    for prefix in ['denoiser_ema0.9999_step', 'denoiser_step', 'misc_step']:
        fs = sorted(glob.glob(os.path.join(ck, f'{prefix}*.pt')),
                    key=lambda p_: int(os.path.basename(p_).split('step')[-1].split('.')[0]))
        # Keep initial step 1000 + last 5 checkpoints
        for f in fs[:-5]:
            bn = os.path.basename(f)
            if 'step0001000' in bn:
                continue
            try:
                os.remove(f)
                print(f'pruned {bn}', flush=True)
            except Exception as e:
                print('prune err', e, flush=True)

def run_training():
    env = dict(os.environ)
    env.update({
        'ATTN_BACKEND': 'sdpa',
        'HF_HUB_OFFLINE': '1',
        'PYTHONPATH': '/home/linux/redstone_ai/TRELLIS:/data/redstone_v5/code'
    })
    cmd = [
        PY, 'train.py',
        '--config', CFG,
        '--output_dir', OUT_DIR,
        '--load_dir', OUT_DIR,
        '--ckpt', 'latest',
        '--data_dir', f'{TRELLIS}/data/mc',
        '--auto_retry', '3'
    ]
    logf = open(LOG, 'a')
    logf.write(f'\n===== [{time.strftime("%F %T")}] launching flow v5 30k =====\n')
    logf.flush()
    return subprocess.Popen(cmd, cwd=TRELLIS, env=env, stdout=logf, stderr=subprocess.STDOUT)

print(f'v5 supervisor start; current step: {last_step()}', flush=True)
deadline = time.time() + 24 * 3600 * 7 # 7 days
while time.time() < deadline:
    step = last_step()
    if step >= MAX_STEPS:
        print(f'training reached max steps: {step}', flush=True)
        prune_ckpts()
        break
    alive = False
    for p in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            with open(p, 'rb') as f:
                cl = f.read().decode(errors='ignore')
            if 'train.py' in cl and MARKER in cl:
                alive = True
                break
        except Exception:
            pass
    if not alive:
        prune_ckpts()
        print(f'[{time.strftime("%F %T")}] training dead at step {step}, launching...', flush=True)
        run_training()
        time.sleep(60)
    prune_ckpts()
    time.sleep(120)
print('v5 supervisor exit', flush=True)
