# -*- coding: utf-8 -*-
"""v4 training supervisor: relaunch on death, prune ckpts (keep last 5), stop at 200k."""
import os, sys, time, subprocess, glob

TRELLIS = '/home/linux/redstone_ai/TRELLIS'
CFG = '/data/redstone_v4/work/ss_flow_mc_v4_rotfix_cont.json'
OUT_DIR = '/data/redstone_v4/out_v4_rotfix'
LOG = '/home/linux/redstone_ai/logs/trellis_flow_v4_rotfix.log'
PY = '/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python'
MAX_STEPS = 50000
MARKER = 'ss_flow_mc_v4_rotfix_cont'


def last_step():
    p = os.path.join(OUT_DIR, 'log.txt')
    if not os.path.exists(p):
        return 0
    n = 0
    try:
        with open(p) as f:
            for line in f:
                if line.strip():
                    n = int(line.split(':')[0])
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
        for f in fs[:-5]:
            try:
                os.remove(f)
                print(f'pruned {os.path.basename(f)}', flush=True)
            except Exception as e:
                print('prune err', e, flush=True)


def run_training():
    env = dict(os.environ)
    env.update({'ATTN_BACKEND': 'sdpa', 'HF_HUB_OFFLINE': '1'})
    cmd = [PY, 'train.py', '--config', CFG,
           '--output_dir', OUT_DIR,
           '--data_dir', f'{TRELLIS}/data/mc',
           '--auto_retry', '3']
    logf = open(LOG, 'a')
    logf.write(f'\n===== [{time.strftime("%F %T")}] launching flow v4 comb =====\n')
    logf.flush()
    return subprocess.Popen(cmd, cwd=TRELLIS, env=env, stdout=logf, stderr=subprocess.STDOUT)


if os.path.exists('/data/redstone_v4/RETIRED.json'):
    print('v4 training retired; see /data/redstone_v5/README.md', flush=True)
    sys.exit(0)
print('v4 supervisor start; current step', last_step(), flush=True)
deadline = time.time() + 24 * 3600 * 14
while time.time() < deadline:
    step = last_step()
    if step >= MAX_STEPS:
        print('training reached max steps', flush=True)
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
        print(f'[{time.strftime("%F %T")}] training dead at step {step}, relaunching', flush=True)
        run_training()
        time.sleep(60)
    prune_ckpts()
    time.sleep(300)
print('v4 supervisor exit', flush=True)
