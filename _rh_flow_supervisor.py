# -*- coding: utf-8 -*-
"""flow v3 重启 + 监督(独立 nohup): 训练进程死后自动重启, 记录日志."""
import os, sys, time, subprocess, glob, signal

TRELLIS = '/home/linux/redstone_ai/TRELLIS'
OUT_DIR = f'{TRELLIS}/outputs/ss_flow_mc_img_v3'
LOG = '/home/linux/redstone_ai/logs/trellis_flow_v3.log'
PY = '/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python'
MAX_STEPS = 200000

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

def run_training():
    env = dict(os.environ)
    env.update({'ATTN_BACKEND': 'sdpa', 'HF_HUB_OFFLINE': '1'})
    # 防磁盘满: 启动前清理 ckpt 只留最近 5 个
    try:
        ck = os.path.join(OUT_DIR, 'ckpts')
        if os.path.isdir(ck):
            for prefix in ['denoiser_ema0.9999_step', 'denoiser_step', 'misc_step']:
                fs = sorted(glob.glob(os.path.join(ck, f'{prefix}*.pt')),
                            key=lambda p: int(os.path.basename(p).split('step')[-1].split('.')[0]))
                for f in fs[:-5]:
                    os.remove(f)
                    print(f'pruned {os.path.basename(f)}', flush=True)
    except Exception as e:
        print('prune err', e, flush=True)
    cmd = [PY, 'train.py', '--config', 'configs/flow/ss_flow_mc_v3.json',
           '--output_dir', OUT_DIR,
           '--data_dir', f'{TRELLIS}/data/mc',
           '--auto_retry', '3']
    logf = open(LOG, 'a')
    logf.write(f'\n===== [{time.strftime("%F %T")}] launching flow v3 =====\n')
    logf.flush()
    proc = subprocess.Popen(cmd, cwd=TRELLIS, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return proc

print('supervisor start; current step', last_step(), flush=True)
deadline = time.time() + 30 * 3600  # 30h 上限
while time.time() < deadline:
    step = last_step()
    if step >= MAX_STEPS:
        print('training reached max steps', flush=True)
        break
    procs = [p for p in glob.glob(f'/proc/[0-9]*') if os.path.exists(p + '/cmdline')]
    alive = False
    for p in procs:
        try:
            with open(p + '/cmdline', 'rb') as f:
                cl = f.read().decode(errors='ignore')
            if 'train.py' in cl and 'ss_flow_mc_v3' in cl:
                alive = True
                break
        except Exception:
            pass
    if not alive:
        print(f'[{time.strftime("%F %T")}] training dead at step {step}, relaunching', flush=True)
        run_training()
        time.sleep(60)
    time.sleep(300)
print('supervisor exit', flush=True)
