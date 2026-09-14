# -*- coding: utf-8 -*-
"""准备并(可选)启动「修复后续训」: 从 200k EMA 权重出发, 用修复后的数据继续训练.

与项目既有约定一致:
  - finetune_ckpt.denoiser = 200k EMA (只加载 denoiser 权重)
  - 新 cosine 周期 (T_max = max_steps), 从 5e-5 降到 0
  - output_dir 换成 out_v4_rotfix, 不覆盖原 200k 产物
  - 数据: states 指向 ss64_state_all_rotfix.npy, 图像 render_all (已切到旋转图)
用法:
  python _prep_cont_train.py <steps>          # 只生成配置与守护脚本
  python _prep_cont_train.py <steps> --launch # 生成并启动
"""
import json
import os
import sys

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
LAUNCH = '--launch' in sys.argv
W = '/data/redstone_v4/work'
BASE_CFG = f'{W}/ss_flow_mc_v4_rotfix.json'
CFG = f'{W}/ss_flow_mc_v4_rotfix_cont.json'
OUT_DIR = '/data/redstone_v4/out_v4_rotfix'
CKPT = '/data/redstone_v4/out_v4/ckpts/denoiser_ema0.9999_step0200000.pt'
SUP = f'{W}/_rh_rotfix_supervisor.py'
LOG = '/home/linux/redstone_ai/logs/trellis_flow_v4_rotfix.log'

c = json.load(open(BASE_CFG))
c['trainer']['args']['max_steps'] = STEPS
c['trainer']['args']['lr_scheduler']['args']['T_max'] = STEPS
c['trainer']['args']['finetune_ckpt'] = {'denoiser': CKPT}
json.dump(c, open(CFG, 'w'), indent=2)

assert os.path.exists(CKPT), f'缺少起始 ckpt {CKPT}'
assert os.path.exists('/data/redstone_v4/train/ss64_state_all_rotfix.npy')

src = open(f'{W}/_rh_flow_v4_supervisor.py').read()
src = src.replace("CFG = '/data/redstone_v4/work/ss_flow_mc_v4_comb.json'", f"CFG = '{CFG}'")
src = src.replace("OUT_DIR = '/data/redstone_v4/out_v4'", f"OUT_DIR = '{OUT_DIR}'")
src = src.replace("LOG = '/home/linux/redstone_ai/logs/trellis_flow_v4.log'", f"LOG = '{LOG}'")
src = src.replace('MAX_STEPS = 200000', f'MAX_STEPS = {STEPS}')
src = src.replace("MARKER = 'ss_flow_mc_v4_comb'", "MARKER = 'ss_flow_mc_v4_rotfix_cont'")
# 原守护脚本只在「结束/崩溃」时裁剪 ckpt, 长跑会把磁盘吃满 -> 改成每轮都裁(保留每类最后 5 个)
src = src.replace("    time.sleep(300)\nprint('v4 supervisor exit'",
                  "    prune_ckpts()\n    time.sleep(300)\nprint('v4 supervisor exit'")
assert 'prune_ckpts()\n    time.sleep(300)' in src, '周期裁剪注入失败'
open(SUP, 'w').write(src)

print('config :', CFG)
print('  max_steps =', STEPS, ' T_max =', STEPS, ' init =', os.path.basename(CKPT))
print('  states_npy =', c['dataset']['args']['states_npy'])
print('  image_dir  =', c['dataset']['args']['image_dir'])
print('supervisor:', SUP, ' OUT_DIR =', OUT_DIR)
print('MARKER 校验:', "MARKER = 'ss_flow_mc_v4_rotfix_cont'" in open(SUP).read())

if LAUNCH:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.system(f"cd {W} && setsid bash -c '/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python {SUP} > {W}/rotfix_supervisor.log 2>&1' </dev/null >/dev/null 2>&1 &")
    print('LAUNCHED supervisor (dir=%s steps=%d)' % (OUT_DIR, STEPS))
