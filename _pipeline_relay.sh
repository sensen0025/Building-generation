#!/bin/bash
# 完整流水线接力: flow v2 -> VAE v2 -> encode latent -> 更新flow v3 config -> flow v3
LOG=/home/linux/redstone_ai/logs/pipeline_relay.log
PY=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python
echo "[$(date)] PIPELINE RELAY START" >> "$LOG"

# 1) 等 flow v2 完成
while pgrep -f "train.py.*ss_flow_mc_img_v2" > /dev/null 2>&1; do sleep 120; done
echo "[$(date)] flow v2 done" >> "$LOG"

# 2) 启动 VAE v2 (独占GPU)
bash /home/linux/redstone_ai/_launch_vae_v2.sh >> "$LOG" 2>&1
echo "[$(date)] VAE v2 launched" >> "$LOG"

# 3) 等 VAE v2 完成 (50k 步)
while pgrep -f "train.py.*ss_vae_mc_v2" > /dev/null 2>&1; do sleep 300; done
echo "[$(date)] VAE v2 done" >> "$LOG"

# 4) encode v2 latent + 统计 NORM
cd /home/linux/redstone_ai
$PY _encode_v2_latent.py >> "$LOG" 2>&1
echo "[$(date)] encode v2 latent done" >> "$LOG"

# 5) 用 NORM 更新 flow v3 config
$PY - <<'PYEOF'
import json
import numpy as np
norm = np.load('/home/linux/redstone_ai/dataset/v2/latent_norm.npz')
cfg_path = '/home/linux/redstone_ai/TRELLIS/configs/flow/ss_flow_mc_v3.json'
cfg = json.load(open(cfg_path))
cfg['dataset']['args']['normalize']['mean'] = [float(x) for x in norm['mean']]
cfg['dataset']['args']['normalize']['std'] = [float(x) for x in norm['std']]
json.dump(cfg, open(cfg_path, 'w'), indent=4)
print('flow v3 NORM updated:', cfg['dataset']['args']['normalize'])
PYEOF
echo "[$(date)] flow v3 config NORM updated" >> "$LOG"

# 6) 启动 flow v3 (独占GPU, 200k 步)
cd /home/linux/redstone_ai/TRELLIS
setsid nohup env ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1 $PY train.py \
  --config configs/flow/ss_flow_mc_v3.json \
  --output_dir outputs/ss_flow_mc_img_v3 \
  --data_dir /home/linux/redstone_ai/TRELLIS/data/mc \
  > /home/linux/redstone_ai/logs/trellis_flow_v3.log 2>&1 &
echo "[$(date)] flow v3 launched PID=$!" >> "$LOG"
echo "[$(date)] PIPELINE RELAY DONE" >> "$LOG"
