#!/bin/bash
cd /home/linux/redstone_ai/TRELLIS
PY=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python
mkdir -p /home/linux/redstone_ai/logs
setsid nohup env ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1 $PY train.py \
  --config configs/vae/ss_vae_mc_v2.json \
  --output_dir outputs/ss_vae_mc_v2 \
  --data_dir /home/linux/redstone_ai/TRELLIS/data/mc_v2 \
  > /home/linux/redstone_ai/logs/vae_v2_train.log 2>&1 &
echo "VAE_PID=$!"
