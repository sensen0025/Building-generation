#!/bin/bash
cd /home/linux/redstone_ai/TRELLIS
PY=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python
export ATTN_BACKEND=sdpa
export HF_HUB_OFFLINE=1
echo "=== tryrun ==="
$PY train.py --config configs/vae/ss_vae_mc_v2.json --output_dir outputs/ss_vae_mc_v2 --data_dir /home/linux/redstone_ai/TRELLIS/data/mc_v2 --tryrun 2>&1 | tail -20
echo "TRYRUN_EXIT=$?"
