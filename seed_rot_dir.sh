#!/bin/bash
set -e
OUT=/home/linux/redstone_ai/TRELLIS/outputs/ss_flow_mc_rot_400k/ckpts
SRC=/home/linux/redstone_ai/TRELLIS/outputs/ss_flow_mc_img_v3_lr5e5_165k/ckpts
mkdir -p "$OUT"
cp "$SRC/denoiser_ema0.9999_step0164000.pt" "$OUT/denoiser_step0164000.pt"
cp "$SRC/denoiser_ema0.9999_step0164000.pt" "$OUT/denoiser_ema0.9999_step0164000.pt"
cp "$SRC/misc_step0164000.pt"              "$OUT/misc_step0164000.pt"
echo "=== new dir ckpts ==="
ls -la "$OUT"
