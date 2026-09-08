#!/bin/bash
DIR=/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts
KEEP=8
while true; do
  cd "$DIR" 2>/dev/null || { sleep 60; continue; }
  steps=$(ls misc_step*.pt 2>/dev/null | sed -E 's/.*step([0-9]+)\.pt/\1/' | sort -n)
  cnt=$(echo "$steps" | grep -c . 2>/dev/null)
  if [ -n "$cnt" ] && [ "$cnt" -gt "$KEEP" ]; then
    drop=$((cnt - KEEP))
    echo "$steps" | head -n "$drop" | while read -r s; do
      s2=$(echo "$s" | sed 's/^0*//')
      rm -f *step$(printf '%07d' "$s2").pt
      echo "[$(date +%H:%M:%S)] removed step $s2"
    done
  fi
  sleep 300
done
