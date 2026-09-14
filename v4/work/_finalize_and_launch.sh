#!/bin/bash
# Robust finalizer: wait for render to COMPLETE (not just exit), relaunch on crash,
# then reassemble render_all and launch v4 training supervisor.
set -e
LOG=/data/redstone_v4/work/render_new.log
OUT=/data/redstone_v4/render_new
TARGET=6062
MIN_OK=$((TARGET - 30))
NODE=/home/linux/node-v22.14.0-linux-x64/bin/node
SCRIPT=/home/linux/schem_render/render_new_batch.mjs

echo "[$(date +%T)] finalizer v2 start"

# Phase 1: ensure render completes
for round in $(seq 1 60); do
  # wait for current node process to finish
  while pgrep -f "render_new_batch.mjs" > /dev/null; do
    n=$(ls $OUT 2>/dev/null | wc -l)
    echo "[$(date +%T)] render running: $n/$TARGET"
    sleep 300
  done
  n=$(ls $OUT 2>/dev/null | wc -l)
  if [ "$n" -ge "$MIN_OK" ]; then
    echo "[$(date +%T)] render complete: $n/$TARGET images"
    break
  fi
  echo "[$(date +%T)] render exited early with $n/$TARGET -> relaunch (resumable)"
  cd /home/linux/schem_render
  nohup $NODE $SCRIPT 0 999999 400 > $LOG 2>&1 &
  disown
  sleep 10
done

n=$(ls $OUT 2>/dev/null | wc -l)
echo "[$(date +%T)] final render count: $n/$TARGET"

# Phase 2: reassemble render_all
bash /data/redstone_v4/work/_assemble_render_all.sh > /data/redstone_v4/work/assemble.log 2>&1
echo "[$(date +%T)] render_all assembled: $(ls /data/redstone_v4/render_all | wc -l) links"

# Phase 3: launch supervisor (launches training)
cd /data/redstone_v4/work
nohup /home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python _rh_flow_v4_supervisor.py > supervisor.log 2>&1 < /dev/null &
disown
echo "[$(date +%T)] supervisor launched"
sleep 8
ps aux | grep -E "train.py|_rh_flow_v4_supervisor" | grep -v grep | head -3
