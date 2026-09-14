#!/bin/bash
# Full autonomous orchestrator for v4 pipeline.
# 1) wait render complete (relaunch on crash)  2) assemble render_all
# 3) step0 baseline eval (v4_init.pt)          4) launch supervisor (200k training)
# 5) wait training done                         6) final eval (latest ema ckpt)
set -e
LOG=/data/redstone_v4/work/render_new.log
OUT=/data/redstone_v4/render_new
TARGET=6062
MIN_OK=$((TARGET - 30))
NODE=/home/linux/node-v22.14.0-linux-x64/bin/node
SCRIPT=/home/linux/schem_render/render_new_batch.mjs
PY=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python
WORK=/data/redstone_v4/work

echo "[$(date +%T)] orchestrator start" | tee -a $WORK/orchestrator.log

# ---- Phase 1: render completion ----
for round in $(seq 1 60); do
  while pgrep -f "render_new_batch.mjs" > /dev/null; do
    sleep 300
  done
  n=$(ls $OUT 2>/dev/null | wc -l)
  if [ "$n" -ge "$MIN_OK" ]; then
    echo "[$(date +%T)] render complete: $n/$TARGET" | tee -a $WORK/orchestrator.log
    break
  fi
  echo "[$(date +%T)] render exited early $n/$TARGET -> relaunch" | tee -a $WORK/orchestrator.log
  cd /home/linux/schem_render
  nohup $NODE $SCRIPT 0 999999 400 > $LOG 2>&1 &
  disown
  sleep 10
done

# ---- Phase 2: assemble render_all ----
bash $WORK/_assemble_render_all.sh > $WORK/assemble.log 2>&1
echo "[$(date +%T)] render_all: $(ls /data/redstone_v4/render_all | wc -l) links" | tee -a $WORK/orchestrator.log

# ---- Phase 3: step0 baseline eval ----
cd $WORK
ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1 $PY $WORK/_eval_v4_inpaint.py \
  /data/redstone_v4/work/v4_init.pt $WORK/baseline_eval.json 24 \
  >> $WORK/baseline_eval.log 2>&1 || echo "[$(date +%T)] WARN baseline eval failed (non-fatal)" | tee -a $WORK/orchestrator.log
echo "[$(date +%T)] baseline eval done" | tee -a $WORK/orchestrator.log
tail -3 $WORK/baseline_eval.log >> $WORK/orchestrator.log 2>/dev/null || true

# ---- Phase 4: launch supervisor (200k training) ----
nohup $PY $WORK/_rh_flow_v4_supervisor.py > $WORK/supervisor.log 2>&1 < /dev/null &
disown
echo "[$(date +%T)] supervisor launched" | tee -a $WORK/orchestrator.log

# ---- Phase 5: wait training done (supervisor exits at 200k) ----
while pgrep -f "_rh_flow_v4_supervisor.py" > /dev/null; do
  sleep 1800   # 30 min
done
echo "[$(date +%T)] training done (supervisor exited)" | tee -a $WORK/orchestrator.log

# ---- Phase 6: final eval ----
CKPT=$(ls -t /data/redstone_v4/out_v4/ckpts/denoiser_ema*.pt 2>/dev/null | head -1)
echo "[$(date +%T)] final ckpt: $CKPT" | tee -a $WORK/orchestrator.log
ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1 $PY $WORK/_eval_v4_inpaint.py \
  "$CKPT" $WORK/final_eval.json 24 \
  >> $WORK/final_eval.log 2>&1 || echo "[$(date +%T)] WARN final eval failed" | tee -a $WORK/orchestrator.log
echo "[$(date +%T)] final eval done" | tee -a $WORK/orchestrator.log
cat $WORK/final_eval.json.summary 2>/dev/null | tee -a $WORK/orchestrator.log
echo "[$(date +%T)] ORCHESTRATOR COMPLETE" | tee -a $WORK/orchestrator.log
