#!/bin/bash
# Full redo of NEW-data pipeline after transpose fix.
set -e
PY=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python
WORK=/data/redstone_v4/work
NODE=/home/linux/node-v22.14.0-linux-x64/bin/node
RENDSCRIPT=/home/linux/schem_render/render_new_batch.mjs
TARGET=6062
MIN_OK=$((TARGET - 30))
LOG() { echo "[$(date +%T)] $1"; }

LOG "=== REDO start (transpose fixed) ==="

# Phase 0: clean wrong artifacts
rm -f $WORK/ss64_new.npy $WORK/ss64_new_c.npy $WORK/ss64_state_new.npy $WORK/ss64_state_new_c.npy
rm -f $WORK/ingest_manifest.json $WORK/ingest_manifest_ok.json
rm -rf /data/redstone_v4/litematics_new /data/redstone_v4/render_new /data/redstone_v4/render_all /data/redstone_v4/train /data/redstone_v4/smoke_out
mkdir -p /data/redstone_v4/litematics_new /data/redstone_v4/render_new /data/redstone_v4/render_all /data/redstone_v4/train
LOG "cleaned artifacts"

# Phase 1: ingest (fixed transpose)
source /home/linux/anaconda3/etc/profile.d/conda.sh
conda activate torch2.4_cuda12.1
$PY /data/redstone_v4/code/_build_v4_ingest.py > $WORK/ingest.log 2>&1
LOG "ingest done"

# Phase 2: compact
$PY $WORK/_compact_ingest.py > $WORK/compact.log 2>&1
LOG "compact done"

# Phase 3: export litematic + encode/merge (parallel)
$PY $WORK/_export_new_litematics.py > $WORK/export_lit.log 2>&1 &
EXPORT_PID=$!
$PY $WORK/_encode_merge_v4.py > $WORK/encode_merge.log 2>&1 &
ENCODE_PID=$!
wait $EXPORT_PID; LOG "export done"
wait $ENCODE_PID; LOG "encode/merge done"

# Phase 4: render (3 shards, resumable)
cd /home/linux/schem_render
nohup $NODE $RENDSCRIPT 0 2021 400 8942 8943 > $WORK/render_shard0.log 2>&1 &
nohup $NODE $RENDSCRIPT 2021 4042 400 8944 8945 > $WORK/render_shard1.log 2>&1 &
nohup $NODE $RENDSCRIPT 4042 6062 400 8946 8947 > $WORK/render_shard2.log 2>&1 &
disown
LOG "render 3 shards launched"

# wait render complete (relaunch if early exit)
for round in $(seq 1 60); do
  while pgrep -f "render_new_batch.mjs" > /dev/null; do sleep 300; done
  n=$(ls /data/redstone_v4/render_new | wc -l)
  if [ "$n" -ge "$MIN_OK" ]; then LOG "render complete: $n/$TARGET"; break; fi
  LOG "render early exit $n -> relaunch"
  nohup $NODE $RENDSCRIPT 0 999999 400 > $WORK/render_new.log 2>&1 &
  disown
  sleep 10
done

# Phase 5: assemble render_all
bash $WORK/_assemble_render_all.sh > $WORK/assemble.log 2>&1
LOG "render_all: $(ls /data/redstone_v4/render_all | wc -l) links"

# Phase 6: smoke test (4 steps)
cd /home/linux/redstone_ai/TRELLIS
ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1 $PY train.py --config $WORK/ss_flow_mc_v4_smoke.json --output_dir /data/redstone_v4/smoke_out --data_dir /home/linux/redstone_ai/TRELLIS/data/mc > $WORK/smoke_redo.log 2>&1
LOG "smoke done"

# Phase 7: baseline eval + launch supervisor (200k training)
ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1 $PY $WORK/_eval_v4_inpaint.py $WORK/v4_init.pt $WORK/baseline_eval.json 24 >> $WORK/baseline_eval.log 2>&1 || LOG "WARN baseline eval failed (non-fatal)"
nohup $PY $WORK/_rh_flow_v4_supervisor.py > $WORK/supervisor.log 2>&1 < /dev/null &
disown
LOG "supervisor launched (200k training)"
sleep 10
ps aux | grep -E "train.py|supervisor" | grep -v grep | head -3
LOG "=== REDO COMPLETE ==="
