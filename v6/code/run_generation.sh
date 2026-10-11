#!/bin/bash
# Run the HY-World 2.0 full point-cloud generation chain (5 stages) on a single H20.
#
# Recovered from the previous successful run on this box (official_worldgen/case000) and
# hyworld2/worldgen/README.md:
#   1 traj_generate.py   world-nav trajectory planning (needs the local Qwen3-VL service)
#   2 traj_render.py     multi-GPU point-cloud rendering along the trajectories
#   3 video_gen.py       WorldStereo 2.0 expansion (the stage that grows the world)
#   4 gen_gs_data.py     build 3DGS training data (point cloud / depth / normals)
#   5 world_gs_trainer   iversen-free 3DGS optimisation -> gaussians, mesh, point cloud
#
# The docs assume 8 GPUs; this box has 1, so every launcher uses nproc_per_node=1 and the 3DGS
# step count is raised as the README prescribes for fewer GPUs (1 GPU -> max_steps 8000).
#
# Usage:  SCENE=/root/autodl-tmp/hyworld2/town_compare/scene bash run_generation.sh
set -u
ROOT=/root/autodl-tmp/hyworld2/town_compare
SRC=$ROOT/source/hyworld2/worldgen
PY=$ROOT/nav_env/bin/python
SHM=/dev/shm/hyworld2-nav-models
SCENE=${SCENE:-$ROOT/scene}
LOGDIR=${LOGDIR:-$ROOT/gen_logs}
VLM_LOG=$ROOT/vlm_service_gen.log
mkdir -p "$LOGDIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

preflight() {
  local ok=1
  for f in "$SHM/moge/model.pt" "$SHM/zim/zim_vit_l_2092/encoder.onnx" \
           "$SHM/vlm/config.json" "$SHM/grounding-dino/config.json"; do
    if [ -s "$f" ]; then log "ok      $f"; else log "MISSING $f"; ok=0; fi
  done
  [ -s "$SCENE/panorama.png" ] || { log "MISSING $SCENE/panorama.png"; ok=0; }
  [ -s /root/autodl-tmp/hyworld2/worldstereo-memory-dmd-model.safetensors ] \
    || { log "MISSING WorldStereo weights"; ok=0; }
  [ "$ok" = 1 ] || { log "preflight failed"; exit 2; }
}

start_vlm() {  # local OpenAI-compatible Qwen3-VL service on port 8000
  if curl -s -o /dev/null --max-time 3 http://127.0.0.1:8000/v1/models; then
    log "VLM service already up"; return 0
  fi
  log "starting VLM service"
  setsid "$PY" "$ROOT/vlm_service.py" >"$VLM_LOG" 2>&1 </dev/null &
  for _ in $(seq 1 60); do
    sleep 10
    if curl -s -o /dev/null --max-time 3 http://127.0.0.1:8000/v1/models; then
      log "VLM service ready"; return 0
    fi
  done
  log "VLM service did not come up; see $VLM_LOG"
  return 3
}

stage() {  # name command...
  local name="$1"; shift
  log "=== stage $name"
  if ! "$@" >"$LOGDIR/$name.log" 2>&1; then
    log "stage $name FAILED; tail:"
    tail -25 "$LOGDIR/$name.log" | sed 's/^/    /'
    return 1
  fi
  log "=== stage $name done"
}

preflight
start_vlm || exit 3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HYWORLD_SPARSE_BACKEND=scipy
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PYTHONPATH="$ROOT/source"

cd "$SRC"
stage 1_traj_generate "$PY" traj_generate.py --target_path "$SCENE" \
  --llm_addr 127.0.0.1 --llm_port 8000 --llm_name Qwen/Qwen3-VL-8B-Instruct \
  --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm || exit 4
stage 2_traj_render "$PY" -m torch.distributed.run --standalone --nproc_per_node 1 traj_render.py --target_path "$SCENE" \
  --llm_addr 127.0.0.1 --llm_port 8000 --llm_name Qwen/Qwen3-VL-8B-Instruct || exit 5
stage 3_video_gen "$PY" -m torch.distributed.run --standalone --nproc_per_node 1 video_gen.py --target_path "$SCENE" || exit 6
stage 4_gen_gs_data "$PY" -m torch.distributed.run --standalone --nproc_per_node 1 gen_gs_data.py --root_path "$SCENE" \
  --save_normal --split_sky || exit 7
stage 5_gs_trainer "$PY" -m world_gs_trainer default --data_dir "$SCENE/gs_data" \
  --result_dir "$SCENE/gs_result" --max_steps 8000 --save_steps 8000 \
  --eval_steps 8000 --ply_steps 8000 || exit 8

log "GENERATION_DONE"
find "$SCENE" -maxdepth 3 -name "*.ply" -o -maxdepth 3 -name "*.glb" 2>/dev/null | head -20
