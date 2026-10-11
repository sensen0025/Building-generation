#!/bin/bash
# v7 runner: same five stages, tuned first stage for a COMPLETE building exterior.
#
# Verified in traj_generate.py before choosing these:
#   --split_view_num N : the pano is cut into N look directions at 360/N apart from the SAME position
#                        (default 3 -> 0/120/240). More views = finer angular coverage.
#   --apply_up_route   : adds a pitch-UP aerial trajectory (rotation [0,-up_rot_deg]); dropping it
#                        keeps the cameras level so facades, not roofs, are observed.
#   --move_dist        : translation = move_dist * median_depth; larger = more parallax = denser surface.
#   --nframe           : frames per trajectory; longer motion gives WorldMirror more to fuse.
#   --wonder_topk/--recon_topk : how many moving orbit trajectories are kept per view.
ROOT=/root/autodl-tmp/hyworld2/town_compare
SCENE=${SCENE:-$ROOT/scene_v7}
SRC=$ROOT/source/hyworld2/worldgen
PY=$ROOT/nav_env/bin/python
LOGDIR=$ROOT/gen_logs_v7
SHM=/dev/shm/hyworld2-nav-models
mkdir -p "$LOGDIR"
log() { echo "[$(date +%H:%M:%S)] $*"; }
stage() { name=$1; shift
  log "=== stage $name"
  if "$@" >"$LOGDIR/$name.log" 2>&1; then log "=== stage $name done"; else
    log "stage $name FAILED; tail:"; tail -25 "$LOGDIR/$name.log" | sed 's/^/    /'; return 1; fi; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HYWORLD_SPARSE_BACKEND=scipy
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PYTHONPATH="$ROOT/source"
export HF_HOME=/root/autodl-tmp/hyworld2/cache/huggingface
cd "$SRC" || exit 1
stage 1_traj_generate "$PY" traj_generate.py --target_path "$SCENE" \
  --split_view_num 6 --wonder_topk 5 --recon_topk 8 --move_dist 12.0 --nframe 32 \
  --apply_nav_traj --apply_recon_iteration || exit 4
stage 2_traj_render "$PY" -m torch.distributed.run --standalone --nproc_per_node 1 traj_render.py \
  --target_path "$SCENE" || exit 5
bash /root/write_traj_captions_v7.sh || exit 6
stage 3_video_gen "$PY" -m torch.distributed.run --standalone --nproc_per_node 1 video_gen.py \
  --target_path "$SCENE" --skip_exist || exit 7
stage 4_gen_gs_data "$PY" -m torch.distributed.run --standalone --nproc_per_node 1 gen_gs_data.py \
  --root_path "$SCENE" --save_normal --split_sky || exit 8
log GENERATION_DONE
find "$SCENE/render_results" -maxdepth 2 -name "*.ply" 2>/dev/null | head
