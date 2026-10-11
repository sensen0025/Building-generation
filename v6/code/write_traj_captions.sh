#!/bin/bash
# Write the per-trajectory captions that stage 2 normally asks the VLM for.
#
# traj_render.caption_single_video() calls the Qwen3-VL service for every rendered trajectory and
# writes render_results/<view>/<traj>/traj_caption.json. Without the VLM every call raised
# "Connection error" and no caption file was written, so stage 3 died in
# data_utils.load_mutli_traj_dataset() with FileNotFoundError on traj_caption.json.
#
# The caption is the TEXT CONDITION for WorldStereo's video expansion, so writing it ourselves is
# both unblocking and on-target: it states the blocky Minecraft look we require, which is the
# project's hard requirement ("must be Minecraft style"). It is not a VLM's description of the
# render - that distinction is recorded here on purpose.
set -u
S=/root/autodl-tmp/hyworld2/town_compare/scene
CAPTION=${CAPTION:-"a minecraft-style blocky voxel building exterior built from cubic blocks with flat stone brick walls, square windows and sharp right angles, standing on a flat grass ground plane under a clear blue sky, no curves, retro voxel game aesthetic, static architecture seen from outside, camera moving around the building"}

n=0
for d in $(find "$S/render_results" -mindepth 2 -maxdepth 2 -type d -name "traj*" | sort); do
  [ -f "$d/render.mp4" ] || continue
  /root/miniconda3/bin/python - "$d/traj_caption.json" "$CAPTION" <<'PY'
import json, sys
json.dump({"prompt": sys.argv[2]}, open(sys.argv[1], "w"), indent=2)
PY
  n=$((n+1))
done
echo "wrote $n traj_caption.json files"
find "$S/render_results" -name traj_caption.json | head -4
echo "--- sample ---"
cat "$(find "$S/render_results" -name traj_caption.json | head -1)"
