#!/bin/bash
# v7 seed panorama: a SINGLE building in three-quarter view, on grass, with a real horizon.
#
# v6 diagnosis to avoid repeating: with the building filling the frame on a transparent background,
# HY-Pano read it as "the building surrounds the camera" and emitted two curved facades at the frame
# edges plus a far copy in the centre, and the generated cloud contained no coherent building.
set -u
cd /root/autodl-tmp/hyworld2/town_compare/source/hyworld2/panogen || exit 1
export HF_HOME=/root/autodl-tmp/hyworld2/cache/huggingface
S=/root/autodl-tmp/hyworld2/town_compare/input/seed_v7.png
/root/miniconda3/bin/python pipeline_with_qwen_image.py \
  --image "$S" \
  --prompt "a 360 degree equirectangular panorama photographed at eye level by a person standing outside in front of one large blocky voxel building, seen from a three-quarter angle so both the front facade and one side wall are fully visible, flat grass ground plane extending to the horizon in every direction, clear blue sky only above the horizon, minecraft-style architecture built from cubic blocks, 1x1 meter cubes, flat stone brick walls, square windows, sharp right angles, no curves, retro voxel game aesthetic, sunny day" \
  --negative-prompt "aerial view, bird's eye view, top down, floating island, white background, no ground, empty grey sky, photorealistic, smooth curved surfaces, filigree, blurry, depth of field, lens flare, organic shapes, multiple buildings, village" \
  --pretrained-model-name-or-path /dev/shm/hyworld2-town-pano \
  --lora-path /root/autodl-tmp/hyworld2/town_compare/pano_lora \
  --lora-subfolder "" \
  --num-inference-steps 40 --height 960 --width 1952 --seed 42 \
  --vae-tiling \
  --save /root/autodl-tmp/hyworld2/town_compare/out/v7_pano.png
echo "PANO_EXIT=$?"
