#!/bin/bash
# Build an offline mirror of the Wan2.1-I2V-14B-480P-Diffusers base model for HY-World 2.0 stage 3.
#
# Why: worldstereo_wrapper._load_transformer() calls
#   WorldStereoRefSModel.from_pretrained(cfg.base_model, subfolder="transformer", ...)
# which needs a weights file under the base model. The local HF snapshot has the
# transformer *config* but no weights, and the Hub is unreachable from the stage-3
# process (no proxy env) so diffusers cannot fetch them.
#
# The downloaded base weights would be thrown away anyway: the very next step is
#   weights = load_safetensors(<WorldStereo 34.8 GB file>); transformer.load_state_dict(weights, strict=False)
# and that file holds the complete, unprefixed WorldStereoRefSModel state dict
# (1799 tensors: blocks.*, controlnet.*, condition_embedder.*, camera_embedding.*, ...).
# So pointing the base model's transformer weights at the WorldStereo file gives a
# correct offline instantiation with zero downloads.
set -euo pipefail

SNAP=/root/autodl-tmp/hyworld2/cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/snapshots/b184e23a8a16b20f108f727c902e769e873ffc73
WS=/root/autodl-tmp/hyworld2/worldstereo-memory-dmd-model.safetensors
OUT=/dev/shm/WanLocal
WSCFG=/dev/shm/WorldStereo/worldstereo-memory-dmd/config.json

[ -d "$SNAP" ] || { echo "MISSING Wan snapshot: $SNAP"; exit 1; }
[ -f "$WS" ]   || { echo "MISSING WorldStereo weights: $WS"; exit 1; }

mkdir -p "$OUT/transformer"
for d in vae text_encoder tokenizer scheduler image_encoder image_processor; do
    [ -d "$SNAP/$d" ] && ln -sfn "$SNAP/$d" "$OUT/$d"
done
cp -f "$SNAP/transformer/config.json" "$OUT/transformer/config.json"
ln -sfn "$WS" "$OUT/transformer/diffusion_pytorch_model.safetensors"

echo "=== $OUT ==="
ls -la "$OUT"
echo "=== transformer ==="
ls -la "$OUT/transformer"

# Point the WorldStereo config's base_model at the mirror.
/root/miniconda3/bin/python - "$WSCFG" "$OUT" <<'PY'
import json, shutil, sys, os
cfg, out = sys.argv[1], sys.argv[2]
d = json.load(open(cfg))
old = d.get("base_model")
d["base_model"] = out
if not os.path.exists(cfg + ".remote"):
    shutil.copy2(cfg, cfg + ".remote")
json.dump(d, open(cfg, "w"), indent=2)
print(f"base_model: {old} -> {out}")
PY
echo "MIRROR_READY"
