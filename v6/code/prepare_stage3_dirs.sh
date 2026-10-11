#!/bin/bash
# Rebuild the stage-3 model directories ON THE PERSISTENT DATA DISK and expose them at the
# /dev/shm paths the code expects. Idempotent, and safe to run after any container restart.
#
# Two directories are involved:
#   WorldStereo/<model_type>/{config.json, model.safetensors}  <- repo_id passed as a LOCAL DIR
#       (worldstereo_wrapper treats a local dir as "read from disk", which is the only way that
#        works with HF_HUB_OFFLINE=1 and a cache whose snapshot blobs were pruned)
#   WanLocal/{vae,text_encoder,tokenizer,scheduler,image_encoder,image_processor,transformer}
#       the base model cfg.base_model points at. The transformer weights are NOT downloaded:
#       the wrapper loads the real weights from the 34.8 GB WorldStereo file right after, so the
#       base file only has to exist for the architecture to instantiate. It is symlinked to that
#       same WorldStereo file.
set -u

DATA=/root/autodl-tmp/hyworld2
WS_WEIGHTS=$DATA/worldstereo-memory-dmd-model.safetensors
MODEL_TYPE=worldstereo-memory-dmd
WAN_SNAP=$(ls -d $DATA/cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/snapshots/*/ 2>/dev/null | head -1)
WS_SNAP=$(ls -d $DATA/cache/huggingface/hub/models--hanshanxue--WorldStereo/snapshots/*/ 2>/dev/null | head -1)

[ -f "$WS_WEIGHTS" ] || { echo "MISSING WorldStereo weights $WS_WEIGHTS"; exit 1; }
[ -n "$WAN_SNAP" ] || { echo "MISSING Wan snapshot"; exit 1; }

# ---- WorldStereo dir on the data disk ----
WSDIR=$DATA/WorldStereo/$MODEL_TYPE
mkdir -p "$WSDIR"
if [ ! -s "$WSDIR/config.json" ]; then
  if [ -n "$WS_SNAP" ] && [ -s "$WS_SNAP/$MODEL_TYPE/config.json" ]; then
    cp -f "$WS_SNAP/$MODEL_TYPE/config.json" "$WSDIR/config.json"
    echo "config.json copied from HF cache"
  else
    source /etc/network_turbo >/dev/null 2>&1 || true
    T=$(cat /root/.hf_token 2>/dev/null || true)
    curl -sL -H "Authorization: Bearer $T" -o "$WSDIR/config.json" \
      "https://huggingface.co/hanshanxue/WorldStereo/raw/main/$MODEL_TYPE/config.json"
    echo "config.json fetched from the Hub"
  fi
fi
ln -sfn "$WS_WEIGHTS" "$WSDIR/model.safetensors"

# ---- WanLocal mirror on the data disk ----
WL=$DATA/WanLocal
mkdir -p "$WL/transformer"
for d in vae text_encoder tokenizer scheduler image_encoder image_processor; do
  [ -d "$WAN_SNAP/$d" ] && ln -sfn "$WAN_SNAP/$d" "$WL/$d"
done
cp -f "$WAN_SNAP/transformer/config.json" "$WL/transformer/config.json"
ln -sfn "$WS_WEIGHTS" "$WL/transformer/diffusion_pytorch_model.safetensors"

# point the WorldStereo config at the mirror (keep a pristine copy once)
/root/miniconda3/bin/python - "$WSDIR/config.json" /dev/shm/WanLocal <<'PY'
import json, shutil, sys, os
cfg, out = sys.argv[1], sys.argv[2]
d = json.load(open(cfg))
if not os.path.exists(cfg + ".remote"):
    os.makedirs(os.path.dirname(cfg + ".remote"), exist_ok=True)
    shutil.copy2(cfg, cfg + ".remote")
d["base_model"] = out
json.dump(d, open(cfg, "w"), indent=2)
print(f"base_model -> {out}")
PY

# ---- expose at the /dev/shm paths the code hardcodes ----
ln -sfn "$DATA/WorldStereo" /dev/shm/WorldStereo
ln -sfn "$WL" /dev/shm/WanLocal
ln -sfn "$DATA/nav-models" /dev/shm/hyworld2-nav-models
ln -sfn "$DATA/town-pano" /dev/shm/hyworld2-town-pano
ln -sfn "$DATA/cache/huggingface" /dev/shm/hf-cache 2>/dev/null || true

echo "=== /dev/shm ==="
ls -la /dev/shm/ | grep -E "hyworld|WorldStereo|WanLocal"
echo "=== WorldStereo ==="
ls -laL /dev/shm/WorldStereo/$MODEL_TYPE/ | tail -3
echo "=== WanLocal ==="
ls /dev/shm/WanLocal/
echo "=== base_model ==="
/root/miniconda3/bin/python -c "import json;print(json.load(open('/dev/shm/WorldStereo/$MODEL_TYPE/config.json'))['base_model'])"
echo STAGE3_DIRS_READY
