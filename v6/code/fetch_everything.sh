#!/bin/bash
# Single resumable downloader for everything the generation chain needs on this box.
#
# Written after finding three overlapping downloader instances writing the same files, and a pano
# refetch that died with a broken pipe at 15 GB. Both are avoided here: one instance only, curl -C -
# resume, and an exact size check per file against the provider's API.
#
# Sources: ModelScope (16 MB/s measured) for Qwen-Image-Edit; huggingface.co for the rest, which
# only reaches ~9 MB/s when redirects are followed (curl -L) - without -L you download a 302 body.
set -u
SHM=/dev/shm/hyworld2-town-pano
NAV=/dev/shm/hyworld2-nav-models
TOKEN=$(cat /root/.hf_token 2>/dev/null || true)
PY=/root/miniconda3/bin/python
QS=Qwen/Qwen-Image-Edit-2509
QREV=d3968ef930e841f4c73640fb8afa3b306a78167e
mkdir -p "$SHM/transformer" "$SHM/text_encoder" "$NAV"

fetch_ms() {  # rel_path want_bytes
  local rel="$1" want="$2" out="$SHM/$1"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$want" ]; then echo "ok   $rel"; return 0; fi
  echo "get  $rel"
  curl -sL -C - -o "$out" \
    "https://modelscope.cn/api/v1/models/$QS/repo?Revision=master&FilePath=$rel" \
    -w "  http=%{http_code} got=%{size_download}\n"
  [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" = "$want" ] && echo "ok   $rel verified" || echo "SHORT $rel"
}

fetch_ms transformer/diffusion_pytorch_model-00001-of-00005.safetensors 9973578592
fetch_ms transformer/diffusion_pytorch_model-00003-of-00005.safetensors 9987307440
fetch_ms text_encoder/model-00002-of-00004.safetensors 4991495816
echo PANO_SHARDS_DONE

# The remaining pano shards are symlinks into the persistent HF cache, created by
# prepare_pano_cache.py. Recreate any that are missing or broken.
BASE=/root/autodl-tmp/hyworld2/cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2509/snapshots/$QREV
if [ -d "$BASE" ]; then
  while IFS= read -r src; do
    rel=${src#$BASE/}
    dst="$SHM/$rel"
    mkdir -p "$(dirname "$dst")"
    [ -e "$dst" ] || ln -s "$src" "$dst"
  done < <(find "$BASE" -type f)
fi
broken=0
for f in $(find "$SHM" -xtype l); do echo "BROKEN $f"; broken=$((broken + 1)); done
echo "broken_symlinks=$broken"

export HF_TOKEN="$TOKEN"
source /etc/network_turbo >/dev/null 2>&1
bash /root/fetch_nav_models.sh
echo ALL_DOWNLOADS_DONE
du -sh "$SHM" "$NAV"/* 2>/dev/null
