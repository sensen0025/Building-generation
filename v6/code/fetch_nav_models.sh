#!/bin/bash
# Fetch the world-generation model sets into /dev/shm (RAM, wiped on container restart).
#
# Measured from this host: huggingface.co resolve URLs reach the CDN at ~9-10 MB/s *only* when
# redirects are followed (curl -L). Without -L the "download" is just the 302 body, which is what
# made me briefly conclude HF was dead. hf-mirror is genuinely slow here (662 B/s).
# ModelScope is an alternative for the two Qwen/IDEA repos, but not for MoGe or ZiM.
#
# Idempotent: complete files are skipped, partial files resume, sizes are verified against the API.
# Usage:  HF_TOKEN=hf_xxx bash fetch_nav_models.sh
set -u
SHM=${SHM:-/dev/shm/hyworld2-nav-models}
PY=${PY:-/root/miniconda3/bin/python}
TOKEN=${HF_TOKEN:-$(cat /root/.hf_token 2>/dev/null || true)}
BASE=${HF_BASE:-https://huggingface.co}
mkdir -p "$SHM"

fetch_repo() {  # owner/repo  dest_folder  [include_prefix]
  local repo="$1" dest="$SHM/$2" prefix="${3:-}" rev
  rev=$(curl -s -H "Authorization: Bearer $TOKEN" "$BASE/api/models/$repo" \
        | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["sha"])')
  echo "=== $repo @ ${rev:0:8} -> $dest"
  curl -s -H "Authorization: Bearer $TOKEN" "$BASE/api/models/$repo/tree/$rev?recursive=true" \
    | "$PY" -c '
import json, sys
for f in json.load(sys.stdin):
    if f.get("type") != "file":
        continue
    n = f["path"]
    if not (n.endswith((".safetensors", ".pt", ".onnx", ".json", ".txt", ".model"))
            or n.startswith("LICENSE")):
        continue
    print("%s|%s" % (n, f.get("size", 0)))
' > "/tmp/hf_list_$2.txt"
  local total=0 got=0
  while IFS='|' read -r path size; do
    [ -z "$path" ] && continue
    case "$path" in "$prefix"*) ;; *) continue ;; esac
    total=$((total + 1))
    mkdir -p "$dest/$(dirname "$path")"
    if [ -f "$dest/$path" ] && [ "$(stat -c%s "$dest/$path")" = "$size" ]; then
      got=$((got + 1)); echo "ok   $path ($size)"; continue
    fi
    echo "get  $path (want $size)"
    curl -sL -C - -H "Authorization: Bearer $TOKEN" -o "$dest/$path" \
         "$BASE/$repo/resolve/$rev/$path" -w "  http=%{http_code} got=%{size_download} bps=%{speed_download}\n"
    if [ "$(stat -c%s "$dest/$path" 2>/dev/null || echo 0)" = "$size" ]; then
      got=$((got + 1)); echo "ok   $path verified"
    else
      echo "SHORT $path have=$(stat -c%s "$dest/$path" 2>/dev/null || echo 0) want=$size"
    fi
  done < "/tmp/hf_list_$2.txt"
  echo "SUMMARY $repo complete=$got/$total"
}

fetch_repo "Ruicheng/moge-2-vitl-normal" moge
fetch_repo "naver-iv/zim-anything-vitl" zim "zim_vit_l_2092/"
fetch_repo "Qwen/Qwen3-VL-8B-Instruct" vlm
fetch_repo "IDEA-Research/grounding-dino-tiny" grounding-dino
chmod -R a+rX "$SHM" 2>/dev/null
echo NAV_MODELS_DONE
du -sh "$SHM"/* 2>/dev/null
