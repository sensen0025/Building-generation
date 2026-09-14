#!/bin/bash
# Server cleanup: A-tier delete (verified dead) + B-tier move to /data/archive_v3
set -u
ARCH=/data/archive_v3
RA=/home/linux/redstone_ai
V4=/data/redstone_v4
mkdir -p $ARCH/datasets_v1 $ARCH/v2_preaug $ARCH/oldvae $ARCH/checkpoints $ARCH/renders $ARCH/v4_intermediate $ARCH/logs
exec > >(tee -a $ARCH/cleanup.log) 2>&1
log(){ echo "[$(date +%T)] $*"; }

log "=== cleanup start ==="

# ---------- preflight: canonical assets MUST exist ----------
log "--- preflight ---"
FAIL=0
for p in \
  "$RA/dataset/v2/ss64_rot.npy" \
  "$RA/dataset/v2/ss64_state_rot.npy" \
  "$RA/dataset/v2/ss_latent_mc64_v2_rot.npy" \
  "$RA/dataset/v2/latent_norm_rot.npz" \
  "$RA/TRELLIS/outputs/ss_vae_mc_v2/ckpts/encoder_step0050000.pt" \
  "$RA/TRELLIS/outputs/ss_flow_mc_rot_400k/ckpts/denoiser_ema0.9999_step0400000.pt" \
  "$RA/render_output_rot" \
  "$V4/train/ss_latent_all.npy" \
  "$V4/train/ss64_all.npy" \
  "$V4/train/ss64_state_all.npy" \
  "$V4/train/latent_norm_all.npz" \
  "$V4/work/v4_init.pt" \
  "$V4/work/ss_flow_mc_v4_comb.json" \
  "$V4/sources" \
  "$V4/litematics_new" ; do
  if [ -e "$p" ]; then log "  OK   $p"; else log "  MISS $p"; FAIL=1; fi
done
[ $FAIL -eq 0 ] || { log "ABORT: canonical asset missing"; exit 1; }
log "preflight passed"

# ---------- Phase A: delete ----------
log "--- Phase A: delete dead files ---"
for f in "$RA/_staging.zip" "$RA/trellis_main.zip" "$RA/inpaint_upgrade_codex.tar" \
         "$RA/TRELLIS/train.py.bak_rh" "$V4/work/render_new_batch.mjs.bak" ; do
  [ -e "$f" ] && { log "  rm $f ($(du -h "$f" 2>/dev/null | cut -f1))"; rm -f "$f"; }
done
for f in "$RA"/TRELLIS/configs/flow/*.bak_* ; do
  [ -e "$f" ] && { log "  rm $f"; rm -f "$f"; }
done

N=$(find "$RA" "$V4" -name __pycache__ -type d 2>/dev/null | wc -l)
find "$RA" "$V4" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
log "  removed $N __pycache__ dirs"

ACTIVE=$(ps -eo args 2>/dev/null | grep -o '/tmp/puppeteer_dev_chrome_profile-[A-Za-z0-9]*' | sort -u)
for d in /tmp/puppeteer_dev_chrome_profile-* ; do
  [ -d "$d" ] || continue
  if echo "$ACTIVE" | grep -qF "$d" ; then log "  keep (active in ps) $d"; continue; fi
  if [ -n "$(find "$d" -maxdepth 0 -mmin -30 2>/dev/null)" ] ; then log "  keep (modified <30min) $d"; continue; fi
  log "  rm $d ($(du -sh "$d" 2>/dev/null | cut -f1))"; rm -rf "$d"
done

for f in "$RA/rot_render_full.log" "$V4/work/render_shard3.log" "$V4/work/render_new.log" \
         "$V4/work/orchestrator.log" "$V4/work/finalize.log" "$V4/work/assemble.log" \
         "$V4/work/smoke.log" "$V4/work/smoke2.log" "$V4/work/smoke3.log" ; do
  [ -e "$f" ] && { mv -f "$f" "$ARCH/logs/$(basename $f)"; log "  archived log $(basename $f)"; }
done
log "Phase A done"

# ---------- Phase B: move to archive ----------
log "--- Phase B: move to archive ---"
move(){ src="$1"; dst="$2"; if [ -e "$src" ]; then log "  mv $src ($(du -sh "$src" 2>/dev/null | cut -f1))"; mv -f "$src" "$dst" || log "  ** mv FAILED $src"; else log "  skip (absent) $src"; fi; }

move "$V4/work/ss64_new.npy"                        "$ARCH/v4_intermediate/"
move "$V4/work/ss64_state_new.npy"                  "$ARCH/v4_intermediate/"
move "$V4/work/ss64_new_c.npy"                      "$ARCH/v4_intermediate/"
move "$V4/work/ss64_state_new_c.npy"                "$ARCH/v4_intermediate/"
move "$RA/dataset/voxel96_id.dat"                   "$ARCH/datasets_v1/"
move "$RA/dataset/voxel96_state.dat"                "$ARCH/datasets_v1/"
move "$RA/dataset/v2/ss64.npy"                      "$ARCH/v2_preaug/"
move "$RA/dataset/v2/ss64_state.npy"                "$ARCH/v2_preaug/"
move "$RA/dataset/v2/ss_latent_mc64_v2.npy"         "$ARCH/v2_preaug/"
move "$RA/dataset/v2/ss_latent_mc64_v2_oldvae.npy"  "$ARCH/oldvae/"
move "$RA/dataset/v2/latent_norm_oldvae.npz"        "$ARCH/oldvae/"
move "$RA/TRELLIS/outputs/ss_flow_mc_img_v3.old68k" "$ARCH/checkpoints/"
move "$RA/render_output"                            "$ARCH/renders/"
move "$RA/render_output_placeholder_bak"            "$ARCH/renders/"
log "Phase B done"

# ---------- verify ----------
log "--- verify canonical assets survived ---"
for p in "$RA/dataset/v2/ss64_rot.npy" "$RA/dataset/v2/ss_latent_mc64_v2_rot.npy" \
         "$RA/render_output_rot" "$V4/train/ss_latent_all.npy" "$V4/work/v4_init.pt" ; do
  [ -e "$p" ] && log "  STILL OK  $p" || log "  ** LOST $p"
done
log "--- render still running? ---"
log "  render_new files: $(ls $V4/render_new 2>/dev/null | wc -l)"
pgrep -f "render_new_batch.mjs" >/dev/null && log "  render shards: ALIVE" || log "  render shards: not running"
df -h / /data | sed 's/^/  /'
log "=== cleanup complete ==="
