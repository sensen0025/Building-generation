#!/bin/bash
# Assemble render_all: symlinks old rot images + new images (new idx offset = old count 21296)
# combined idx i (>=21296): new sample j = (i - 21296 - 60) ... NO: use direct naming bld_%05d with combined idx
# The encoder script writes ss64_all with old first (21356) then new (N*4).
# Old images exist as render_output_rot/bld_%04d.png (idx = old combined idx 0..21355, 60 missing -> holes OK, dataset filters)
# New images: render_new/bld_%05d.png numbered by NEW sample index j (0..6061) = combined idx 21356 + j*4 + 0 (only 0deg gets the image!)
# MCFlowImageDataset requires image per combined idx. Strategy: only the 0deg entry of each new sample gets an image;
# for rotated copies (k=1,2,3) ALSO reuse the same image (same building, rotated view unavailable) -> old rot pipeline
# had true rotated renders; for new data we reuse the 0deg render for all 4 rotations (condition image rotation mismatch accepted).
set -e
OUT=/data/redstone_v4/render_all
mkdir -p $OUT
# old images
for f in /home/linux/redstone_ai/render_output_rot/bld_*.png; do
  b=$(basename $f)
  ln -sf $f $OUT/$b
done
# new images: j -> combined idx 21356 + j*4, replicate 4x
for f in /data/redstone_v4/render_new/bld_*.png; do
  j=$(basename $f .png | sed 's/bld_//')
  j=$((10#$j))
  for k in 0 1 2 3; do
    i=$((21356 + j*4 + k))
    ln -sf $f $OUT/$(printf 'bld_%05d.png' $i)
  done
done
echo "render_all total: $(ls $OUT | wc -l)"
