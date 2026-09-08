#!/bin/bash
set -e
cd /home/linux/redstone_ai/inpaint_upgrade_codex
mkdir -p originals
tar xf originals.tar -C originals
echo "=== extracted ==="
find originals -type f
echo
echo "=== diff line counts vs current ==="
for f in trellis/models/sparse_structure_flow.py trellis/datasets/mc_flow.py trellis/trainers/flow_matching/dense_flow.py; do
  echo "-- $f --"
  diff -u "originals/$f" "/home/linux/redstone_ai/TRELLIS/$f" | wc -l
done
