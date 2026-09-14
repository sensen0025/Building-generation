#!/bin/bash
# Driver: render all rotated litematics in resumable chunks (fresh Chrome per chunk).
cd /home/linux/schem_render
NODE=/home/linux/node-v22.14.0-linux-x64/bin/node
START=0
CHUNK=1500
TOTAL=15957
while [ $START -lt $TOTAL ]; do
  END=$((START + CHUNK))
  echo "=== chunk $START..$END $(date '+%F %T') ==="
  $NODE rot_batch_render.mjs $START $END 500 300
  START=$END
done
echo "ALL RENDER DONE $(date '+%F %T')"
