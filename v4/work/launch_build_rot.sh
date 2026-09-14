#!/bin/bash
# 启动 8 个并行 worker 构建旋转槽位 litematic. 用法: bash launch_build_rot.sh
PY=/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python
W=/data/redstone_v4/work
S=$W/_build_rot_litematics.py
OUT=/data/redstone_v4/litematics_rot

RANGES="21356:24387 24387:27418 27418:30449 30449:33480 33480:36511 36511:39542 39542:42573 42573:45604"

# 已在跑就不重复启动
if pgrep -f "_build_rot_litematics" > /dev/null 2>&1; then
  echo "already running:"; pgrep -af "_build_rot_litematics" | head -3; exit 0
fi

rm -f /tmp/buildrot_*.done
i=0
for r in $RANGES; do
  a=${r%%:*}
  b=${r##*:}
  setsid bash -c "$PY $S $a $b > $W/build_rot_$i.log 2>&1; echo DONE > /tmp/buildrot_$i.done" </dev/null >/dev/null 2>&1 &
  echo "worker$i pid=$! range=$a-$b"
  i=$((i+1))
done
sleep 3
echo "--- alive ---"
pgrep -af "_build_rot_litematics" | wc -l
