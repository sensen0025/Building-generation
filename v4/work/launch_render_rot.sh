#!/bin/bash
# 幂等启动旋转条件图渲染: 对每个 slot 区间, 若该区间 litematic 已建完且渲染未在跑, 则开一个分片.
# 用法: bash launch_render_rot.sh
NODE=/home/linux/node-v22.14.0-linux-x64/bin/node
R=/home/linux/schem_render/render_rot_batch.mjs
W=/data/redstone_v4/work
LITDIR=/data/redstone_v4/litematics_rot

RANGES="21356:24387 24387:27418 27418:30449 30449:33480 33480:36511 36511:39542 39542:42573 42573:45604"

cd /home/linux/schem_render
i=0
launched=0
for r in $RANGES; do
  a=${r%%:*}
  b=${r##*:}
  # 该区间应有多少个 k!=0 的 slot (4 个 slot 里 3 个)
  want=$(( (b - a) / 4 * 3 ))
  have=$(ls $LITDIR 2>/dev/null | awk -v A=$a -v B=$b '{n=substr($0,1,5)+0; if(n>=A && n<B) c++} END{print c+0}')
  if [ "$have" -lt "$want" ]; then
    echo "range $a-$b: not built yet ($have/$want) -> skip"
    i=$((i+1)); continue
  fi
  if pgrep -f "render_rot_batch.mjs $a $b" > /dev/null 2>&1; then
    echo "range $a-$b: already rendering"
    i=$((i+1)); continue
  fi
  lp=$(( 8990 + i * 2 ))
  vp=$(( 8991 + i * 2 ))
  setsid bash -c "$NODE $R $a $b 400 $lp $vp > $W/render_rot_shard$i.log 2>&1; echo DONE > /tmp/renderrot_$i.done" </dev/null >/dev/null 2>&1 &
  echo "range $a-$b: LAUNCHED shard$i pid=$! ports=$lp/$vp (have=$have/$want)"
  launched=$((launched+1))
  i=$((i+1))
done
sleep 3
echo "--- launched=$launched  alive=$(pgrep -fc render_rot_batch) ---"
