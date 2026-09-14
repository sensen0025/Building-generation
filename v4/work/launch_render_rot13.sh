#!/bin/bash
# 幂等启动旋转条件图渲染 (13 个细分片). 可反复调用: 已建完且未在跑的区间才启动.
# 用法: bash launch_render_rot13.sh
NODE=/home/linux/node-v22.14.0-linux-x64/bin/node
R=/home/linux/schem_render/render_rot_batch.mjs
W=/data/redstone_v4/work
LITDIR=/data/redstone_v4/litematics_rot

# idx:slot_start:slot_end
ENTRIES="0:21356:24387 1:24387:27418 2:27418:30449 3:30449:31965 4:31965:33480 5:33480:34996 6:34996:36511 7:36511:38027 8:38027:39542 9:39542:41058 10:41058:42573 11:42573:44089 12:44089:45604"

cd /home/linux/schem_render
launched=0
for e in $ENTRIES; do
  idx=${e%%:*}; rest=${e#*:}; a=${rest%%:*}; b=${rest##*:}
  want=$(awk -v A=$a -v B=$b 'BEGIN{c=0; for(s=A;s<B;s++) if(((s-21356)%4)!=0) c++; print c}')
  have=$(ls $LITDIR 2>/dev/null | awk -v A=$a -v B=$b '{n=substr($0,1,5)+0; if(n>=A && n<B) c++} END{print c+0}')
  if [ "$have" -lt "$want" ]; then
    echo "shard$idx [$a,$b): not built ($have/$want)"
    continue
  fi
  if pgrep -f "render_rot_batch.mjs $a $b" > /dev/null 2>&1; then
    echo "shard$idx [$a,$b): already rendering"
    continue
  fi
  lp=$(( 8990 + idx * 2 ))
  vp=$(( 8991 + idx * 2 ))
  setsid bash -c "$NODE $R $a $b 400 $lp $vp > $W/render_rot_shard$idx.log 2>&1; echo DONE > /tmp/renderrot_$idx.done" </dev/null >/dev/null 2>&1 &
  echo "shard$idx [$a,$b): LAUNCHED pid=$! ports=$lp/$vp files=$have"
  launched=$((launched+1))
done
sleep 3
echo "--- launched=$launched alive=$(pgrep -fc render_rot_batch) ---"
