
import os, time, subprocess
LIST = '/home/linux/redstone_ai/dataset/valid_files_list.txt'
LOG = '/home/linux/redstone_ai/logs/rescan.log'
# 等 rescan 完成(文件出现且不再增长或出现 saved 标记)
for _ in range(600):  # 最长 600*10s = 100min
    if os.path.exists(LIST):
        sz1 = os.path.getsize(LIST)
        time.sleep(10)
        sz2 = os.path.getsize(LIST)
        if sz1 == sz2 and sz1 > 0:
            break
    time.sleep(10)
print('rescan list ready, lines =', sum(1 for _ in open(LIST)), flush=True)
# 启动 rebuild
r = subprocess.Popen(
    ['/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python',
     '/home/linux/redstone_ai/_rh_rebuild_full.py'],
    cwd='/home/linux/redstone_ai',
    stdout=open('/home/linux/redstone_ai/logs/rebuild_full.log', 'a'),
    stderr=subprocess.STDOUT)
print('rebuild launched pid', r.pid, flush=True)
r.wait()
print('rebuild exit', r.returncode, flush=True)
