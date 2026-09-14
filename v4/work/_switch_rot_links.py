# -*- coding: utf-8 -*-
"""把 render_all 中 k=1,2,3 槽位的条件图指向对应旋转渲染, 修复图像与朝向错配.

k=0 槽位保持指向 render_new/bld_{j}.png (0 度, 与体素朝向一致).
用法:
  python _switch_rot_links.py --verify    # 只检查当前状态
  python _switch_rot_links.py             # 执行切换
"""
import os
import sys
import json

N0 = 21356
J = 6062
ALL = '/data/redstone_v4/render_all'
ROT = '/data/redstone_v4/render_rot'
NEW = '/data/redstone_v4/render_new'


def main():
    verify_only = '--verify' in sys.argv
    switched = 0
    missing = 0
    k0_ok = 0
    k0_bad = []
    for j in range(J):
        # k=0 应指向 render_new (0 度)
        dst0 = f'{ALL}/bld_{N0 + j * 4:05d}.png'
        should0 = f'{NEW}/bld_{j:05d}.png'
        tgt0 = os.path.realpath(dst0) if os.path.lexists(dst0) else ''
        if tgt0 == os.path.realpath(should0):
            k0_ok += 1
        else:
            k0_bad.append((N0 + j * 4, tgt0))
        # k=1,2,3 应指向 render_rot
        for k in (1, 2, 3):
            slot = N0 + j * 4 + k
            src = f'{ROT}/bld_{slot:05d}.png'
            dst = f'{ALL}/bld_{slot:05d}.png'
            if not (os.path.exists(src) and os.path.getsize(src) > 5000):
                missing += 1
                continue
            cur = os.path.realpath(dst) if os.path.lexists(dst) else ''
            if cur == os.path.realpath(src):
                switched += 1
                continue
            if not verify_only:
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
                switched += 1
    print(f'k=0 指向 render_new 且正确: {k0_ok}/{J}')
    if k0_bad:
        print(f'  !! k=0 异常 {len(k0_bad)} 个, 例: {k0_bad[:3]}')
    print(f'k=1,2,3 已指向旋转图: {switched}/{J * 3}')
    print(f'旋转图缺失: {missing}')
    rep = {'k0_ok': k0_ok, 'k0_bad': len(k0_bad), 'switched': switched,
           'missing': missing, 'mode': 'verify' if verify_only else 'apply'}
    json.dump(rep, open('/data/redstone_v4/work/rot_link_report.json', 'w'), indent=1)
    print('REPORT ' + json.dumps(rep))


if __name__ == '__main__':
    main()
