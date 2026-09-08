# -*- coding: utf-8 -*-
"""plan-mode 只读探索: 确认绕 y 轴 90° 旋转 ss64 的坐标映射正确性 + 评估可行性.
不写文件, 仅打印验证.
"""
import numpy as np
import os

BASE = '/home/linux/redstone_ai/dataset/v2'
ss = np.load(f'{BASE}/ss64.npy', mmap_mode='r')
print('ss64 shape:', ss.shape, 'dtype:', ss.dtype, flush=True)

# 取 idx250 验证: 体素 y 轴是否竖直(高度), 方法: 检查非零块的 y 分布 vs 假设
idx = 250
v = np.asarray(ss[idx]).astype(np.int16)
nz = np.argwhere(v > 0)
print(f'idx{idx} occ={(v>0).mean()*100:.2f}%', flush=True)
if len(nz):
    mn = nz.min(0); mx = nz.max(0)
    print('  bbox (x,y,z):', mn, '..', mx, ' spans:', (mx-mn+1).tolist(), flush=True)
    # 相机 look 在 mass center; 竖直轴应该是 spans 里高度对应的那个
    # 从 meta 看 idx250 spans=[27,9,36] -> 中间值 9 是高度(y), 27/36 是水平(x,z)
    # 所以 y 轴(索引1)是竖直; 绕 y 旋转 -> 交换 (x,z) 并可能反号

# 验证绕 y 轴 90° 旋转的整数映射: rotation90_y: (x,y,z) -> (L-1-z, y, x)  (绕着过中心的 y 轴转)
# 但体素网格是 0..63, 旋转后形状变吗? 绕 y 转 90°, 64x64x64 正方体 -> 形状不变(64x64x64), 因为 Lx=Ly=Lz=64 且只动 x,z.
def rot90_y(arr):
    """绕几何中心 y 轴顺时针 90° (逆时针同理, 只是方向). 适用 64^3."""
    L = arr.shape[0]
    # (x,y,z) -> (L-1-z, y, x)
    return np.transpose(arr[:, :, ::-1], (2, 1, 0))  # z reverse then swap x,z

# 用简单非对称测试块验证映射
test = np.zeros((4,4,4), dtype=np.int16)
test[0,1,2] = 7  # x=0,y=1,z=2
r = rot90_y(test)
# 期望: 新坐标 (L-1-z, y, x) = (3-2, 1, 0) = (1,1,0)
nz_new = np.argwhere(r>0)
print('test block rot: input (0,1,2) -> output', nz_new[0].tolist(), '(expect [1,1,0])', flush=True)
print('rot preserves shape:', r.shape == test.shape, flush=True)

# 对 idx250 应用旋转, 检查 occ 不变 + bbox 的 x/z 交换
vr = rot90_y(v)
print(f'idx250 rot: occ={(vr>0).mean()*100:.2f}% (应= {(v>0).mean()*100:.2f}%)', flush=True)
nzr = np.argwhere(vr>0)
if len(nzr):
    print('  rot bbox (x,y,z):', nzr.min(0), '..', nzr.max(0), flush=True)

# 无状态 vs 有状态: state 也需同步旋转(state 是逐体素属性, 跟随体素坐标), 直接对 state 用同样转置
print('OK - rotation mechanics valid (y is vertical axis)', flush=True)
