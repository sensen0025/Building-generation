"""Render the generated HY-World 2.0 outdoor cloud from several angles, with and without sky.

The three-view inspector used earlier was a diagnostic: axis-aligned scatters that answer "is there
a building" but read poorly. This is a display render - perspective 3D views at Minecraft-like
elevations, coloured by the cloud's own RGB, sky separable so the terrain and the build are not
buried under a blue-grey dome.

Library note, stated rather than hidden: mplot3d paints in painter order instead of using a depth
buffer, so at extreme angles a few pixels of far geometry can show through. open3d/pyvista are not
installed here.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def read_ply(path):
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'ply'
        assert b'binary_little_endian' in f.readline()
        n, props = None, []
        while True:
            line = f.readline().strip().decode()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                props.append(line.split()[-1])
            elif line == 'end_header':
                break
        dtype = [(p, '<f4' if p in ('x', 'y', 'z', 'nx', 'ny', 'nz')
                  else ('u1' if p in ('red', 'green', 'blue', 'alpha') else '<f4')) for p in props]
        raw = np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=np.dtype(dtype))
    return (np.stack([raw['x'], raw['y'], raw['z']], 1).astype(np.float32),
            np.stack([raw['red'], raw['green'], raw['blue']], 1).astype(np.float32) / 255.0)


def sky_split(xyz, rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (b >= r) & ((b - r) >= 0.012) & (np.abs(b - g) <= 0.10) & (b >= 0.27)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-points', type=int, default=420000)
    args = ap.parse_args()

    xyz, rgb = read_ply(args.ply)
    sky = sky_split(xyz, rgb)
    print(f'points {len(xyz)}  sky {sky.mean()*100:.1f}%')

    def sub(mask):
        idx = np.nonzero(mask)[0]
        if len(idx) > args.max_points:
            idx = idx[::max(1, len(idx) // args.max_points)]
        return xyz[idx], rgb[idx]

    # Z is up for HY-World clouds; show it that way rather than in the raw frame.
    def frame(p):
        return p[:, 0], p[:, 2], p[:, 1]          # plot X, Z(up), Y

    panels = [
        ('yaw 0, elev 22', 0, 22, 'scene'),
        ('yaw 90, elev 22', 90, 22, 'scene'),
        ('yaw 180, elev 22', 180, 22, 'scene'),
        ('yaw 270, elev 22', 270, 22, 'scene'),
        ('top down', 0, 89, 'scene'),
        ('whole cloud incl. sky', 30, 26, 'all'),
    ]
    fig = plt.figure(figsize=(21, 14), dpi=110)
    for i, (title, yaw, elev, kind) in enumerate(panels, 1):
        ax = fig.add_subplot(2, 3, i, projection='3d')
        pts, cols = sub(~sky if kind == 'scene' else np.ones(len(xyz), bool))
        x, y, z = frame(pts)
        ax.scatter(x, y, z, s=1.6 if kind == 'scene' else 0.6, c=cols, linewidths=0, marker='.')
        ax.view_init(elev=elev, azim=yaw)
        ax.set_box_aspect((np.ptp(x), np.ptp(z), np.ptp(y)))
        ax.set_title(title, fontsize=13)
        ax.set_axis_off()
        try:
            ax.set_proj_type('persp')
        except Exception:
            pass
    fig.suptitle('HY-World 2.0 generated outdoor cloud - new ground-level panorama '
                 '(2000000 points, sky excluded except bottom-right)', fontsize=15)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor='white')
    print('SAVED', args.out)


if __name__ == '__main__':
    main()
