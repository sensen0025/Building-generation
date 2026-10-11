"""Three-way comparison: point cloud -> voxels -> model output, at one scale, on a clean background.

Fixes over the first attempt: the renderer writes RGBA with a transparent background, so converting
to RGB turned it black - the alpha channel is now used as a paste mask on white; and every panel is
cropped to its own content and scaled to the same height so the three representations can be
compared directly instead of each floating in its own padding.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/run/media/sensen/Data1/minecraft/AI_3D_World_Repos')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/run/media/sensen/Data1/minecraft/redstone_v5_local/code')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.cloudbuild.hyworld2_cloud_infer import read_ply, sky_mask, hyworld_to_minecraft

PLY = Path('/home/sensen/Desktop/hyworld2_exterior_ref/new2_aligned_pcd.ply')
V = ROOT / 'outputs/cloudbuild/hyworld2_deploy_v1'
OUT = Path('/home/sensen/Desktop/hyworld2_exterior_ref/three_way_comparison.png')
PANEL_H = 900
HEADER = 96
FOOTER = 54


def on_white(path):
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA'):
        rgba = im.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, (0, 0), rgba)
        return bg
    return im.convert('RGB')


def crop_content(im, pad=24):
    a = np.asarray(im.convert('L'))
    ink = a < 245
    if not ink.any():
        return im
    ys, xs = np.nonzero(ink)
    lo_y, hi_y = max(ys.min() - pad, 0), min(ys.max() + pad, im.height)
    lo_x, hi_x = max(xs.min() - pad, 0), min(xs.max() + pad, im.width)
    return im.crop((lo_x, lo_y, hi_x, hi_y))


def point_panel():
    import json
    meta = json.loads((V / 'cloudview/hyworld2_cloud.json').read_text())
    origin, side = np.array(meta['origin_xyz'], np.int64), meta['shape'][0]
    xyz, rgb = read_ply(PLY)
    keep = ~sky_mask(xyz, rgb)
    mc = hyworld_to_minecraft(xyz[keep]) * meta['blocks_per_unit']
    rgb, xyz = rgb[keep], xyz[keep]
    idx = np.floor(mc - origin).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < side), axis=1)
    p, c = mc[inside], rgb[inside]
    n = len(p)
    if n > 400000:
        p, c = p[::n // 400000], c[::n // 400000]

    fig = plt.figure(figsize=(9, 9), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2.4, c=c, linewidths=0, marker='.')
    ax.view_init(elev=25, azim=-60)
    ax.set_box_aspect((np.ptp(p[:, 0]), np.ptp(p[:, 1]), np.ptp(p[:, 2])))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    tmp = V / 'point_panel.png'
    fig.savefig(tmp, facecolor='white')
    plt.close(fig)
    return tmp, len(inside)


def main():
    pt_path, n_inside = point_panel()
    items = [
        (pt_path, '1  MODEL-GENERATED POINT CLOUD', f'{n_inside:,} points in this window',
         'this is the raw HY-World 2.0 output', (32, 64, 116)),
        (V / 'cmp_cloud/hyworld2_cloud_view000.png', '2  SAME POINTS, VOXELISED',
         'each occupied cell -> 1 block = 8,656 blocks',
         'colour -> block by a FIXED display palette', (122, 76, 22)),
        (V / 'cmp_model/hyworld2_scene_view000.png', '3  MODEL OUTPUT (Flow + material)',
         '7,846 cells, 147 legal block states',
         'same window, same camera as panel 2', (22, 104, 62)),
    ]
    panels = []
    for path, title, l1, l2, colour in items:
        im = crop_content(on_white(path))
        scale = PANEL_H / im.height
        panels.append((im.resize((max(int(im.width * scale), 1), PANEL_H)), title, l1, l2, colour))

    width = sum(p[0].width for p in panels) + 40 * (len(panels) + 1)
    sheet = Image.new('RGB', (width, HEADER + PANEL_H + FOOTER), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    try:
        fb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
        fs = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    except Exception:
        fb = fs = None

    x = 40
    for im, title, l1, l2, colour in panels:
        sheet.paste(im, (x, HEADER))
        d.rectangle([x, 0, x + im.width, HEADER - 6], fill=colour)
        d.text((x + 12, 8), title, fill='white', font=fb)
        d.text((x + 12, 48), l1, fill='white', font=fs)
        d.text((x, HEADER + PANEL_H + 6), l2, fill=(60, 60, 60), font=fs)
        x += im.width + 40
    sheet.save(OUT)
    print('SAVED', OUT, sheet.size)


if __name__ == '__main__':
    main()
