"""Compose the panorama ANALYTICALLY from a real Minecraft render, instead of letting HY-Pano do it.

Why this exists (v6/v7 evidence, three attempts): the pano stage is image-conditioned, and at every
seed framing it did the same two things to a Minecraft building - shrunk it, and lifted it off the
ground. Measured outcomes:
  * seed filling the frame, transparent background  -> building tiny at the horizon, invented curved
    facades at the left/right edges (which in an equirect are the SAME azimuth: behind the camera),
  * equirect-shaped seed with the building at 42% width, horizon at 50% -> same, plus a floated house,
  * large seed (80% width), neutral prompt -> edges clean, but the house hovered above the horizon.
So the layout is composed here instead: a perspective render of the real build, reprojected exactly
through its own camera pose into equirect, over an analytic sky and ground plane. Every part of the
layout is then controlled and checkable, and the generative step is skipped entirely.

The building's own base slab is hidden by the ground plane: ground wins for every below-horizon ray,
so the build reads as planted in the ground rather than floating.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render', type=Path, required=True, help='perspective render, alpha = build')
    ap.add_argument('--eye', type=float, nargs=3, required=True)
    ap.add_argument('--target', type=float, nargs=3, required=True)
    ap.add_argument('--fov-v', type=float, required=True, help='vertical FOV of the render, degrees')
    ap.add_argument('--ground-y', type=float, default=0.0)
    ap.add_argument('--width', type=int, default=1952)
    ap.add_argument('--height', type=int, default=960)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path, default=None)
    args = ap.parse_args()

    img = Image.open(args.render).convert('RGBA')
    W, H = img.size
    rgba = np.asarray(img).astype(np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32)
    opaque = rgba[:, :, 3] > 128
    print(f'render {W}x{H}  opaque {opaque.mean()*100:.1f}%')

    eye = np.array(args.eye, np.float64)
    tgt = np.array(args.target, np.float64)
    f = normalize(tgt - eye)
    world_up = np.array([0.0, 1.0, 0.0])
    r = normalize(np.cross(f, world_up))
    u = np.cross(r, f)
    tan_v = np.tan(np.radians(args.fov_v) / 2)
    tan_h = tan_v * (W / H)

    PW, PH = args.width, args.height
    ex = np.arange(PW) + 0.5
    ey = np.arange(PH) + 0.5
    theta = (ex / PW) * 2 * np.pi - np.pi          # -pi..pi, hemisphere wrap
    phi = (0.5 - ey / PH) * np.pi                  # +pi/2 (up) .. -pi/2 (down)
    TH, PHI = np.meshgrid(theta, phi)              # [PH, PW]
    # Align the panorama so the render camera's own forward direction sits at azimuth 0, i.e. the
    # building lands in the middle of the image instead of wherever +Z happens to be.
    az0 = np.arctan2(f[0], f[2])
    TH_W = TH + az0
    cos_phi = np.cos(PHI)
    dirs = np.stack([np.sin(TH_W) * cos_phi, np.sin(PHI), np.cos(TH_W) * cos_phi], -1)   # [PH,PW,3]

    # ---- analytic sky and ground ----
    out = np.zeros((PH, PW, 3), np.float32)
    sky_top, sky_bot = np.array([104, 162, 222], np.float32), np.array([198, 222, 242], np.float32)
    t = np.clip(PHI / (np.pi / 2), 0, 1)
    out[:] = sky_top * (1 - t[..., None]) + sky_bot * t[..., None]

    cam_h = eye[1] - args.ground_y
    below = PHI < 0
    # distance along the ground for each below-horizon ray
    dist = np.where(below, cam_h / np.maximum(-np.sin(PHI), 1e-6), 0.0)
    grass_far, grass_near = np.array([132, 176, 88], np.float32), np.array([88, 128, 56], np.float32)
    g = np.clip(dist / 60.0, 0, 1)
    grass = grass_far * (1 - g[..., None]) + grass_near * g[..., None]
    # soft grass-block checker that fades with distance (visual cue for ground slope/scale)
    step_world = 2.0
    gx = np.floor(dirs[:, :, 0] * dist / step_world)
    gz = np.floor(dirs[:, :, 2] * dist / step_world)
    checker = ((gx + gz) % 2 == 0).astype(np.float32)
    grass *= (1.0 + 0.05 * checker[..., None] * (1 - g)[..., None])
    out[below] = grass[below]

    # ---- project the render into the pano for above-horizon rays ----
    above = ~below
    fx = (dirs @ r) / tan_h
    fy = (dirs @ u) / tan_v
    fz = dirs @ f
    valid = above & (fz > 1e-6)
    # NO clamping: clamping smeared the render's border pixels (the chimney) across the whole
    # azimuth range, which is the classic equirect reprojection artefact. A ray only samples the
    # render when it genuinely falls inside the frame.
    inside = np.zeros_like(valid)
    m = 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        px = ((fx / np.maximum(fz, 1e-6)) * 0.5 + 0.5) * W
        py = (0.5 - (fy / np.maximum(fz, 1e-6)) * 0.5) * H
    inside = valid & (px >= m) & (px < W - m) & (py >= m) & (py < H - m)
    xi = np.clip(np.rint(np.where(inside, px, 0)), 0, W - 1).astype(np.int64)
    yi = np.clip(np.rint(np.where(inside, py, 0)), 0, H - 1).astype(np.int64)
    hit = inside & opaque[yi, xi]
    out[hit] = rgb[yi[hit], xi[hit]]
    print(f'pano pixels: building {hit.mean()*100:.1f}%  ground {below.mean()*100:.1f}%  '
          f'sky {(above & ~hit).mean()*100:.1f}%')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(args.out)

    # seam check: the two columns either side of the wrap must agree (analytic ground/sky do)
    a = np.clip(out, 0, 255).astype(np.uint8)
    seam = float(np.abs(a[:, 0].astype(np.int16) - a[:, -1].astype(np.int16)).mean())
    report = {'render': str(args.render), 'eye': list(eye), 'target': list(tgt),
              'fov_v': args.fov_v, 'ground_y': args.ground_y,
              'building_pixels_fraction': round(float(hit.mean()), 4),
              'ground_fraction': round(float(below.mean()), 4),
              'seam_mean_abs_diff': round(seam, 3),
              'note': 'analytic equirect from a real render; HY-Pano skipped on purpose'}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('SAVED', args.out, '| seam diff', round(seam, 2))


if __name__ == '__main__':
    main()
