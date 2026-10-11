"""What is actually inside the generated HY-World 2.0 outdoor cloud?

Answers, with numbers instead of impressions:
  * how many points are sky (the grey dome) vs scene, and how the split was made,
  * where the ground plane is, and how much sits above it (i.e. anything standing),
  * where the vertical structure is (candidate building) and how big it is in blocks
    at a stated non-metric scale.

No ground truth exists for a generated scene, so nothing here is a quality score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_ply(path):
    """Binary little-endian PLY with x,y,z + red,green,blue (uchar)."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fmt = f.readline().strip().decode()
        assert "binary_little_endian" in fmt, fmt
        n = None
        props = []
        while True:
            line = f.readline().strip().decode()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        dtype = []
        for p in props:
            if p in ("x", "y", "z", "nx", "ny", "nz"):
                dtype.append((p, "<f4"))
            elif p in ("red", "green", "blue", "alpha"):
                dtype.append((p, "u1"))
            else:
                dtype.append((p, "<f4"))
        raw = np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=np.dtype(dtype))
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float32)
    rgb = (np.stack([raw["red"], raw["green"], raw["blue"]], 1).astype(np.float32) / 255.0
           if "red" in props else np.full_like(xyz, 0.5))
    return xyz, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks-per-unit", type=float, default=16.0,
                    help="chosen non-metric mapping: how many blocks one source unit becomes")
    args = ap.parse_args()

    xyz, rgb = read_ply(args.ply)
    m = {
        "ply": str(args.ply),
        "points": int(len(xyz)),
        "extent_xyz": [round(float(v), 4) for v in (xyz.max(0) - xyz.min(0))],
        "bbox_min": [round(float(v), 4) for v in xyz.min(0)],
        "bbox_max": [round(float(v), 4) for v in xyz.max(0)],
        "rgb_mean": [round(float(v), 4) for v in rgb.mean(0)],
    }

    # --- which axis is up? the terrain plate is the flattest axis ---
    spreads = xyz.std(0)
    up = int(np.argmin(spreads))
    m["up_axis"] = "xyz"[up]
    m["spread_per_axis"] = [round(float(v), 4) for v in spreads]

    # --- colour rule for sky: near-neutral and bright-ish, plus far from the median ---
    mx, mn = rgb.max(1), rgb.min(1)
    sat = mx - mn
    neutral = sat < 0.06
    m["neutral_fraction"] = round(float(neutral.mean()), 4)

    # terrain/ground = greenish OR non-neutral; sky = neutral
    scene = ~neutral
    m["scene_points"] = int(scene.sum())
    m["neutral_points"] = int((~scene).sum())
    if scene.sum() < 100:
        raise SystemExit("too few non-neutral points; colour rule does not apply")

    # --- ground plane from the scene points, using the up axis ---
    upv = xyz[:, up]
    ground = float(np.percentile(upv[scene], 25))
    above = upv - ground
    m["ground_level"] = round(ground, 4)

    # --- what stands above the ground, and how tall ---
    hist, edges = np.histogram(above[scene], bins=40)
    m["height_histogram"] = {"edges": [round(float(e), 3) for e in edges],
                             "counts": [int(c) for c in hist]}
    tall = above > (edges[1] - edges[0]) * 2      # more than ~2 bins above ground
    m["above_ground_points"] = int((scene & tall).sum())

    # --- cluster the tall points on the horizontal plane to find structures ---
    other = [i for i in range(3) if i != up]
    tall_pts = xyz[scene & tall]
    m["tall_point_count"] = int(len(tall_pts))
    structures = []
    if len(tall_pts) > 500:
        # coarse 2D grid clustering at 2% of the scene's horizontal extent
        step = float(np.percentile(xyz[:, other].std(0).mean(), 50) * 0.25)
        step = max(step, 1e-3)
        keys = np.floor(tall_pts[:, other] / step).astype(np.int64)
        uniq, counts = np.unique(keys, axis=0, return_counts=True)
        order = np.argsort(-counts)[:8]
        for idx in order:
            cell = uniq[idx]
            sel = np.all(keys == cell, axis=1)
            pts = tall_pts[sel]
            lo, hi = pts.min(0), pts.max(0)
            structures.append({
                "cells": int(counts[idx]),
                "center": [round(float(v), 3) for v in pts.mean(0)],
                "size_src": [round(float(v), 3) for v in (hi - lo)],
                "height_src": round(float(hi[up] - lo[up]), 3),
                "height_blocks": round(float(hi[up] - lo[up]) * args.blocks_per_unit, 1),
            })
    m["candidate_structures"] = structures
    m["blocks_per_unit"] = args.blocks_per_unit

    # --- renders ---
    fig, ax = plt.subplots(2, 2, figsize=(18, 12), dpi=110)
    sm = scene
    sub = slice(None, None, max(1, len(xyz) // 250000))

    a = ax[0, 0]
    a.scatter(xyz[sm][sub, other[0]], xyz[sm][sub, other[1]], s=0.4, c=rgb[sm][sub], linewidths=0)
    a.set_title(f"TOP down (sky removed), {int(sm.sum())} scene pts")
    a.set_aspect("equal")

    a = ax[0, 1]
    a.scatter(xyz[sub, other[0]], xyz[sub, up], s=0.4, c=rgb[sub], linewidths=0)
    a.set_title("FRONT (sky kept = grey dome)")
    a.set_aspect("equal")

    a = ax[1, 0]
    a.scatter(xyz[sm][sub, other[0]], xyz[sm][sub, up], s=0.4, c=rgb[sm][sub], linewidths=0)
    a.set_title("FRONT (sky removed)")
    a.set_aspect("equal")

    a = ax[1, 1]
    if len(tall_pts) > 500:
        tsub = tall_pts[slice(None, None, max(1, len(tall_pts) // 120000))]
        a.scatter(tsub[:, other[0]], tsub[:, up], s=2.0, linewidths=0,
                  c=rgb[scene & tall][slice(None, None, max(1, len(tall_pts) // 120000))])
    a.set_title(f"things ABOVE ground ({int(tall.sum())} pts) - candidate structures")
    a.set_aspect("equal")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(json.dumps(m, indent=2)[:4000])
    print(f"\nSAVED {args.out}")


if __name__ == "__main__":
    main()
