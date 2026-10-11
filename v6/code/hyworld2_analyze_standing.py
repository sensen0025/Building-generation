"""Decisive test: does the generated scene contain anything standing on the ground?

The colour table shows two clearly separated populations: dark green (ground, mean rgb
[56,77,23]) and blue-grey (mean rgb ~[121,135,142] / [112,123,133] / [99,111,113]).
A sky dome from an equirect panorama is blue-grey, so this drops blue-grey points and
asks what is left above the ground plane. If a building exists, its walls must survive
as clusters taller than one block at a stated non-metric scale.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_ply(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        assert b"binary_little_endian" in f.readline()
        n, props = None, []
        while True:
            line = f.readline().strip().decode()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        dtype = [(p, "<f4" if p in ("x", "y", "z", "nx", "ny", "nz")
                  else ("u1" if p in ("red", "green", "blue", "alpha") else "<f4"))
                 for p in props]
        raw = np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=np.dtype(dtype))
    return (np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float32),
            np.stack([raw["red"], raw["green"], raw["blue"]], 1).astype(np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks-per-unit", type=float, default=16.0)
    args = ap.parse_args()

    xyz, rgb = read_ply(args.ply)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    up = 2

    bluegrey = (b >= r) & ((b - r) >= 3) & (np.abs(b - g) <= 25) & (b >= 70)
    print(f"blue-grey (sky) points: {int(bluegrey.sum())} ({bluegrey.mean()*100:.1f}%)")
    solid = ~bluegrey
    print(f"remaining points:       {int(solid.sum())}")

    z = xyz[:, up]
    ground = float(np.percentile(z[solid], 50))
    above = z - ground
    unit = args.blocks_per_unit

    print(f"ground z (median of remaining) = {ground:.4f}")
    for cut_blocks in (1, 2, 3, 5, 8):
        thr = cut_blocks / unit
        n = int((solid & (above > thr)).sum())
        print(f"  points above ground by >{cut_blocks:>2} blocks ({thr:.4f} src): {n}")

    thr = 2 / unit
    stand = solid & (above > thr)
    pts = xyz[stand]
    cols = rgb[stand] / 255.0
    print(f"\nstanding candidates at >2 blocks: {len(pts)} points")

    fig, ax = plt.subplots(1, 3, figsize=(21, 7), dpi=110)
    sub = slice(None, None, max(1, len(pts) // 200000)) if len(pts) else slice(0)

    a = ax[0]
    if len(pts):
        a.scatter(pts[sub, 0], pts[sub, 1], s=2.0, c=cols[sub], linewidths=0)
    a.set_title(f"TOP-DOWN of everything >2 blocks above ground ({len(pts)} pts)")
    a.set_aspect("equal")

    a = ax[1]
    if len(pts):
        a.scatter(pts[sub, 0], pts[sub, up], s=2.0, c=cols[sub], linewidths=0)
    a.set_title("FRONT of the same (Z up); a wall would show as a vertical band")
    a.set_aspect("equal")

    a = ax[2]
    s2 = slice(None, None, max(1, len(xyz) // 250000))
    a.scatter(xyz[solid][s2, 0], xyz[solid][s2, 1], s=0.4, c=rgb[solid][s2] / 255.0, linewidths=0)
    a.set_title(f"TOP-DOWN, sky (blue-grey) removed: {int(solid.sum())} pts")
    a.set_aspect("equal")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)

    out = {
        "sky_points": int(bluegrey.sum()),
        "solid_points": int(solid.sum()),
        "ground_z": round(ground, 4),
        "standing_gt2_blocks": int(len(pts)),
        "blocks_per_unit": unit,
    }
    print(json.dumps(out, indent=1))
    print(f"SAVED {args.out}")


if __name__ == "__main__":
    main()
