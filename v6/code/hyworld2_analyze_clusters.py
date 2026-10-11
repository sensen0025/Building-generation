"""Break the generated cloud into colour clusters and zoom into the ground plate.

My first saturation-based sky rule did not work (sky is grey-blue, not neutral), so this
does it empirically: quantise RGB, table the biggest clusters with their height, then
render the ground region close up with the sky clusters dropped. No quality claim: a
generated scene has no ground truth.
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
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], 1).astype(np.float32)
    rgb = np.stack([raw["red"], raw["green"], raw["blue"]], 1).astype(np.uint8)
    return xyz, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=6, help="quantisation levels per channel")
    args = ap.parse_args()

    xyz, rgb = read_ply(args.ply)
    up = 2                                    # Z is up for this cloud (from the spread test)
    q = (rgb.astype(np.int32) * args.bins // 256)
    key = (q[:, 0] * args.bins + q[:, 1]) * args.bins + q[:, 2]
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)[:10]

    table = []
    for idx in order:
        sel = inv == idx
        centre = rgb[sel].mean(0)
        table.append({
            "count": int(counts[idx]),
            "rgb_mean": [int(v) for v in centre],
            "height_mean": round(float(xyz[sel, up].mean()), 4),
            "height_p95": round(float(np.percentile(xyz[sel, up], 95)), 4),
            "radius_mean": round(float(np.linalg.norm(xyz[sel][:, :2], axis=1).mean()), 4),
        })
    print("=== top colour clusters ===")
    print(json.dumps(table, indent=1))

    # sky = the clusters that sit high AND far from the plate centre
    plate = xyz[:, up] < (np.percentile(xyz[:, up], 35) + 0.15)
    print(f"\npoints on/near the ground plate: {int(plate.sum())}")
    print("plate rgb mean:", [int(v) for v in rgb[plate].mean(0)])

    lo = np.percentile(xyz[:, :2], 5, axis=0)
    hi = np.percentile(xyz[:, :2], 95, axis=0)
    print(f"plate horizontal bbox p5..p95: {np.round(lo,3)} .. {np.round(hi,3)}")

    # close-up of the ground plate, sky dropped by height
    fig, ax = plt.subplots(1, 3, figsize=(21, 7), dpi=110)
    sub = slice(None, None, max(1, len(xyz) // 300000))
    ground = plate

    a = ax[0]
    a.scatter(xyz[ground][sub, 0], xyz[ground][sub, 1], s=0.5, c=rgb[ground][sub] / 255, linewidths=0)
    a.set_title(f"GROUND PLATE top-down ({int(ground.sum())} pts)")
    a.set_aspect("equal")

    a = ax[1]
    mid = xyz[ground][:, up] > (np.percentile(xyz[ground][:, up], 60))
    a.scatter(xyz[ground][mid][sub, 0], xyz[ground][mid][sub, up], s=1.0,
              c=rgb[ground][mid][sub] / 255, linewidths=0)
    a.set_title("GROUND PLATE front: upper half only (anything standing)")
    a.set_aspect("equal")

    a = ax[2]
    tall = xyz[:, up] > (np.percentile(xyz[:, up], 60))
    r = np.linalg.norm(xyz[:, :2], axis=1)
    near = tall & (r < 0.9)
    a.scatter(xyz[near][sub, 0], xyz[near][sub, 1], s=1.0, c=rgb[near][sub] / 255, linewidths=0)
    a.set_title(f"central region r<0.9, upper half ({int(near.sum())} pts)")
    a.set_aspect("equal")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"\nSAVED {args.out}")


if __name__ == "__main__":
    main()
