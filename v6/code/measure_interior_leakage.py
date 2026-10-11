"""Measure how much of the corpus' evidence is INTERIOR leakage - i.e. the learning signal for A.

Goal (chosen direction): train the Flow on pairs where the input evidence is what an outside scan
sees (the exterior surface) and the target is the real solid build. Before building that dataset,
quantify how much evidence would actually be removed, because if the interior share is tiny the
signal is tiny.

Method per building, no guessing:
  solid   = labels != air
  outside = air cells 6-connected-reachable from the volume border (flood fill)
  shell   = solid cells with at least one `outside` 6-neighbour  (what an outside scan can see)
  keep    = evidence cells within `shell` dilated by dilate cells (real scans also land just outside)
  drop    = evidence cells that are solid-interior or otherwise not visible from outside
"""
import argparse
import glob
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy import ndimage

AIR = 0


def analyse(path, dilate=1):
    with np.load(path) as z:
        labels, counts = z['labels'], z['counts']
    solid = labels != AIR
    if not solid.any():
        return None
    air = ~solid
    # outside = air reachable from the border without crossing solids
    seed = np.zeros_like(air)
    seed[0, :, :] = seed[-1, :, :] = True
    seed[:, 0, :] = seed[:, -1, :] = True
    seed[:, :, 0] = seed[:, :, -1] = True
    seed &= air
    outside = ndimage.binary_propagation(seed, mask=air)
    # a solid cell is on the exterior surface if it touches the outside region
    shell = solid & ndimage.binary_dilation(outside)
    visible = ndimage.binary_dilation(shell, iterations=dilate) if dilate else shell
    ev = counts > 0
    keep = ev & visible
    drop = ev & ~visible
    return dict(
        building=Path(path).stem,
        bbox=list(labels.shape),
        solid_cells=int(solid.sum()),
        evidence_cells=int(ev.sum()),
        evidence_ratio=round(float(ev.sum() / max(solid.sum(), 1)), 3),
        shell_cells=int(shell.sum()),
        evidence_kept=int(keep.sum()),
        evidence_dropped_interior=int(drop.sum()),
        dropped_fraction=round(float(drop.sum() / max(ev.sum(), 1)), 4),
        solid_fraction_of_bbox=round(float(solid.mean()), 4),
        evidence_fraction_of_bbox=round(float(ev.mean()), 4),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path,
                    default=Path('/run/media/sensen/Data1/minecraft/AI_3D_World_Repos/'
                                 'outputs/cloudbuild/gs2blocks_all'))
    ap.add_argument('--count', type=int, default=40)
    ap.add_argument('--dilate', type=int, default=1)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.data / '*.npz')))[:args.count]
    rows = [r for r in (analyse(f, args.dilate) for f in files) if r]
    if not rows:
        raise SystemExit('no usable buildings')
    agg = {
        'buildings': len(rows),
        'mean_evidence_ratio': round(float(np.mean([r['evidence_ratio'] for r in rows])), 3),
        'mean_solid_fraction_of_bbox': round(float(np.mean([r['solid_fraction_of_bbox'] for r in rows])), 4),
        'mean_evidence_fraction_of_bbox': round(float(np.mean([r['evidence_fraction_of_bbox'] for r in rows])), 4),
        'total_evidence': int(sum(r['evidence_cells'] for r in rows)),
        'total_dropped_interior': int(sum(r['evidence_dropped_interior'] for r in rows)),
        'mean_dropped_fraction': round(float(np.mean([r['dropped_fraction'] for r in rows])), 4),
        'median_dropped_fraction': round(float(np.median([r['dropped_fraction'] for r in rows])), 4),
        'buildings_with_zero_drop': int(sum(1 for r in rows if r['evidence_dropped_interior'] == 0)),
        'dilate': args.dilate,
    }
    print(json.dumps(agg, indent=2))
    print('\nper-building (first 12):')
    for r in rows[:12]:
        print(f"  {r['building'][:12]}  bbox {str(r['bbox']):>14}  证据/实体 {r['evidence_ratio']:>5}  "
              f"可见壳 {r['shell_cells']:>6}  挖掉 {r['evidence_dropped_interior']:>6} "
              f"({r['dropped_fraction']*100:5.1f}%)")
    if args.out:
        args.out.write_text(json.dumps({'aggregate': agg, 'per_building': rows}, indent=2) + '\n')
        print('SAVED', args.out)


if __name__ == '__main__':
    main()
