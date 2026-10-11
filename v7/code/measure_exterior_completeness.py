"""Measure how COMPLETE a generated building's exterior surface is.

Written for v7 (the "complete exterior wall" run). Every number is defined here so the v6 baseline
and the v7 result are directly comparable, and none of them is a proxy for "looks nice":

  azimuth_coverage   the standing structure's points binned into 24 azimuth sectors around its own
                     centroid; a sector counts as covered when it holds >= --min-per-sector points.
                     A missing sector is a wall nobody observed.
  quadrant_bands     the 4 principal quadrants (E/S/W/N) x 3 height bands must all be non-empty:
                     "walled on every side, floor to roof" as a boolean grid.
  largest_component  occupied cells of the standing structure voxelised at one block, labelled
                     6-connected; the biggest label's share of them. Scattered fragments mean the
                     surface came back in pieces.
  standing_height    points above the ground plane by >1/2/3/5/8 blocks (blocks_per_unit is a CHOSEN
                     non-metric scale, so this is a shape statement, not metres).
  sky_fraction       blue-grey share, the same rule the deployment filter uses.
  building_ratio     building height / whole-scene width: a complete BUILDING must not be a detail
                     inside a landscape.

No ground truth exists for a generated scene, so nothing here is an accuracy or quality score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/run/media/sensen/Data1/minecraft/AI_3D_World_Repos')
SECTORS = 24


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ply', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='output prefix (json + png)')
    ap.add_argument('--blocks-per-unit', type=float, default=8.0)
    ap.add_argument('--standing-blocks', type=float, default=1.0)
    ap.add_argument('--min-per-sector', type=int, default=200)
    ap.add_argument('--min-per-band', type=int, default=100)
    ap.add_argument('--column-blocks', type=float, default=6.0,
                   help='column height (blocks) above which a cell column counts as building, not terrain')
    ap.add_argument('--label', default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    from tools.cloudbuild.hyworld2_cloud_infer import read_ply, sky_mask, hyworld_to_minecraft

    xyz, rgb = read_ply(args.ply)
    sky = sky_mask(xyz, rgb)
    keep = ~sky
    mc = hyworld_to_minecraft(xyz[keep]) * args.blocks_per_unit
    print(f'points {len(xyz)}  sky {sky.mean()*100:.1f}%  scene pts {len(mc)}')

    up = 1
    ground = float(np.percentile(mc[:, up], 25))
    above = mc[:, up] - ground
    standing = above > args.standing_blocks
    b = mc[standing]
    b_rgb = rgb[keep][standing]
    print(f'ground {ground:.2f}  standing points {len(b)}')

    m = {'ply': str(args.ply), 'label': args.label or args.ply.stem,
         'points': int(len(xyz)), 'sky_fraction': round(float(sky.mean()), 4),
         'blocks_per_unit': args.blocks_per_unit, 'ground_blocks': round(ground, 3),
         'standing_points': int(len(b)), 'sectors': SECTORS}

    # ---- standing height histogram ----
    for cut in (1, 2, 3, 5, 8):
        m[f'above_{cut}_blocks'] = int((above[standing] > cut).sum())

    # ---- azimuth coverage around the building's own centroid ----
    # Isolate the BUILDING from the terrain by COLUMN HEIGHT, not by absolute elevation.
    # Two earlier attempts failed for a reason worth recording: "above ground by k blocks" is still
    # the terrain whenever the ground itself rolls more than k blocks (the v6 cloud's ground sheet is
    # hilly, so azimuth coverage came out 24/24 and the "building" spanned 95 blocks, i.e. the metric
    # could not distinguish a complete facade from one wall). A terrain column is 1-3 cells tall; a
    # building column is not. So: voxelise, take each (x,z) column's height, keep the tall columns,
    # and let the largest 8-connected group of them be the building's footprint.
    idx_all = np.floor(b).astype(np.int64)
    base = idx_all.min(0)
    idx_all0 = idx_all - base
    shape = tuple((idx_all0.max(0) + 1).tolist())
    occ_all = np.zeros(shape, bool)
    occ_all[tuple(idx_all0.T)] = True
    ys = np.arange(shape[1])[None, :, None]          # broadcast over (x, y, z)
    occ_any = occ_all.any(axis=1)                    # (x, z) column has any point
    col_lo = np.where(occ_all, ys, 10 ** 6).min(axis=1)
    col_hi = np.where(occ_all, ys, -1).max(axis=1)
    col_height = np.where(occ_any, col_hi - col_lo + 1, 0)
    tall_cols = col_height > args.column_blocks
    m['column_blocks'] = args.column_blocks
    m['tall_columns'] = int(tall_cols.sum())
    if tall_cols.any():
        lab2, n2 = ndimage.label(tall_cols, structure=np.ones((3, 3), int))
        sizes2 = np.bincount(lab2.ravel())
        sizes2[0] = 0
        keep = int(sizes2.argmax())
        footprint = lab2 == keep
        m['footprint_columns'] = int(footprint.sum())
        m['footprint_groups'] = int(n2)
        fx = np.nonzero(footprint.any(axis=1))[0]
        fz = np.nonzero(footprint.any(axis=0))[0]
        fe = [int(fx.max() - fx.min() + 1), int(fz.max() - fz.min() + 1)]
        m['footprint_extent_blocks'] = fe
        m['footprint_aspect'] = round(max(fe) / max(min(fe), 1), 2)
        m['footprint_fill'] = round(float(footprint.sum() / (fe[0] * fe[1])), 3)
        cell_xz = idx_all0[:, [0, 2]]
        keep_point = footprint[cell_xz[:, 0], cell_xz[:, 1]]
        building_pts = b[keep_point]
        building_rgb = b_rgb[keep_point]
    else:
        building_pts = b
    m['building_points'] = int(len(building_pts))
    if len(building_pts) < 500:
        print('WARNING: footprint isolation left too few points; using all standing points')
        building_pts, building_rgb = b, b_rgb
    print(f'isolated building: {len(building_pts)} points from a {int(footprint.sum()) if tall_cols.any() else 0}-column footprint '
          f'(tall columns {int(tall_cols.sum())} of {int((col_height > 0).sum())})')
    b, b_rgb = building_pts, building_rgb

    cx, cz = float(np.median(b[:, 0])), float(np.median(b[:, 2]))
    az = (np.degrees(np.arctan2(b[:, 2] - cz, b[:, 0] - cx)) + 360.0) % 360.0
    counts, _ = np.histogram(az, bins=SECTORS, range=(0, 360))
    covered = counts >= args.min_per_sector
    m['azimuth_counts'] = [int(c) for c in counts]
    m['azimuth_covered_sectors'] = int(covered.sum())
    m['azimuth_coverage'] = round(float(covered.mean()), 4)
    m['centroid_xz'] = [round(cx, 3), round(cz, 3)]

    # ---- 4 quadrants x 3 height bands ----
    lo, hi = float(np.percentile(b[:, up], 2)), float(np.percentile(b[:, up], 98))
    edges = np.linspace(lo, hi + 1e-6, 4)
    grid = {}
    for qi, qname in enumerate(('E', 'S', 'W', 'N')):
        sel_q = (az >= qi * 90) & (az < (qi + 1) * 90)
        row = []
        for bi in range(3):
            sel = sel_q & (b[:, up] >= edges[bi]) & (b[:, up] < edges[bi + 1])
            row.append(int(sel.sum()))
        grid[qname] = row
    m['quadrant_bands'] = grid
    m['quadrant_bands_all_filled'] = bool(all(v >= args.min_per_band for row in grid.values() for v in row))
    m['quadrant_bands_min'] = int(min(v for row in grid.values() for v in row))

    # ---- connectivity of the standing structure at 1 block ----
    idx = np.floor(b).astype(np.int64)
    idx -= idx.min(0)
    shape = idx.max(0) + 1
    occ = np.zeros(tuple(shape.tolist()), bool)
    occ[tuple(idx.T)] = True
    lab, n = ndimage.label(occ)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    m['occupied_cells'] = int(occ.sum())
    m['components'] = int(n)
    m['largest_component_share'] = round(float(sizes.max() / max(occ.sum(), 1)), 4)

    # ---- building size vs whole scene ----
    scene_span = float((mc.max(0) - mc.min(0)).max())
    b_span = (b.max(0) - b.min(0)) if len(b) else np.zeros(3)
    m['building_height_blocks'] = round(float(b[:, up].max() - ground), 2)
    m['building_span_blocks'] = [round(float(v), 1) for v in b_span]
    m['scene_span_blocks'] = round(scene_span, 1)
    m['building_ratio'] = round(float((b[:, up].max() - ground) / max(scene_span, 1e-6)), 4)

    # ---- figure: azimuth rose + four facades ----
    fig = plt.figure(figsize=(20, 10), dpi=110)
    ax = fig.add_subplot(2, 3, 1, projection='polar')
    theta = np.radians(np.arange(SECTORS) * 360 / SECTORS + 180 / SECTORS)
    widths = np.radians(360 / SECTORS) * 0.9
    ax.bar(theta, counts, width=widths,
           color=['#2e7d32' if c >= args.min_per_sector else '#c62828' for c in counts])
    ax.set_title(f'azimuth coverage {covered.sum()}/{SECTORS} sectors\n'
                 f'({covered.mean()*100:.0f}%, need >={args.min_per_sector} pts)', fontsize=12)

    for qi, (qname, lo_a, hi_a) in enumerate((('E', 0, 90), ('S', 90, 180), ('W', 180, 270), ('N', 270, 360))):
        a2 = fig.add_subplot(2, 3, qi + 2)
        sel = (az >= lo_a) & (az < hi_a)
        pts = b[sel]
        if len(pts):
            # horizontal axis = the in-quadrant direction, vertical = height
            ang = np.radians(az[sel])
            horiz = np.cos(ang - np.radians(lo_a + 45)) * np.hypot(pts[:, 0] - cx, pts[:, 2] - cz)
            s = slice(None, None, max(1, len(pts) // 60000))
            a2.scatter(horiz[s], pts[s, up], s=1.5, c=b_rgb[sel][s], linewidths=0)
        a2.set_title(f'{qname} facade  bands {grid[qname]}  '
                     f'{"OK" if min(grid[qname]) >= args.min_per_band else "GAP"}', fontsize=11)
        a2.set_aspect('equal')
        a2.set_xlabel('lateral (blocks)')
        a2.set_ylabel('height (blocks)')
    fig.suptitle(f'exterior completeness - {m["label"]}   '
                 f'(sky {m["sky_fraction"]*100:.1f}%, largest component '
                 f'{m["largest_component_share"]*100:.0f}% of {m["occupied_cells"]} cells, '
                 f'building/scene {m["building_ratio"]*100:.0f}%)', fontsize=14)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.png'), facecolor='white')
    out.with_suffix('.json').write_text(json.dumps(m, indent=2) + '\n')
    print(json.dumps({k: v for k, v in m.items() if k not in ('azimuth_counts', 'quadrant_bands')}, indent=1))
    print('SAVED', out.with_suffix('.png'), 'and', out.with_suffix('.json'))


if __name__ == '__main__':
    main()
