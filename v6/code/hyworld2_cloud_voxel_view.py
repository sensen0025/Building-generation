"""Show the generated HY-World 2.0 cloud as Minecraft blocks through the REAL renderer.

The point cloud is the input, so this only DISPLAYS it: each cell that contains at least one point
becomes one block, and the block is chosen from the cell's mean colour by a fixed, stated palette.
Nothing is generated, denoised or completed here - that is the stage-2/3 model's job, and its output
is rendered separately.

The frame contract is imported from hyworld2_cloud_infer so this view cannot drift from the frame the
trained models were fed (same signed axis map, same chosen blocks-per-unit, same crop rule).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/run/media/sensen/Data1/minecraft/AI_3D_World_Repos')
SKY = 'minecraft:air'

# Display palette: mean cell colour -> block. Deliberately tiny and fixed; this is a viewer, not a
# material model. Thresholds are stated in RGB terms so the choice is auditable.
PALETTE = (
    # (name, test(r,g,b))
    ('minecraft:water[level=0]', lambda r, g, b: b > r + 0.06 and b > 0.35 and g > r),
    ('minecraft:grass_block[snowy=false]',
                               lambda r, g, b: g > r + 0.04 and g > b + 0.04 and g > 0.30),
    # full blocks only: states with properties must be spelled out completely for the registry,
    # and this is a viewer, so no property-carrying block is used except the two that need one.
    ('minecraft:moss_block',   lambda r, g, b: g >= r and g >= b and g > 0.16),
    ('minecraft:dirt',         lambda r, g, b: r > g and g > b and r < 0.55),
    ('minecraft:oak_planks',   lambda r, g, b: r > g >= b and r >= 0.55),
    ('minecraft:stone_bricks', lambda r, g, b: max(r, g, b) - min(r, g, b) < 0.05 and r < 0.45),
    ('minecraft:stone',        lambda r, g, b: max(r, g, b) - min(r, g, b) < 0.05),
    ('minecraft:sand',         lambda r, g, b: True),
)


def block_for(rgb):
    r, g, b = (float(v) for v in rgb)
    for name, test in PALETTE:
        if test(r, g, b):
            return name
    return 'minecraft:stone'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--name', default='hyworld2_cloud')
    ap.add_argument('--blocks-per-unit', type=float, default=8.0)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--whole-scene', action='store_true',
                    help='voxelise the entire cloud instead of the deployment crop')
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, '/run/media/sensen/Data1/minecraft/redstone_v5_local/code')
    from tools.cloudbuild.hyworld2_cloud_infer import read_ply, sky_mask, hyworld_to_minecraft
    from tools.cloudbuild.gs2blocks_data import Vocabulary
    from mcfix.containers import export_litematic, decode_path

    xyz, rgb = read_ply(args.ply)
    sky = sky_mask(xyz, rgb)
    xyz, rgb = xyz[~sky], rgb[~sky]
    mc = hyworld_to_minecraft(xyz) * args.blocks_per_unit
    print(f'points {len(mc)} after sky removal (sky was {sky.mean()*100:.1f}%)')

    origin = np.floor(mc.min(0)).astype(np.int64) - 1
    extent = np.ceil(mc.max(0)).astype(np.int64) + 1 - origin
    if not args.whole_scene:
        side = args.window
        ground = np.percentile(mc[:, 1], 25)
        standing = (mc[:, 1] - ground) > (args.blocks_per_unit * 0.4)
        best, best_n = None, -1
        step = max(side // 4, 1)
        for x0 in range(int(mc[:, 0].min()), int(mc[:, 0].max()) + 1, step):
            for z0 in range(int(mc[:, 2].min()), int(mc[:, 2].max()) + 1, step):
                n = int((standing & (mc[:, 0] >= x0) & (mc[:, 0] < x0 + side)
                         & (mc[:, 2] >= z0) & (mc[:, 2] < z0 + side)).sum())
                if n > best_n:
                    best, best_n = (x0, z0), n
        origin = np.array([best[0], int(np.floor(ground)) - 2, best[1]], np.int64)
        extent = np.array([side, side, side], np.int64)

    index = np.floor(mc - origin).astype(np.int64)
    inside = np.all((index >= 0) & (index < extent), axis=1)
    index, cols = index[inside], rgb[inside]
    print(f'cell volume {tuple(extent.tolist())}  points inside {int(inside.sum())}')

    counts = np.zeros(tuple(extent.tolist()), np.uint32)
    sums = np.zeros(tuple(extent.tolist()) + (3,), np.float64)
    flat = np.ravel_multi_index((index[:, 0], index[:, 1], index[:, 2]), tuple(extent.tolist()))
    np.add.at(counts.reshape(-1), flat, 1)
    for ch in range(3):
        np.add.at(sums.reshape(-1, 3)[:, ch], flat, cols[:, ch])
    mean = np.zeros(tuple(extent.tolist()) + (3,), np.float32)
    occ = counts > 0
    mean[occ] = (sums[occ] / counts[occ][:, None]).astype(np.float32)

    vocab = Vocabulary()
    names = np.empty(tuple(extent.tolist()), object)
    names[:] = SKY
    uniq_cache = {}
    coords = np.argwhere(occ)
    for c in coords:
        key = tuple(np.round(mean[tuple(c)] * 64).astype(int))
        name = uniq_cache.get(key)
        if name is None:
            name = block_for(mean[tuple(c)])
            uniq_cache[key] = name
        names[tuple(c)] = name
    print(f'occupied cells {int(occ.sum())}  distinct blocks {len(set(names[occ].ravel()))}')

    palette = sorted(set(names[occ].ravel()) | {SKY})
    lut = {n: i for i, n in enumerate(palette)}
    vol = np.vectorize(lut.__getitem__, otypes=[np.int32])(names)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    path = out / f'{args.name}.litematic'
    export_litematic(path, vol, palette, origin=tuple(int(v) for v in origin),
                     data_version=4189, name=args.name[:40])
    decoded = decode_path(path)
    # decode_path indexes its volume as [y][z][x] and returns palette ORDER, so compare NAMES
    # cell by cell instead of mixing palette indices with registry state ids.
    names_written = np.transpose(names, (1, 2, 0))
    names_back = np.array(decoded['names'])[np.asarray(decoded['vol'])]
    if names_back.shape != names_written.shape or not (names_back == names_written).all():
        bad = int((names_back != names_written).sum()) if names_back.shape == names_written.shape else -1
        raise AssertionError(f'round-trip mismatch on {bad} cells')
    print(f'round-trip exact on all {names_back.size} cells')
    (out / f'{args.name}.json').write_text(json.dumps({
        'viewer_only': True, 'ply': str(args.ply), 'blocks_per_unit': args.blocks_per_unit,
        'origin_xyz': origin.tolist(), 'shape': extent.tolist(),
        'occupied_cells': int(occ.sum()), 'sky_removed_fraction': round(float(sky.mean()), 4),
        'palette': palette,
        'note': 'each block is one cell containing at least one cloud point; colour->block is a '
                'fixed display palette, NOT a trained material prediction',
    }, indent=2) + '\n')
    print('SAVED', path)


if __name__ == '__main__':
    main()
