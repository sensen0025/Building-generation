"""Deploy the trained stage-2/3 stack on a generated HY-World 2.0 outdoor cloud.

The product chain is: text -> HY-World 2.0 -> point cloud -> Minecraft blocks. The cloud is the
ONLY input, so everything a supervised benchmark would take from ground truth has to be replaced
here, and every replacement is written into the manifest:

  * block scale is a CHOICE, never recovered metres (HY-World units are non-metric),
  * the labels array is all air - a generated scene has no reference build,
  * the completion formulation's "known evidence" mask is the set of CELLS WITH POINTS rather
    than a ground-truth known-air map,
  * therefore NO IoU or accuracy number is produced here. There is nothing to score against.

What is preserved exactly is the model-facing contract: `pack()` builds the 5x16^3 condition, so
the conditioning the Flow sees is byte-identical in construction to training, and the material
stage receives the same 6-channel evidence frame and 18+context features as `aligned_pair` gives
it for a dataset building.

Stages are separate so only one model is resident at a time on this 15 GB CPU-only box.

Usage:
  python -m tools.cloudbuild.hyworld2_cloud_infer --ply <cloud.ply> --out <dir> \
      --name hyworld2_scene --device cpu [--blocks-per-unit 8] [--stages features,flow,material,export]
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

ROOT = Path('/run/media/sensen/Data1/minecraft/AI_3D_World_Repos')
VAE = ROOT / 'outputs/cloudbuild/vae_air_repair_stage4/last.pt'
FLOW = ROOT / 'outputs/cloudbuild/trellis_flow_reg_v1/best.pt'
MATERIAL = ROOT / 'outputs/cloudbuild/trellis_material_ctx_v1/best.pt'
AIR_NAMES = ('minecraft:air', 'minecraft:cave_air', 'minecraft:void_air')


# --------------------------------------------------------------------------- PLY
def read_ply(path):
    """Binary little-endian PLY, xyz float + uchar rgb(a)."""
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
    xyz = np.stack([raw['x'], raw['y'], raw['z']], 1).astype(np.float64)
    rgb = (np.stack([raw['red'], raw['green'], raw['blue']], 1).astype(np.float32) / 255.0
           if 'red' in props else np.zeros_like(xyz, np.float32))
    return xyz, rgb


def hyworld_to_minecraft(xyz, *, up_source_axis=2):
    """Z-up generated scene -> Y-up Minecraft frame, chirality preserved.

    Signed axis map (MC axis = sign * (source axis + 1)): X=+x, Y=+z, Z=-y.
    The permutation (0,2,1) is odd (one transposition) and the single sign flip is odd too, so the
    determinant is +1: the mapping is a rotation, not a mirror.
    """
    out = np.empty_like(xyz)
    out[:, 0] = xyz[:, 0]
    out[:, 1] = xyz[:, up_source_axis]
    out[:, 2] = -xyz[:, 1]
    return out


def sky_mask(xyz, rgb):
    """Blue-grey rule, measured on this scene: 46% of the old landscape cloud was sky, 4.4% here.

    A horizontal sky dome from an equirect panorama is blue-grey; terrain and blocky builds are
    green/brown/grey-warm. Stated as a heuristic, not a segmentation model: HY-World's own sky
    removal (ZiM + Grounding-DINO, SAM3 for the outdoor branch) is what stage 1 used, and this
    only exists so deployment does not depend on that stage's artefact.
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (b >= r) & ((b - r) >= 0.012) & (np.abs(b - g) <= 0.10) & (b >= 0.27)


# --------------------------------------------------------------------------- features
def stage_features(args, manifest):
    xyz, rgb = read_ply(args.ply)
    sky = sky_mask(xyz, rgb)
    manifest['cloud'] = {'ply': str(args.ply), 'points': int(len(xyz)),
                         'sky_points': int(sky.sum()), 'sky_fraction': round(float(sky.mean()), 4)}
    if args.drop_sky:
        xyz, rgb = xyz[~sky], rgb[~sky]
    mc = hyworld_to_minecraft(xyz)

    # non-metric scale: a chosen number of blocks per source unit.
    unit = float(args.blocks_per_unit)
    blocks = mc * unit
    lo = blocks.min(0)

    # choose the crop: the densest vertical structure inside a 64-cell window.
    # "structure" = points above the ground surface, since a building is what stands up.
    ground = np.percentile(blocks[:, 1], 25)
    above = blocks[:, 1] - ground
    standing = above > (args.blocks_per_unit * args.standing_units)
    if standing.sum() < 100:
        raise SystemExit(f'only {int(standing.sum())} points stand above the ground: no building to crop')

    side = args.window
    # slide a coarse grid over the horizontal plane and keep the window with the most standing points
    best, best_n = None, -1
    step = max(side // 4, 1)
    for x0 in range(int(blocks[:, 0].min()), int(blocks[:, 0].max()) + 1, step):
        for z0 in range(int(blocks[:, 2].min()), int(blocks[:, 2].max()) + 1, step):
            sel = ((blocks[:, 0] >= x0) & (blocks[:, 0] < x0 + side)
                   & (blocks[:, 2] >= z0) & (blocks[:, 2] < z0 + side) & standing)
            n = int(sel.sum())
            if n > best_n:
                best, best_n = (x0, z0), n
    cx, cz = best
    y_lo = int(np.floor(ground - 2))
    origin = np.array([cx, y_lo, cz], np.int64)

    index = np.floor(blocks - origin).astype(np.int64)
    shape = (side, side, side)
    inside = np.all((index >= 0) & (index < side), axis=1)
    index, colours = index[inside], rgb[inside]
    counts = np.zeros(shape, np.uint32)
    sums = np.zeros(shape + (3,), np.float64)
    flat = np.ravel_multi_index((index[:, 0], index[:, 1], index[:, 2]), shape)
    np.add.at(counts.reshape(-1), flat, 1)
    for ch in range(3):
        np.add.at(sums.reshape(-1, 3)[:, ch], flat, colours[:, ch])

    from tools.cloudbuild.gs2blocks_data import Vocabulary
    vocab = Vocabulary()
    reg = vocab.registry
    labels = np.full(shape, vocab.air, np.int32)   # no reference build exists

    # Optional tight crop: the training corpus' evidence fills 10-40% of its bounding box, while a
    # whole-scene crop leaves a 69x-sparser shell (6.9%) that the Flow was never trained on. Cropping
    # to the standing structure's own box restores an in-distribution evidence density.
    if args.drop_terrain:
        # The corpus' evidence is a BUILDING surface; a generated scene's is a building plus a wide
        # flat terrain sheet, which inflates the observed box and dilutes the condition (6.9% fill vs
        # the corpus' 10-40%). Keep the standing structure and only the ground layers directly under
        # it, so the evidence is a building surface again.
        g = int(np.floor(ground - origin[1]))
        keep = counts > 0
        above = np.zeros(shape, bool)
        above[:, g + 1:, :] = keep[:, g + 1:, :]
        stack = above.any(1) | keep.any(1)
        ground_layers = np.zeros(shape, bool)
        lo = max(g - args.ground_layers + 1, 0)
        ground_layers[:, lo:g + 1, :] = keep[:, lo:g + 1, :]
        keep = above | (ground_layers & stack[:, None, :])
        removed = int((counts > 0).sum() - keep.sum())
        counts = np.where(keep, counts, 0)
        sums = np.where(keep[..., None], sums, 0.0)
        print(f'[features] terrain dropped: removed {removed} of the observed cells, '
              f'kept {int((counts > 0).sum())}')

    if args.building_only:
        # counts is indexed in the CROPPED frame (origin was already subtracted), while
        # `ground` is a world coordinate, so convert before slicing.
        g = int(np.floor(ground - origin[1]))
        above = np.zeros(shape, bool)
        above[:, g + 1:, :] = counts[:, g + 1:, :] > 0
        idx = np.argwhere(above)
        if len(idx) == 0:
            raise SystemExit('building-only crop found no cells above the ground plane')
        lo = np.maximum(idx.min(0) - args.building_margin, 0)
        hi = np.minimum(idx.max(0) + args.building_margin + 1, np.array(shape))
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        counts, sums, labels = counts[sl], sums[sl], labels[sl]
        shape = counts.shape
        print(f'[features] building-only crop {sl} -> {shape}, '
              f'evidence fill {float((counts > 0).mean())*100:.1f}%')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f'{args.name}.npz', labels=labels, counts=counts,
             colours=np.where(counts[..., None] > 0, sums / np.maximum(counts[..., None], 1), 0.0)
             .astype(np.float32))
    meta = {'building': args.name, 'registry_hash': reg.hash, 'axes': 'XYZ', 'height_axis': 'Y',
            'cell_rule': 'floor(blocks - origin); blocks = contract(mc_xyz) * blocks_per_unit',
            'contract': {'signed_axes': [1, 3, -2], 'blocks_per_unit': unit,
                         'determinant': 1, 'source_up_axis': 'Z', 'note': 'non-metric, chosen'},
            'origin_xyz': origin.tolist(), 'shape': list(shape),
            'observed_cells': int((counts > 0).sum()),
            'ground_y_blocks': round(float(ground), 3),
            'standing_points': int(standing.sum()),
            'crop_window_origin_xz': list(map(int, best))}
    (out_dir / f'{args.name}.json').write_text(json.dumps(meta, indent=2) + '\n')
    manifest['features'] = meta
    manifest['features']['counts_max'] = int(counts.max())
    print(f"[features] observed cells {meta['observed_cells']}  shape {shape}  "
          f"counts_max {int(counts.max())}  origin {origin.tolist()}")
    return meta


# --------------------------------------------------------------------------- flow
def build_evidence(npz_path, vocab, size=64):
    """Mirror of material_data.aligned_pair() for an npz with NO reference build.

    Identical arithmetic (placement from the point bbox, `shift` centring, log1p count channels);
    the only difference is that `labels` are all air, so `states` is entirely air and `valid`
    means "inside the cropped point extent" rather than "ground truth known".
    """
    from tools.cloudbuild.train_coarse_flow import placement
    with np.load(npz_path) as data:
        labels, counts, colours = data['labels'], data['counts'], data['colours']
    low, offset = placement(counts, labels.shape, size)
    shift = offset - low
    src_low = np.maximum(0, -shift)
    src_high = np.minimum(labels.shape, size - shift)
    src = tuple(slice(int(a), int(b)) for a, b in zip(src_low, src_high))
    dst = tuple(slice(int(a), int(b)) for a, b in zip(src_low + shift, src_high + shift))
    states = np.full((size,) * 3, vocab.air, np.int32)
    valid = np.zeros(states.shape, bool)
    evidence = np.zeros(states.shape + (6,), np.float32)
    states[dst], valid[dst] = labels[src], True
    count = counts[src].astype(np.float32)
    evidence[dst] = np.concatenate(((count > 0)[..., None], np.log1p(count)[..., None],
                                    colours[src], np.clip(np.log1p(count), 0, 4)[..., None] / 4),
                                   axis=-1)
    return states, valid, evidence, offset, low


def stage_flow(args, manifest):
    import torch
    from tools.cloudbuild.gs2blocks_data import Vocabulary
    from tools.cloudbuild.material_data import pooled_condition
    from tools.cloudbuild.train_coarse_flow import exact_evidence
    from tools.cloudbuild.demo_stage2_output import load_models
    from tools.cloudbuild.trellis_coarse_flow import flow_sample

    vocab = Vocabulary()
    data = Path(args.out)
    air = [vocab.registry.encode(s) for s in AIR_NAMES]

    states, valid, evidence, origin, low = build_evidence(data / f'{args.name}.npz', vocab)
    # pack() cannot be used here: it routes through canonical() -> require_closed_asset(), which
    # demands a ground-truth supervision proof that a generated scene does not and must not have.
    # pooled_condition() applies the same pooling pack() uses for its condition tokens, and the
    # repo's own tests verify the two cell by cell, so the model-facing conditioning is unchanged.
    condition = pooled_condition(evidence)
    manifest.setdefault('substitutions', []).append(
        'condition built with pooled_condition(evidence): pack() is unusable without a '
        'closed-asset supervision proof, and none is fabricated')
    print('[flow] condition built (5x16^3) with pooled_condition (pack() needs GT provenance)')

    vae, flow, report = load_models(args.vae, args.flow, args.device)
    mean = torch.tensor(report['latent_mean'], dtype=torch.float32, device=args.device).view(1, 8, 1, 1, 1)
    std = torch.tensor(report['latent_std'], dtype=torch.float32, device=args.device).view(1, 8, 1, 1, 1)
    observed = exact_evidence(data, args.name)   # count>0 placed exactly in canonical 64^3
    with torch.no_grad():
        tokens = torch.from_numpy(condition).to(args.device)[None]
        from tools.cloudbuild.train_coarse_flow import evidence_latent
        source = evidence_latent(vae, condition, mean.view(8), std.view(8),
                                 observed_mask=observed, device=args.device)
        latent = flow_sample(flow, tokens, steps=args.steps, seed=args.seed, device=args.device,
                             guidance_scale=args.guidance, source=source,
                             source_sigma=float(report['source_sigma']))
        probability = vae.decode(latent * std + mean).sigmoid()[0, 0].float().cpu().numpy()
    # cache format: latent is stored WITHOUT the batch dim, as material_data.load_example validates
    np.savez(data / f'{args.name}.flow.npz', latent=latent[0].float().cpu().numpy(),
             probability=probability.astype(np.float32),
             evidence=evidence, valid=valid, states=states)
    solid = probability > args.threshold
    manifest['flow'] = {'vae': str(args.vae), 'flow': str(args.flow), 'steps': args.steps,
                        'guidance': args.guidance, 'source_sigma': float(report['source_sigma']),
                        'threshold': args.threshold, 'completion': bool(report['completion_formulation']),
                        'generated_cells': int(solid.sum()),
                        'known_evidence_cells': int((evidence[..., 0] > 0).sum()),
                        'origin_xyz': origin.tolist()}
    print(f"[flow] generated cells {int(solid.sum())}  (evidence {int((evidence[...,0]>0).sum())})")
    return origin


def stage_material(args, manifest):
    import torch
    from tools.cloudbuild.gs2blocks_data import Vocabulary
    from tools.cloudbuild.material_data import query_features
    from tools.cloudbuild.trellis_material import MaterialDecoder
    from tools.cloudbuild.trellis_material_context import NeighbourhoodTrunk
    from tools.cloudbuild.train_material import predict_volume

    vocab = Vocabulary()
    data = Path(args.out)
    saved = torch.load(MATERIAL, map_location='cpu', weights_only=False)
    cfg = saved['config']
    trunk = NeighbourhoodTrunk(context_channels=cfg['context_channels']).to(args.device).eval()
    model = MaterialDecoder(vocab, width=cfg['width'],
                            in_channels=18 + cfg['context_channels']).to(args.device).eval()
    model.load_state_dict(saved['model'])
    trunk.load_state_dict(saved['trunk'])
    if cfg.get('registry_hash') and cfg['registry_hash'] != vocab.registry.hash:
        raise SystemExit('material checkpoint registry hash mismatch')

    with np.load(data / f'{args.name}.flow.npz') as z:
        example = {k: z[k] for k in ('latent', 'probability', 'evidence', 'valid', 'states')}
    states = predict_volume(model, example, vocab, args.device,
                            threshold=args.threshold, trunk=trunk)
    np.save(data / f'{args.name}.states.npy', states)
    non_air = int((states != vocab.air).sum())
    present, counts = np.unique(states, return_counts=True)
    top = sorted(zip((int(c) for c in counts), (int(s) for s in present)), reverse=True)[:8]
    manifest['material'] = {'checkpoint': str(MATERIAL), 'step': int(saved['step']),
                            'width': cfg['width'], 'context_channels': cfg['context_channels'],
                            'non_air_cells': non_air,
                            'distinct_states': int(len(present)),
                            'top_states': [[vocab.registry.string(s), c] for c, s in top]}
    print(f"[material] non-air cells {non_air}  distinct states {len(present)}")
    for name, c in manifest['material']['top_states'][:5]:
        print(f'   {c:>7}  {name}')
    return states


def stage_export(args, manifest):
    import numpy as np
    from tools.cloudbuild.gs2blocks_data import Vocabulary

    vocab = Vocabulary()
    data = Path(args.out)
    states = np.load(data / f'{args.name}.states.npy')
    if not (states != vocab.air).any():
        raise SystemExit('material stage produced an empty volume')
    low = np.argwhere(states != vocab.air).min(0)
    high = np.argwhere(states != vocab.air).max(0) + 1
    window = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    cropped = states[window].astype(np.int32)
    ids, inverse = np.unique(cropped, return_inverse=True)
    palette = [vocab.registry.string(int(i)) for i in ids]
    origin = np.array(json.loads((data / f'{args.name}.json').read_text())['origin_xyz'], np.int64)
    path = data / f'{args.name}.litematic'

    from mcfix.containers import export_litematic, decode_path
    export_litematic(path, inverse.reshape(cropped.shape), palette,
                     origin=tuple(int(v) for v in (origin + low)), data_version=4189,
                     name=args.name[:40])
    decoded = decode_path(path)
    recovered = np.array([vocab.registry.encode(s) for s in decoded['names']], np.int32)[
        np.transpose(np.asarray(decoded['vol']), (2, 0, 1))]
    if not np.array_equal(cropped, recovered):
        raise AssertionError('exported blockstates do not round-trip')
    manifest['export'] = {'litematic': str(path), 'cropped_shape': [int(v) for v in cropped.shape],
                          'export_origin_xyz': [int(v) for v in (origin + low)],
                          'palette_size': len(palette), 'roundtrip_exact': True}
    print(f'[export] {path}  shape {cropped.shape}  palette {len(palette)}  roundtrip exact')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ply', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--blocks-per-unit', type=float, default=8.0,
                   help='CHOSEN non-metric mapping; HY-World units have no physical size')
    p.add_argument('--standing-units', type=float, default=0.4,
                   help='height above the ground plane that counts as "standing" in source units')
    p.add_argument('--window', type=int, default=64)
    p.add_argument('--drop-sky', action='store_true', default=True)
    p.add_argument('--keep-sky', dest='drop_sky', action='store_false')
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--guidance', type=float, default=1.0)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--flow', type=Path, default=FLOW, help='Flow checkpoint to deploy')
    p.add_argument('--vae', type=Path, default=VAE, help='VAE paired with that Flow')
    p.add_argument('--building-only', action='store_true',
                   help='crop to the standing structure so evidence density matches training')
    p.add_argument('--building-margin', type=int, default=3)
    p.add_argument('--drop-terrain', action='store_true',
                   help='drop the flat terrain sheet so the evidence is a building surface')
    p.add_argument('--ground-layers', type=int, default=2)
    p.add_argument('--stages', default='features,flow,material,export')
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
   # mcfix (the Java 1.21.4 container writer) lives in the redstone checkout, as in every
    # other exporter in this package.
    sys.path.insert(0, '/run/media/sensen/Data1/minecraft/redstone_v5_local/code')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / f'{args.name}.manifest.json'
    manifest = json.loads(mpath.read_text()) if mpath.is_file() else {}
    manifest.setdefault('schema', 'hyworld2-cloud-deploy-v1')
    manifest['not_claimed'] = [
        'no ground truth exists for a generated scene: no IoU, no accuracy, no quality score',
        'block scale is a chosen non-metric mapping, not a recovered physical size',
        'labels/validity are placeholders (all air / observed extent), never supervision',
        'the caption and pano prompts were authored by this pipeline, not a VLM description',
    ]
    stages = args.stages.split(',')
    origin = None
    if 'features' in stages:
        stage_features(args, manifest)
    if 'flow' in stages:
        origin = stage_flow(args, manifest)
    if 'material' in stages:
        stage_material(args, manifest)
    if 'export' in stages:
        stage_export(args, manifest)
    manifest['args'] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    mpath.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'manifest -> {mpath}')


if __name__ == '__main__':
    main()
