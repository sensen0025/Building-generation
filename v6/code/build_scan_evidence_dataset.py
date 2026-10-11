"""Build the "outside-scan" evidence corpus: input = what an outside scanner sees, target = the real build.

Why this dataset exists. Deployment on a generated HY-World 2.0 cloud fed the Flow a surface shell
(3.3% of its window, 6.9% of its bounding box) while the corpus' own evidence covers the solid
(measured mean evidence/solid = 1.13). The Flow therefore learned "evidence ~ occupancy, slightly
denoised" and every deployment arm generated LESS than the evidence (ratios 0.79-0.92 across two
scales, two thresholds, tight-crop and no-terrain variants). It never learned to turn a shell into a
solid, because the corpus never asked it to.

This builder asks it to. For every building the evidence is restricted to cells a scan from OUTSIDE
could see:
    outside = air reachable from the volume border without crossing a solid (flood fill)
    shell   = solid cells 6-adjacent to `outside`
    visible = shell dilated by `--dilate` (real scans also land just outside the surface)
and every observed cell that is not `visible` - interior leakage - is zeroed. The target `labels`
are UNTOUCHED, so the closed-asset proof still describes the same standalone schematic; only the
input evidence is transformed, and `supervision_contract.audit_source` re-issues the proof bound to
the new pair hash (it checks the target and never inspects counts/colours).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path('/run/media/sensen/Data1/minecraft/AI_3D_World_Repos')
SOURCE = ROOT / 'outputs/cloudbuild/gs2blocks_all'


def visible_mask(labels, *, air_ids, dilate=1):
    solid = ~np.isin(labels, air_ids)
    if not solid.any():
        return None, None
    air = ~solid
    seed = np.zeros_like(air)
    seed[0, :, :] = seed[-1, :, :] = True
    seed[:, 0, :] = seed[:, -1, :] = True
    seed[:, :, 0] = seed[:, :, -1] = True
    seed &= air
    outside = ndimage.binary_propagation(seed, mask=air)
    shell = solid & ndimage.binary_dilation(outside)
    visible = ndimage.binary_dilation(shell, iterations=dilate) if dilate else shell
    return solid, visible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/cloudbuild/gs2blocks_scan')
    ap.add_argument('--count', type=int, default=0, help='0 = the whole corpus')
    ap.add_argument('--dilate', type=int, default=1)
    ap.add_argument('--reissue-proofs', action='store_true',
                    help='run supervision_contract.audit_source for every written building')
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, '/run/media/sensen/Data1/minecraft/redstone_v5_local/code')
    from tools.cloudbuild.gs2blocks_data import Vocabulary
    from tools.cloudbuild.supervision_contract import audit_source

    vocab = Vocabulary()
    air_ids = [vocab.registry.encode(s) for s in
               ('minecraft:air', 'minecraft:cave_air', 'minecraft:void_air')]
    files = sorted(args.source.glob('*.npz'))
    if args.count:
        files = files[:args.count]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    done = dropped_total = evidence_total = 0
    reports = []
    for i, f in enumerate(files, 1):
        building = f.stem
        meta_src = args.source / f'{building}.json'
        if not meta_src.is_file():
            continue
        with np.load(f) as z:
            labels, counts, colours = z['labels'], z['counts'], z['colours']
        solid, visible = visible_mask(labels, air_ids=air_ids, dilate=args.dilate)
        if visible is None:
            continue
        observed = counts > 0
        keep = observed & visible
        dropped = int((observed & ~visible).sum())
        new_counts = np.where(keep, counts, 0).astype(counts.dtype)
        new_colours = np.where(keep[..., None], colours, 0.0).astype(colours.dtype)
        np.savez(out / f'{building}.npz', labels=labels, counts=new_counts, colours=new_colours)
        shutil.copy2(meta_src, out / f'{building}.json')
        if args.reissue_proofs:
            proof = audit_source(out, building)
            (out / f'{building}.supervision.json').write_text(json.dumps(proof, indent=2) + '\n')
        dropped_total += dropped
        evidence_total += int(observed.sum())
        done += 1
        if i % 500 == 0:
            print(f'  {i}/{len(files)}  dropped {dropped_total}/{evidence_total} '
                  f'({dropped_total/max(evidence_total,1)*100:.1f}%)', flush=True)
        if len(reports) < 20:
            reports.append({'building': building, 'observed': int(observed.sum()),
                            'kept': int(keep.sum()), 'dropped': dropped,
                            'solid': int(solid.sum()),
                            'solid_not_visible': int((solid & ~visible).sum())})

    summary = {
        'schema': 'scan-evidence-v1',
        'source': str(args.source), 'out': str(out), 'dilate': args.dilate,
        'buildings': done,
        'observed_cells': evidence_total,
        'dropped_interior_cells': dropped_total,
        'dropped_fraction': round(dropped_total / max(evidence_total, 1), 4),
        'proofs_reissued': bool(args.reissue_proofs),
        'note': ('evidence restricted to cells visible from outside the solid; labels (target) '
                 'unchanged; proofs re-issued against the new pair hash by audit_source'),
        'sample': reports,
    }
    (out / 'scan_evidence_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'sample'}, indent=2))


if __name__ == '__main__':
    main()
