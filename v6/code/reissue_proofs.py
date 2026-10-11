"""Re-issue closed-asset proofs for a rebuilt evidence corpus, in parallel.

`supervision_contract.main()` cannot be called as a module here because it imports mcfix lazily and
that package lives in the redstone checkout, so this launcher fixes sys.path and uses a process pool.
The audit itself is the repo's own: it decodes each source schematic, verifies the pair's `labels`
still equal the source crop with zero solids outside the crop, and hashes the NEW pair. It never
looks at counts/colours, which is exactly why restricting the evidence is allowed while the target
stays untouched.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path('/run/media/sensen/Data1/minecraft/AI_3D_World_Repos')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/run/media/sensen/Data1/minecraft/redstone_v5_local/code')


def one(arg):
    data, building = arg
    from tools.cloudbuild.supervision_contract import audit_source
    try:
        proof = audit_source(Path(data), building)
        path = Path(data) / f'{building}.supervision.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(proof, indent=2) + '\n')
        tmp.replace(path)
        return (building, True, None)
    except Exception as exc:
        return (building, False, f'{type(exc).__name__}: {exc}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    buildings = sorted(p.stem for p in args.data.glob('*.npz'))
    have = {p.stem for p in args.data.glob('*.supervision.json')}
    todo = [b for b in buildings if b not in have]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(buildings)} pairs, {len(todo)} need a fresh proof, {args.workers} workers', flush=True)

    ok = bad = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (building, good, err) in enumerate(pool.map(one, [(str(args.data), b) for b in todo], chunksize=8), 1):
            if good:
                ok += 1
            else:
                bad += 1
                if len(failures) < 10:
                    failures.append({'building': building, 'error': err})
            if i % 1000 == 0:
                print(f'  {i}/{len(todo)}  ok {ok}  failed {bad}', flush=True)
    print(json.dumps({'proofs_written': ok, 'failed': bad, 'failures': failures}, indent=2))


if __name__ == '__main__':
    main()
