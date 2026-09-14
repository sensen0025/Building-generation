"""Bounded resumable source workers, followed by one authoritative split pass."""
import argparse,json,os,time,resource
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .data import class_contract,digest,atomic_json,build_dataset,finalize_dataset


def worker(args):
    out,paths,contract,historic=args
    # Decoding requires no GPU libraries; cap each worker's virtual address space.
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    start=time.time();result=build_dataset(out,paths,contract,historic,finalize=False)
    result.update(seconds=round(time.time()-start,2),max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(Path(out)/'complete.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--jobs',type=int,default=3);p.add_argument('--limit',type=int);a=p.parse_args()
    if not 1<=a.jobs<=3:raise ValueError('Use 1..3 memory-bounded workers')
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    old=Path('/home/linux/redstone_ai/dataset');new=Path('/data/redstone_v4')
    contract=class_contract(old/'BLOCK_VOCAB_FULL.txt',old/'v2/block_map.npy');atomic_json(out/'classes.json',contract)
    for name in ['sources','samples']:(out/name).mkdir(exist_ok=True)
    paths=sorted({str(p) for d in [old/'litematics',new/'sources'] for p in d.iterdir() if p.suffix.lower() in ('.litematic','.schem','.schematic','.nbt','.mcstructure')})
    if a.limit:paths=paths[:a.limit]
    historic=[str(old/'litematics'/s) for s in (old/'valid_files_list.txt').read_text().splitlines()]
    historic += [str(new/'sources'/r['file']) for r in json.loads((new/'work/ingest_manifest_ok.json').read_text())]
    buckets=[[] for _ in range(a.jobs)]
    for path in paths:buckets[int(digest(path)[:8],16)%a.jobs].append(path)
    cached=json.loads((out/'sources.json').read_text()) if (out/'sources.json').exists() else []
    arguments=[]
    for i,paths in enumerate(buckets):
        shard=out/'shards'/str(i);shard.mkdir(parents=True,exist_ok=True)
        for name in ['sources','samples']:
            link=shard/name
            if not link.exists():link.symlink_to(out/name,target_is_directory=True)
        previous=json.loads((shard/'sources.json').read_text()) if (shard/'sources.json').exists() else []
        rows={r['sha256']:r for r in cached if int(r['sha256'][:8],16)%a.jobs==i}
        rows.update({r['sha256']:r for r in previous});atomic_json(shard/'sources.json',list(rows.values()))
        arguments.append((str(shard),paths,contract,historic))
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        results=list(ex.map(worker,arguments))
    records=[]
    for i in range(a.jobs):records+=json.loads((out/'shards'/str(i)/'sources.json').read_text())
    atomic_json(out/'worker_report.json',results);atomic_json(out/'sources.json',records)
    print(json.dumps({'workers_finished':results,'phase':'split_validation'}),flush=True)
    m=finalize_dataset(out,records,contract);m['smoke_only']=False;atomic_json(out/'manifest.json',m)
    print(json.dumps({k:v for k,v in m.items() if k!='samples'}),flush=True)
if __name__=='__main__':main()
