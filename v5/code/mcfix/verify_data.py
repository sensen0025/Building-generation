"""Independent source-coordinate and registry roundtrip verification."""
import argparse,json
from pathlib import Path
import numpy as np
from .data import digest,atomic_json
from .codec import registry,rotate_state,rotate_volume,parse_state,RENAMES
from .containers import decode_path


def verify(dataset,limit=200):
    dataset=Path(dataset);m=json.loads((dataset/'manifest.json').read_text());records=json.loads((dataset/'sources.json').read_text())
    bysha={r['sha256']:r for r in records};r=registry();classes=json.loads((dataset/'classes.json').read_text())
    if m['registry_hash']!=r.hash or m['class_contract_hash']!=classes['hash']:raise ValueError('Manifest contract mismatch')
    groups={s:{i['group'] for i in m['samples'] if i['split']==s} for s in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if groups[a]&groups[b]:raise AssertionError('Source group leakage')
    if any(i['historic_seen'] for i in m['samples'] if i['split']!='train'):raise AssertionError('Historical source leakage')
    samples=m['samples'];rng=np.random.default_rng(42)
    selected=[samples[i] for i in rng.choice(len(samples),min(limit,len(samples)),replace=False)]
    selected.sort(key=lambda i:i['source_sha256']);last=None;source=None;checked_sources=set();cells=0;state_cells=0
    for item in selected:
        sha=item['source_sha256'];rec=bysha[sha]
        if sha!=last:
            if digest(rec['snapshot'])!=sha:raise AssertionError('Source snapshot digest mismatch')
            source=decode_path(rec['snapshot']);last=sha;checked_sources.add(sha)
        if digest(item['path'])!=item['sample_sha256']:raise AssertionError('Sample digest mismatch')
        with np.load(item['path']) as sample:
            raw=sample['raw'];palette=sample['palette'].tolist()
            pad=np.array(item['pad_xyz']);origin=np.array(item['crop_origin_xyz'])-np.array(source['meta']['origin_xyz'])
            xyz=source['vol'].transpose(2,0,1)
            # Crop extent is the intersection of the original bbox and the 64-grid.
            full_shape=np.array(rec['dims_xyz']);offset=origin-np.array(rec['source_bbox_min']);size=np.minimum(64,full_shape-offset)
            dest=tuple(slice(int(p),int(p+s)) for p,s in zip(pad,size));src=tuple(slice(int(p),int(p+s)) for p,s in zip(origin,size))
            actual=np.array(palette,dtype=object)[raw[dest]];expected=np.array(source['names'],dtype=object)[xyz[src]]
            if not np.array_equal(actual,expected):raise AssertionError(f'Source coordinate/state mismatch: {item["id"]}')
            cells+=actual.size
            lut=[]
            for state in palette:
                try:sid,_=r.normalize_source(state,data_version=source['meta']['data_version'])
                except ValueError:sid=-1
                lut.append(sid)
            canonical=np.array(lut)[raw]
            if not np.array_equal(canonical,sample['states']):raise AssertionError('Canonical state mismatch')
            valid=canonical>=0;state_cells+=int(valid.sum())
            for k in range(4):
                # Compare numerical rotation with independently parsed symbolic states.
                unique=np.unique(canonical[valid])
                for sid in unique:
                    expected=r.encode(rotate_state(r.string(sid),k),fill_defaults=False)
                    if expected!=r.rotation[k,sid]:raise AssertionError('State rotation mismatch')
    return {'checked_samples':len(selected),'checked_sources':len(checked_sources),'exact_source_cells':cells,
            'registry_cells':state_cells,'mismatches':0,'group_counts':{s:len(g) for s,g in groups.items()},
            'smoke_only':m.get('smoke_only',False)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--limit',type=int,default=200);p.add_argument('--out',required=True);a=p.parse_args()
    result=verify(a.dataset,a.limit);atomic_json(a.out,result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
