"""Render jobs derived from canonical registry states, with immutable provenance."""
import argparse,json
from pathlib import Path
import numpy as np
from .codec import registry,rotate_volume
from .containers import export_litematic
from .data import atomic_json,digest

def prepare(dataset,out,limit=None):
    dataset=Path(dataset);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((dataset/'manifest.json').read_text());r=registry()
    if manifest['registry_hash']!=r.hash:raise ValueError('Registry mismatch')
    tasks=[];samples=list(enumerate(manifest['samples']))
    if limit:
        budgets={'validation':max(1,limit//8),'test':max(1,limit//8),'train':max(1,limit-2*max(1,limit//8))}
        selected=[]
        for split,budget in budgets.items():
            candidates=[(i,v) for i,v in samples if v['split']==split and v['supported_fraction']>=.85 and v.get('observed_state_cells',8)>=8]
            # Diverse source groups before multiple overlapping crops.
            seen=set(); first=[];rest=[]
            for item in candidates:
                if item[1]['group'] in seen:rest.append(item)
                else:first.append(item);seen.add(item[1]['group'])
            selected.extend((first+rest)[:budget])
        samples=sorted(selected)
    for idx,item in samples:
        with np.load(item['path']) as a: states=a['states'];blocks=a['blocks']
        unsupported=int((states<0).sum())
        # Render target-version states. Every replacement is recorded, raw states
        # remain in the source/sample archive and are never training labels.
        states=np.where(states>=0,states,r.defaults['minecraft:stone'])
        for k in range(4):
            slot=idx*4+k;transformed=r.rotation[k,rotate_volume(states,k)]
            used,inv=np.unique(transformed,return_inverse=True)
            palette=[r.string(i) for i in used]
            path=out/f'{slot:05d}.litematic'
            stamp={'sample_sha256':item['sample_sha256'],'registry_hash':r.hash,'rotation':k,
                   'export_contract':'mcfix-litematic-v1'}
            meta=path.with_suffix('.json')
            if not path.exists() or not meta.exists() or json.loads(meta.read_text())!=stamp:
                export_litematic(path,inv.reshape(transformed.shape),palette,name=item['id']+f'_r{k}')
                atomic_json(meta,stamp)
            tasks.append({'slot':slot,'sample_id':item['id'],'rotation':k,'split':item['split'],
                          'litematic':str(path),'litematic_sha256':digest(path),
                          'unsupported_replacements':unsupported,**stamp})
    atomic_json(out/'tasks.json',tasks);return tasks

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    print(len(prepare(a.dataset,a.out,a.limit)),flush=True)
if __name__=='__main__':main()
