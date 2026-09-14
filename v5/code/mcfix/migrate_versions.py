"""Targeted, recorded correction of ambiguous pre/post-flattening block names."""
import argparse,json
from pathlib import Path
import numpy as np
from .codec import registry,parse_state
from .data import atomic_json,digest,finalize_dataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args();root=Path(a.dataset)
    manifest=json.loads((root/'manifest.json').read_text());records=json.loads((root/'sources.json').read_text());contract=json.loads((root/'classes.json').read_text());r=registry()
    names={}
    for c in contract['classes']:
        if not c['proxy']:names.setdefault(c['name'],c['id'])
    changed=0;sources=0
    for rec in records:
        if rec.get('status')!='ok' or not rec['samples']:continue
        with np.load(rec['samples'][0]['path']) as sample:pal=sample['palette'].tolist()
        if not any(parse_state(s)[0]=='minecraft:grass' for s in pal):continue
        dv=rec['meta']['data_version'];sid=[];cls=[];observed=[]
        for state in pal:
            try:
                value,info=r.normalize_source(state,data_version=dv);name=r.names[value];c=0 if name in ('minecraft:air','minecraft:cave_air','minecraft:void_air') else names.get(name,127);obs=not info['defaulted_properties']
            except ValueError:value=-1;c=127;obs=False
            sid.append(value);cls.append(c);observed.append(obs)
        sid=np.array(sid,np.int32);cls=np.array(cls,np.uint8);observed=np.array(observed,bool)
        source_changed=False
        for item in rec['samples']:
            path=Path(item['path'])
            with np.load(path) as sample:data={k:sample[k] for k in sample.files}
            states=sid[data['raw']];blocks=cls[data['raw']]
            if np.array_equal(states,data['states']) and np.array_equal(blocks,data['blocks']):continue
            data.update(states=states,blocks=blocks,valid=(states>=0)&(blocks!=127)&(blocks!=63),observed=observed[data['raw']])
            tmp=path.with_suffix('.migration.tmp')
            with open(tmp,'wb') as f:np.savez_compressed(f,**data)
            tmp.replace(path);item['sample_sha256']=digest(path)
            raw_names=np.array([parse_state(s)[0] for s in pal],object)[data['raw']]
            solid=~np.isin(raw_names,['minecraft:air','minecraft:cave_air','minecraft:void_air'])
            item['supported_fraction']=float(data['valid'][solid].mean()) if solid.any() else 1.
            changed+=1;source_changed=True
        if source_changed:
            rec['version_normalization']='minecraft:grass: pre-flattening grass_block, modern short_grass, unknown version quarantined per cell'
            sources+=1
    atomic_json(root/'sources.json',records)
    # Recompute authoritative source groups/splits after any label changes.
    m=finalize_dataset(root,records,contract);m['smoke_only']=manifest.get('smoke_only',False);m['canonicalization_contract']='version-aware-grass-v2'
    atomic_json(root/'manifest.json',m);atomic_json(root/'version_migration.json',{'sources_changed':sources,'samples_changed':changed})
    print(json.dumps({'sources_changed':sources,'samples_changed':changed}),flush=True)
if __name__=='__main__':main()
