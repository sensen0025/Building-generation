"""Immutable source snapshots, canonical crops and source-group splits."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time
import numpy as np
from .codec import registry, RENAMES, parse_state, rotate_volume
from .containers import decode_path, non_air


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def class_contract(vocab_path, block_map_path):
    vocab=Path(vocab_path).read_text().splitlines()+['minecraft:other']
    bm=np.load(block_map_path)
    r=registry(); rows=[]
    for cid,old in enumerate(bm):
        original=vocab[int(old)]
        name=RENAMES.get(original,original)
        proxy=name not in r.blocks
        if proxy: name='minecraft:stone'
        b=r.blocks[name]
        rows.append({'id':cid,'original_name':original,'name':name,'proxy':proxy,
                     'min_state':b['minStateId'],'count':b['maxStateId']-b['minStateId']+1,
                     'default_local':b['defaultState']-b['minStateId']})
    value={'schema':'mcfix-class-contract-v1','registry_hash':r.hash,'classes':rows}
    value['hash']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return value


def tile_starts(n,size=64,stride=48):
    if n<=size:return [0]
    return sorted(set(list(range(0,n-size+1,stride))+[n-size]))


def semantic_hash(xyz,palette):
    # Split identity ignores state/material spelling variants and unused palette
    # entries, preventing alternate states/rotations of one layout from leaking.
    used,inv=np.unique(xyz,return_inverse=True)
    names=[RENAMES.get(parse_state(palette[int(i)])[0],parse_state(palette[int(i)])[0]) for i in used]
    order=sorted(set(names));lookup={n:i for i,n in enumerate(order)}
    a=np.array([lookup[n] for n in names],np.uint16 if len(order)<=65536 else np.uint32)[inv].reshape(xyz.shape)
    header=json.dumps(order,ensure_ascii=False,separators=(',',':')).encode()
    hashes=[]
    for k in range(4):
        rotated=rotate_volume(a,k)
        h=hashlib.sha256(header);h.update(str(rotated.shape).encode());h.update(rotated.tobytes());hashes.append(h.hexdigest())
    return min(hashes)


def build_dataset(out, sources, contract, historic_paths=(), max_tiles=256, limit=None, finalize=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'sources').mkdir(exist_ok=True);(out/'samples').mkdir(exist_ok=True)
    r=registry();atomic_json(out/'classes.json',contract)
    historic={digest(p) for p in historic_paths if Path(p).is_file()}
    byname={}
    for c in contract['classes']:
        if not c['proxy']:byname.setdefault(c['name'],c['id'])
    records=[];samples=[];seen={};schema=r.hash
    old_records={x['sha256']:x for x in json.loads((out/'sources.json').read_text())} if (out/'sources.json').exists() else {}
    paths=sorted(set(map(str,sources)))
    if limit: paths=paths[:limit]
    t=time.time()
    for n,path in enumerate(paths):
        sha=digest(path)
        if sha in seen:
            if path not in seen[sha]['aliases'] and path != seen[sha]['source']:seen[sha]['aliases'].append(path)
            continue
        suffix=Path(path).suffix.lower();snapshot=out/'sources'/(sha+suffix)
        if not snapshot.exists():
            tmp=snapshot.with_suffix(suffix+'.tmp');shutil.copyfile(path,tmp)
            if digest(tmp)!=sha:raise RuntimeError('Source changed during snapshot')
            tmp.replace(snapshot)
        rec={'source':path,'snapshot':str(snapshot),'sha256':sha,'aliases':[],
             'historic_seen':sha in historic,'registry_hash':schema}
        seen[sha]=rec;records.append(rec)
        previous=old_records.get(sha)
        if previous and previous.get('status')=='ok' and previous.get('registry_hash')==schema:
            prior=previous.get('samples',[])
            if all((out/'samples'/(x['id']+'.npz')).exists() for x in prior):
                rec.update(previous);rec['historic_seen']=sha in historic;samples.extend(prior);continue
        try:
            d=decode_path(snapshot)
            xyz=d['vol'].transpose(2,0,1)
            active=non_air(xyz,d['names'])
            if not active.any(): raise ValueError('empty structure')
            axes=[np.flatnonzero(active.any(axis=tuple(j for j in range(3) if j!=i))) for i in range(3)]
            low=np.array([v[0] for v in axes]);high=np.array([v[-1]+1 for v in axes])
            xyz=np.ascontiguousarray(xyz[tuple(slice(a,b) for a,b in zip(low,high))])
            # Drop unused names only, never used states.
            ids,inv=np.unique(xyz,return_inverse=True);palette=[d['names'][int(i)] for i in ids]
            xyz=inv.reshape(xyz.shape).astype(np.uint32)
            group=semantic_hash(xyz,palette)
            spans=[tile_starts(i) for i in xyz.shape]
            ntiles=int(np.prod([len(s) for s in spans]))
            if ntiles>max_tiles:raise ValueError(f'resource_limited: {ntiles} tiles exceeds {max_tiles}; original preserved')
            sid_lut=np.full(len(palette),-1,np.int32);cls_lut=np.full(len(palette),127,np.uint8)
            issues=[]
            for i,state in enumerate(palette):
                try:
                    sid,info=r.normalize_source(state,data_version=d['meta']['data_version']);sid_lut[i]=sid
                    name=r.names[sid]
                    cls_lut[i]=0 if name in ('minecraft:air','minecraft:cave_air','minecraft:void_air') else byname.get(name,127)
                    if info['renamed'] or info['defaulted_properties']:issues.append({'palette_index':i,**info})
                except ValueError as e:
                    issues.append({'palette_index':i,'unsupported':str(e)})
            rec.update(status='ok',group=group,meta=d['meta'],dims_xyz=list(xyz.shape),source_bbox_min=low.tolist(),
                       palette_issues=issues,samples=[])
            for x in spans[0]:
                for y in spans[1]:
                    for z in spans[2]:
                        raw=xyz[x:x+64,y:y+64,z:z+64]
                        solid=non_air(raw,palette)
                        if not solid.any():continue
                        pad=(64-np.array(raw.shape))//2
                        dest=tuple(slice(int(a),int(a+b)) for a,b in zip(pad,raw.shape))
                        # Raw palette array carries every original state and exceptional air.
                        pal=['minecraft:air']+palette
                        raw64=np.zeros((64,64,64),np.uint32);raw64[dest]=raw+1
                        states=np.full((64,64,64),r.defaults['minecraft:air'],np.int32);states[dest]=sid_lut[raw]
                        blocks=np.zeros((64,64,64),np.uint8);blocks[dest]=cls_lut[raw]
                        valid=(states>=0)&(blocks!=127)&(blocks!=63)
                        defaulted=np.array([bool(any(v.get('palette_index')==i and v.get('defaulted_properties') for v in issues)) for i in range(len(palette))])
                        observed=np.ones((64,64,64),bool);observed[dest]=~defaulted[raw]
                        sample_id=sha[:24]+f'_{x}_{y}_{z}'
                        fp=out/'samples'/(sample_id+'.npz');tmp=fp.with_suffix('.tmp')
                        with open(tmp,'wb') as f:
                            np.savez_compressed(f,blocks=blocks,states=states,valid=valid,observed=observed,
                                                raw=raw64,palette=np.array(pal),registry_hash=np.array(schema))
                        tmp.replace(fp)
                        item={'id':sample_id,'source_sha256':sha,'group':group,'path':str(fp),
                              'crop_origin_xyz':(low+np.array([x,y,z])+np.array(d['meta']['origin_xyz'])).tolist(),
                              'pad_xyz':pad.tolist(),'solid':int(solid.sum()),
                              'supported_fraction':float(valid[dest][solid].mean()),
                              'sample_sha256':digest(fp)}
                        samples.append(item);rec['samples'].append(item)
        except Exception as e:
            rec.update(status='quarantined',reason=f'{type(e).__name__}: {e}')
        if n%100==0:
            atomic_json(out/'sources.json',records)
            print(json.dumps({'source':n,'total':len(paths),'samples':len(samples),'seconds':round(time.time()-t)}),flush=True)
    atomic_json(out/'sources.json',records)
    if not finalize:return {'partial':True,'sources':len(records),'samples':len(samples)}
    return finalize_dataset(out,records,contract,max_tiles)


def finalize_dataset(out,records,contract,max_tiles=256):
    out=Path(out);r=registry();schema=r.hash
    samples=[i for rec in records if rec.get('status')=='ok' for i in rec['samples']]
    # Exact duplicate crops across different source files are unioned too.
    parent={r['group']:r['group'] for r in records if r.get('group')}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    crop_seen={};canonical_palettes={}
    versions={rec['sha256']:rec.get('meta',{}).get('data_version',0) for rec in records}
    from concurrent.futures import ThreadPoolExecutor
    stateful=np.array([c['count']>1 and not c['proxy'] for c in contract['classes']])
    def inspect(item):
        with np.load(item['path']) as a:
            key=item['source_sha256']
            if key not in canonical_palettes:
                palette=[]
                for state in a['palette'].tolist():
                    try:sid,_=r.normalize_source(state,data_version=versions[key]);palette.append(r.names[sid])
                    except ValueError:palette.append(parse_state(state)[0])
                canonical_palettes[key]=palette
            crop=semantic_hash(a['raw'],canonical_palettes[key])
            observed=int((a['valid'] & a['observed'] & stateful[a['blocks']]).sum())
        return crop,observed
    started=time.time()
    # NumPy, decompression and file I/O release the GIL. Bound queued work to
    # 1000 rows and three concurrent grids instead of loading the corpus.
    with ThreadPoolExecutor(max_workers=3) as executor:
        for begin in range(0,len(samples),1000):
            chunk=samples[begin:begin+1000]
            for item,(crop,observed) in zip(chunk,executor.map(inspect,chunk)):
                item['observed_state_cells']=observed
                if crop in crop_seen:
                    a,b=find(item['group']),find(crop_seen[crop]);parent[max(a,b)]=min(a,b)
                else:crop_seen[crop]=item['group']
            progress={'phase':'group_validation','checked':begin+len(chunk),'total':len(samples),'seconds':round(time.time()-started)}
            atomic_json(out/'group_progress.json',progress);print(json.dumps(progress),flush=True)
    group_historic={}
    for rec in records:
        if 'group' in rec:
            rec['group']=find(rec['group']);group_historic[rec['group']]=group_historic.get(rec['group'],False) or rec['historic_seen']
    # Only truly unseen source groups can be validation/test for pretrained flow.
    eligible={find(i['group']) for i in samples if i['supported_fraction']>=.85 and i['observed_state_cells']>=8}
    fresh=sorted(g for g in parent if find(g)==g and not group_historic.get(g,False) and g in eligible)
    if len(fresh)<2:raise RuntimeError('Need at least two unseen eligible source groups for validation/test')
    nval=max(1,len(fresh)//10);ntest=max(1,len(fresh)//10)
    split={g:('validation' if i<nval else 'test' if i<nval+ntest else 'train') for i,g in enumerate(fresh)}
    for item in samples:
        item['group']=find(item['group']);item['split']=split.get(item['group'],'train')
        item['historic_seen']=group_historic.get(item['group'],False)
    manifest={'schema':'mcfix-dataset-v1','registry_hash':schema,'class_contract_hash':contract['hash'],
              'sources':str(out/'sources.json'),'samples':samples,
              'split_counts':{s:sum(i['split']==s for i in samples) for s in ('train','validation','test')},
              'source_counts':{s:sum(r['status']==s for r in records) for s in ('ok','quarantined')},
              'max_tiles':max_tiles,'split_contract':'source-and-rotation-geometry-groups-v2'}
    atomic_json(out/'sources.json',records);atomic_json(out/'manifest.json',manifest)
    return manifest
