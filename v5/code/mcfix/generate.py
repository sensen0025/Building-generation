import argparse,json
from pathlib import Path
import numpy as np
from .pipeline import CompletionPipeline
from .containers import decode_path
from .codec import parse_state
from .data import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--out',required=True)
    p.add_argument('--active',help='Model selection manifest; defaults to /data/redstone_v5/active.json when no explicit model paths are supplied')
    p.add_argument('--state-checkpoint');p.add_argument('--geometry-config');p.add_argument('--geometry-checkpoint');p.add_argument('--decoder-checkpoint')
    p.add_argument('--known-litematic');p.add_argument('--known-mask');p.add_argument('--steps',type=int,default=50);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    keys=['state_checkpoint','geometry_config','geometry_checkpoint','decoder_checkpoint']
    supplied=[getattr(a,k) is not None for k in keys]
    if a.active and any(supplied):raise ValueError('Use either --active or explicit model paths')
    if any(supplied) and not all(supplied):raise ValueError('Supply all four model paths or use --active')
    if not any(supplied):
        selection=json.loads(Path(a.active or '/data/redstone_v5/active.json').read_text())
        for key in keys:setattr(a,key,selection[key])
    pipe=CompletionPipeline(a.geometry_config,a.geometry_checkpoint,a.state_checkpoint,a.decoder_checkpoint)
    mask=raw=None;kwargs={}
    if a.known_litematic:
        if not a.known_mask:raise ValueError('--known-mask is required: distinguish known air from unknown space')
        d=decode_path(a.known_litematic);xyz=d['vol'].transpose(2,0,1)
        if xyz.shape!=(64,64,64):raise ValueError('Known input must be 64³; explicit positioning required')
        mask=np.load(a.known_mask)
        if mask.shape!=xyz.shape or mask.dtype!=bool:raise ValueError('Invalid known mask')
        names={c['name']:c['id'] for c in reversed(pipe.classes['classes']) if not c['proxy']}
        ids=[];classes=[]
        for state in d['names']:
            try:sid,_=pipe.r.normalize_source(state,data_version=d['meta']['data_version']);name=pipe.r.names[sid]
            except ValueError:sid=-1;name=parse_state(state)[0]
            ids.append(sid);classes.append(0 if name in ('minecraft:air','minecraft:cave_air','minecraft:void_air') else names.get(name,127))
        raw=np.array(d['names'],dtype=object)[xyz]
        kwargs={'known_blocks':np.array(classes)[xyz],'known_states':np.array(ids)[xyz],'known_mask':mask}
    elif a.known_mask:raise ValueError('--known-mask requires --known-litematic')
    b,s=pipe(a.image,steps=a.steps,seed=a.seed,**kwargs);pipe.export(a.out,s,mask,raw)
    atomic_json(str(a.out)+'.json',{'provenance':pipe.provenance,'seed':a.seed,'steps':a.steps,
                                 'proxy_class_cells':int(np.isin(b,[63,127]).sum()),'known_cells':int(mask.sum()) if mask is not None else 0,
                                 'target_data_version':pipe.r.data_version,
                                 'unregistered_known_states':sorted(set(raw[(s<0)&mask].tolist())) if raw is not None else [],
                                 'entity_scope':'blocks only; original input retains entities and tile entity NBT'})
if __name__=='__main__':main()
