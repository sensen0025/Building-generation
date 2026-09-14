"""Encode geometry-only targets; legacy states are explicit conditioner features."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .codec import registry,rotate_volume
from .data import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--tasks',required=True);p.add_argument('--renders',required=True);p.add_argument('--out',required=True)
    p.add_argument('--encoder-checkpoint',required=True);p.add_argument('--base-config',required=True);p.add_argument('--base-checkpoint',required=True);p.add_argument('--steps',type=int,default=2000)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True)
    r=registry();m=json.loads((Path(a.dataset)/'manifest.json').read_text());byid={i['id']:i for i in m['samples']}
    tasks=json.loads(Path(a.tasks).read_text());tasks=[t for t in tasks if byid[t['sample_id']]['supported_fraction']>=.85]
    N=len(tasks)
    if not N:raise ValueError('No tasks')
    if (out/'manifest.json').exists():raise RuntimeError('Refuse to overwrite an encoded dataset; select a new output directory')
    # Stage into a new run directory. The ready manifest is written last.
    from trellis.models import SparseStructureEncoder
    enc=SparseStructureEncoder(in_channels=128,latent_channels=8,num_res_blocks=2,num_res_blocks_middle=2,channels=[32,128,512],use_fp16=False).cuda().eval()
    enc.load_state_dict(torch.load(a.encoder_checkpoint,map_location='cpu',weights_only=True))
    blocks=np.lib.format.open_memmap(out/'blocks.npy.tmp',mode='w+',dtype=np.uint8,shape=(N,64,64,64))
    features=np.lib.format.open_memmap(out/'legacy_condition_features.npy.tmp',mode='w+',dtype=np.uint16,shape=blocks.shape)
    latents=np.lib.format.open_memmap(out/'latents.npy.tmp',mode='w+',dtype=np.float32,shape=(N,8,16,16,16))
    legacy=np.array([r.project_legacy(i) for i in range(len(r.names))],np.uint16)
    rows=[];splits={s:[] for s in ['train','validation','test']}
    for i,t in enumerate(tasks):
        item=byid[t['sample_id']];k=t['rotation'];image=Path(a.renders)/f'bld_{t["slot"]:05d}.png'
        stamp=json.loads(image.with_suffix('.png.json').read_text())
        if stamp['key']['input']!=t['litematic_sha256'] or stamp['output']!=digest(image):raise ValueError('Unverified condition render')
        with np.load(item['path']) as data:
            b=rotate_volume(data['blocks'],k);s=rotate_volume(data['states'],k)
        valid=s>=0;s=r.rotation[k,np.maximum(s,0)];s=np.where(valid,legacy[s],0)
        blocks[i]=b;features[i]=s
        x=F.one_hot(torch.from_numpy(b.astype(np.int64)).cuda(),128).permute(3,0,1,2)[None].float()
        with torch.no_grad():z=enc(x,sample_posterior=False)
        latents[i]=z[0].cpu().numpy()
        link=out/'images'/f'bld_{i:04d}.png'
        if link.is_symlink():link.unlink()
        elif link.exists():raise ValueError('Refuse to replace non-symlink image')
        link.symlink_to(image.resolve());splits[item['split']].append(i)
        rows.append({'index':i,**t,'group':item['group']})
        if i%25==0:print(json.dumps({'encoded':i+1,'total':N}),flush=True)
    blocks.flush();features.flush();latents.flush();del blocks,features,latents
    for name in ['blocks.npy','legacy_condition_features.npy','latents.npy']:(out/(name+'.tmp')).replace(out/name)
    for split,ids in splits.items():atomic_json(out/(split+'_indices.json'),ids)
    if not splits['train'] or not splits['validation']:raise ValueError('Missing training/validation split')
    cfg=json.loads(Path(a.base_config).read_text());d=cfg['dataset']['args'];d.update(latent_npy=str(out/'latents.npy'),blocks_npy=str(out/'blocks.npy'),states_npy=str(out/'legacy_condition_features.npy'),image_dir=str(out/'images'),indices_path=str(out/'train_indices.json'))
    tr=cfg['trainer']['args'];tr['max_steps']=a.steps;tr['i_save']=min(500,a.steps);tr['i_sample']=1000000;tr['finetune_ckpt']={'denoiser':a.base_checkpoint}
    tr['lr_scheduler']['args']['T_max']=a.steps;tr['optimizer']['args']['lr']=2e-5
    cfg['mcfix_contract']={'representation':'geometry-only VAE + independent full-state refiner','registry_hash':r.hash,
        'dataset_manifest_sha256':digest(Path(a.dataset)/'manifest.json'),'render_tasks_sha256':digest(a.tasks),
        'encoder_checkpoint_sha256':digest(a.encoder_checkpoint),'base_checkpoint_sha256':digest(a.base_checkpoint),
        'latent_normalization':'fixed pretrained coordinate frame; no validation statistics recomputed'}
    atomic_json(out/'config.json',cfg);atomic_json(out/'manifest.json',{'rows':rows,'splits':{s:len(v) for s,v in splits.items()},'contract':cfg['mcfix_contract'],'array_sha256':{name:digest(out/name) for name in ['blocks.npy','legacy_condition_features.npy','latents.npy']}})
if __name__=='__main__':main()
