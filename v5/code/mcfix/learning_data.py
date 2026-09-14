from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .codec import registry,rotate_volume
from .images import prepare_image
from .data import digest


def sample_known_mask(blocks,up_axis=1,rng=None):
    rng=rng if rng is not None else np.random
    if blocks.ndim!=3 or min(blocks.shape)<2 or up_axis not in (0,1,2):raise ValueError('Invalid voxel grid/up axis')
    coords=np.argwhere(blocks!=0);low=coords.min(0) if len(coords) else np.zeros(3,int);high=coords.max(0)+1 if len(coords) else np.array(blocks.shape)
    mask=np.zeros(blocks.shape,bool);mode=int(rng.choice(3))
    if mode<2:
        axis=up_axis if mode==0 else int(rng.choice([a for a in range(3) if a!=up_axis]))
        cut=int(np.clip(low[axis]+rng.uniform(.25,.7)*(high[axis]-low[axis]),1,blocks.shape[axis]-1))
        sl=[slice(None)]*3;sl[axis]=slice(0,cut) if mode==0 or rng.random()<.5 else slice(cut,None);mask[tuple(sl)]=True
    else:
        mask[:]=True;hole=[]
        for axis,size in enumerate(blocks.shape):
            width=max(1,int((high[axis]-low[axis])*rng.uniform(.35,.75)))
            start=int(rng.randint(int(low[axis]),max(int(low[axis])+1,int(high[axis])-width+1)))
            hole.append(slice(start,min(size,start+width)))
        mask[tuple(hole)]=False
    return mask


class StateDataset(Dataset):
    def __init__(self,dataset,render_dir,tasks_path,split,seed=42):
        self.root=Path(dataset);self.render_dir=Path(render_dir);self.split=split;self.seed=seed
        manifest=json.loads((self.root/'manifest.json').read_text());self.manifest=manifest
        self.contract=json.loads((self.root/'classes.json').read_text());self.r=registry()
        if manifest['registry_hash']!=self.r.hash or manifest['class_contract_hash']!=self.contract['hash']:raise ValueError('Dataset schema mismatch')
        byid={i['id']:i for i in manifest['samples']};self.items=[];self.render_contracts=set()
        for task in json.loads(Path(tasks_path).read_text()):
            item=byid[task['sample_id']]
            if item['split']!=split:continue
            if split!='train' and item['historic_seen']:raise ValueError('Validation/test contaminated by historical training')
            if task['sample_sha256']!=item['sample_sha256'] or task['registry_hash']!=self.r.hash:raise ValueError('Stale render inputs')
            image=self.render_dir/f'bld_{task["slot"]:05d}.png';stamp=image.with_suffix('.png.json')
            if not image.exists() or not stamp.exists():continue
            metadata=json.loads(stamp.read_text())
            if metadata['key']['input']!=task['litematic_sha256'] or metadata['output']!=digest(image):raise ValueError('Stale/corrupt rendered image')
            self.render_contracts.add(json.dumps({k:v for k,v in metadata['key'].items() if k!='input'},sort_keys=True))
            if item['supported_fraction']<.85:continue
            self.items.append((item,task,image))
        if len(self.render_contracts)>1:raise ValueError('Mixed renderer/resource/camera contracts')
        if not self.items:raise ValueError(f'No verified rendered samples for {split}')
        self.groups={i[0]['group'] for i in self.items}

    def __len__(self):return len(self.items)

    def __getitem__(self,index):
        return self.get_sample(index)

    def get_sample(self,index,seed=None):
        item,task,image=self.items[index];k=task['rotation']
        with np.load(item['path']) as a:
            blocks=rotate_volume(a['blocks'],k);states=rotate_volume(a['states'],k)
            valid=rotate_volume(a['valid'],k);observed=rotate_volume(a['observed'],k)
        supported=states>=0
        states=np.where(supported,self.r.rotation[k,np.maximum(states,0)],-1).astype(np.int64)
        # Defaults are valid states, but are not treated as observed source labels.
        valid=valid&observed
        rng=np.random.RandomState(seed) if seed is not None else (np.random if self.split=='train' else np.random.RandomState(self.seed+task['slot']))
        mask=sample_known_mask(blocks,rng=rng) if rng.random()<.65 else np.zeros_like(blocks,bool)
        mask &= supported
        return {'blocks':torch.from_numpy(blocks.astype(np.int64)),
                'states':torch.from_numpy(states),'known_mask':torch.from_numpy(mask),
                'valid':torch.from_numpy(valid),'image':prepare_image(image),
                'slot':task['slot']}
