"""Geometry Flow + independent full-state refinement, with exact known space."""
import json
from pathlib import Path
import numpy as np
import torch
from .codec import registry
from .images import prepare_image,encode_image
from .state_model import StateRefiner
from .containers import export_litematic
from .data import digest


class CompletionPipeline:
    def __init__(self,geometry_config,geometry_checkpoint,state_checkpoint,decoder_checkpoint,device='cuda'):
        from trellis.models import SparseStructureFlowModel,SparseStructureDecoder
        from transformers import AutoModel
        self.device=device;self.r=registry()
        cfg=json.loads(Path(geometry_config).read_text());self.config=cfg
        self.flow=SparseStructureFlowModel(**cfg['models']['denoiser']['args']).to(device).eval()
        self.flow.load_state_dict(torch.load(geometry_checkpoint,map_location=device,weights_only=True))
        self.decoder=SparseStructureDecoder(out_channels=163,latent_channels=8,num_res_blocks=2,
            num_res_blocks_middle=2,channels=[512,128,32],use_fp16=False).to(device).eval()
        self.decoder.load_state_dict(torch.load(decoder_checkpoint,map_location=device,weights_only=True))
        ck=torch.load(state_checkpoint,map_location=device,weights_only=False)
        if ck['contract']['registry_hash']!=self.r.hash or ck['contract']['image_contract']!='imagenet-518-center-v1':raise ValueError('Incompatible state checkpoint')
        self.training_contract=ck['contract'];self.classes=ck['classes'];self.refiner=StateRefiner(self.classes).to(device).eval();self.refiner.load_state_dict(ck['model'])
        self.dino=AutoModel.from_pretrained('facebook/dinov2-large',local_files_only=True).to(device).eval()
        norm=cfg['dataset']['args']['normalize'];self.mean=torch.tensor(norm['mean'],device=device).view(1,-1,1,1,1);self.std=torch.tensor(norm['std'],device=device).view(1,-1,1,1,1)
        if not torch.isfinite(self.std).all() or (self.std<=0).any():raise ValueError('Invalid latent normalization')
        self.legacy=torch.tensor([self.r.project_legacy(s) for s in range(len(self.r.names))],device=device,dtype=torch.long)
        self.provenance={'registry_hash':self.r.hash,'geometry_checkpoint_sha256':digest(geometry_checkpoint),
                         'state_checkpoint_sha256':digest(state_checkpoint),'decoder_checkpoint_sha256':digest(decoder_checkpoint),
                         'state_training_step':ck['step'],'image_contract':'imagenet-518-center-v1'}

    @torch.no_grad()
    def __call__(self,image,known_blocks=None,known_states=None,known_mask=None,steps=50,cfg_strength=1.5,seed=42):
        from trellis.pipelines.samplers.flow_euler import FlowEulerCfgSampler
        if not isinstance(steps,int) or steps<1:raise ValueError('steps must be positive')
        if torch.is_tensor(image):x=image.to(self.device);x=x[None] if x.ndim==3 else x
        else:x=prepare_image(image)[None].to(self.device)
        if len(x)!=1:raise ValueError('Pipeline accepts one completion at a time')
        shape=(1,64,64,64)
        if known_blocks is None:
            if known_states is not None or known_mask is not None:raise ValueError('Supply all known arrays together')
            b=torch.zeros(shape,device=self.device,dtype=torch.long);s=b.clone();mask=torch.zeros_like(b,dtype=torch.bool)
        else:
            if known_states is None or known_mask is None:raise ValueError('Supply all known arrays together')
            def tensor(a):
                t=torch.as_tensor(a,device=self.device);return t[None] if t.ndim==3 else t
            b,s,mask=map(tensor,(known_blocks,known_states,known_mask))
            if b.shape!=shape or s.shape!=shape or mask.shape!=shape or mask.dtype!=torch.bool:raise ValueError('Expected [64,64,64] IDs and boolean known mask')
            if b.is_floating_point() or s.is_floating_point():raise ValueError('Integer IDs required')
            if ((b[mask]<0)|(b[mask]>=128)).any():raise ValueError('Known class out of range')
            if ((s[mask]<-1)|(s[mask]>=len(self.r.names))).any():raise ValueError('Known registry ID out of range')
        cond=encode_image(self.dino,x)
        # Full states are only projected for the frozen geometry conditioner.
        safe=torch.where(mask & (s>=0),s,0).long();legacy=self.legacy[safe]
        kb=torch.where(mask,b,0).long()
        generator=torch.Generator(device=self.device).manual_seed(seed)
        noise=torch.randn(shape[0],8,16,16,16,device=self.device,generator=generator)
        lat=FlowEulerCfgSampler(1e-5).sample(self.flow,noise,cond,torch.zeros_like(cond),steps=steps,
            cfg_strength=cfg_strength,verbose=False,known_blocks=kb,known_states=legacy,known_mask=mask).samples
        logits=self.decoder(lat*self.std+self.mean)
        blocks=torch.where(mask,b,logits[:,:128].argmax(1)).long()
        # Never read state logits from the geometry VAE.
        predicted=self.refiner.predict(blocks,safe,mask&(s>=0),cond)
        states=torch.where(mask,s,predicted)
        return blocks[0].cpu().numpy(),states[0].cpu().numpy()

    def export(self,path,states,known_mask=None,known_raw=None):
        states=np.asarray(states);names=[]
        if states.ndim!=3 or not np.issubdtype(states.dtype,np.integer):raise ValueError('Expected 3D integer registry IDs')
        if (states>=len(self.r.names)).any() or (states < -1).any():raise ValueError('Registry ID out of range')
        # Palette-level conversion, followed by optional exact original known strings.
        unique,inv=np.unique(states,return_inverse=True)
        for sid in unique:
            if sid<0:
                names.append('minecraft:air') # replaced below, otherwise refused
            else:names.append(self.r.string(int(sid)))
        strings=np.array(names,dtype=object)[inv].reshape(states.shape)
        if known_raw is not None:
            mask=np.asarray(known_mask)
            if mask.dtype!=bool or mask.shape!=states.shape or np.asarray(known_raw).shape!=states.shape:raise ValueError('Known raw/mask shape mismatch')
            # Registered states use target-version canonical names. Only unsupported
            # known states retain original strings; never overwrite predictions.
            unresolved=states<0
            if (unresolved & ~mask).any():raise ValueError('Unregistered states outside known space')
            strings[unresolved]=np.asarray(known_raw,dtype=object)[unresolved]
        elif (states<0).any():raise ValueError('Unregistered known states require raw palette preservation')
        palette,indices=np.unique(strings.astype(str),return_inverse=True)
        export_litematic(path,indices.reshape(states.shape),palette.tolist(),data_version=self.r.data_version)
