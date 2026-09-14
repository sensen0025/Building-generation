"""Independent full-state predictor, conditioned on geometry/image/known states.

Frozen geometry latents are not required to encode state information. A head
predicts a legal local variant of each block class; no cross-class illegal props.
"""
import torch
from torch import nn
import torch.nn.functional as F


class StateRefiner(nn.Module):
    def __init__(self,contract,registry_size=27866,width=32,image_channels=1024):
        super().__init__();rows=contract['classes'];self.contract_hash=contract['hash']
        self.registry_hash=contract['registry_hash'];self.width=width
        self.register_buffer('counts',torch.tensor([r['count'] for r in rows],dtype=torch.long))
        self.register_buffer('starts',torch.tensor([r['min_state'] for r in rows],dtype=torch.long))
        self.register_buffer('defaults',torch.tensor([r['default_local'] for r in rows],dtype=torch.long))
        self.register_buffer('proxy',torch.tensor([r['proxy'] for r in rows],dtype=torch.bool))
        self.max_variants=max(r['count'] for r in rows)
        self.blocks=nn.Embedding(len(rows),width)
        self.states=nn.Embedding(registry_size,width)
        self.local=nn.Sequential(nn.Conv3d(width+1,width,3,padding=1),nn.SiLU(),nn.Conv3d(width,width,3,padding=1),nn.SiLU())
        self.down=nn.Sequential(nn.Conv3d(width,width*2,4,stride=2,padding=1),nn.SiLU(),nn.Conv3d(width*2,width*2,4,stride=2,padding=1),nn.SiLU())
        self.image=nn.Linear(image_channels,width*2)
        self.query_pos=nn.Linear(3,width*2)
        self.attention=nn.MultiheadAttention(width*2,4,batch_first=True)
        self.norm=nn.LayerNorm(width*2)
        self.head=nn.Sequential(nn.Conv3d(width*3,width,3,padding=1),nn.SiLU(),nn.Conv3d(width,self.max_variants,1))

    def forward(self,blocks,known_states,known_mask,image_tokens):
        if blocks.ndim!=4 or known_states.shape!=blocks.shape or known_mask.shape!=blocks.shape:
            raise ValueError('Expected aligned [B,X,Y,Z] arrays')
        if known_mask.dtype!=torch.bool or blocks.is_floating_point() or known_states.is_floating_point():
            raise ValueError('Integer IDs and boolean mask required')
        if (blocks<0).any() or (blocks>=len(self.counts)).any():raise ValueError('Block class out of range')
        safe=torch.where(known_mask,known_states,0)
        if (safe<0).any() or (safe>=self.states.num_embeddings).any():raise ValueError('Known state outside registry')
        f=self.blocks(blocks)+self.states(safe)*known_mask[...,None]
        f=self.local(torch.cat([f.permute(0,4,1,2,3),known_mask[:,None].to(f.dtype)],1))
        low=self.down(f);pooled=F.adaptive_avg_pool3d(low,(4,4,4))
        coords=torch.stack(torch.meshgrid(*[torch.linspace(-1,1,4,device=f.device,dtype=f.dtype)]*3,indexing='ij'),-1).reshape(1,64,3)
        q=pooled.flatten(2).transpose(1,2)+self.query_pos(coords)
        # Preserve sequence layout with deterministic pooling. Shared by all calls.
        img=F.adaptive_avg_pool1d(image_tokens.transpose(1,2),65).transpose(1,2)
        kv=self.image(img.to(f.dtype));att,_=self.attention(q,kv,kv,need_weights=False)
        context=self.norm(q+att).transpose(1,2).reshape(-1,self.width*2,4,4,4)
        context=F.interpolate(context,size=blocks.shape[1:],mode='trilinear',align_corners=False)
        logits=self.head(torch.cat([f,context],1))
        allowed=torch.arange(self.max_variants,device=f.device)[None,:,None,None,None]<self.counts[blocks][:,None]
        return logits.masked_fill(~allowed,torch.finfo(logits.dtype).min)

    def loss(self,logits,blocks,states,known_mask,valid):
        target=states-self.starts[blocks]
        use=valid&~known_mask&(blocks!=0)&~self.proxy[blocks]&(self.counts[blocks]>1)
        if ((target[use]<0)|(target[use]>=self.counts[blocks][use])).any():
            raise ValueError('Training state/class mismatch')
        if not use.any():return logits[:,0].sum()*0,0
        # Select before CE to avoid accidental invalid labels and air dominance.
        loss=F.cross_entropy(logits.permute(0,2,3,4,1)[use].float(),target[use].long())
        return loss,int(use.sum())

    @torch.no_grad()
    def predict(self,blocks,known_states,known_mask,image_tokens):
        result=self(blocks,known_states,known_mask,image_tokens).argmax(1)+self.starts[blocks]
        return torch.where(known_mask,known_states,result)
