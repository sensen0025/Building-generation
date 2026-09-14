"""Train full-state refinement; atomic checkpoints and group-clean evaluation."""
import argparse,json,time,os,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,default_collate
from .learning_data import StateDataset
from .state_model import StateRefiner
from .images import encode_image
from .data import atomic_json,digest


def evaluate(model,loader,dino,device,max_batches=None):
    model.eval();total=correct=default_correct=0;loss_sum=0.;cells=0
    with torch.no_grad():
        for i,b in enumerate(loader):
            if max_batches and i>=max_batches:break
            b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}
            tokens=encode_image(dino,b['image']);logits=model(b['blocks'],b['states'],b['known_mask'],tokens)
            loss,n=model.loss(logits,b['blocks'],b['states'],b['known_mask'],b['valid'])
            use=b['valid']&~b['known_mask']&(b['blocks']!=0)&~model.proxy[b['blocks']]&(model.counts[b['blocks']]>1)
            pred=logits.argmax(1)+model.starts[b['blocks']]
            correct+=int(((pred==b['states'])&use).sum());total+=int(use.sum())
            default=model.defaults[b['blocks']]+model.starts[b['blocks']]
            default_correct+=int(((default==b['states'])&use).sum());loss_sum+=float(loss)*n;cells+=n
    model.train()
    return {'state_accuracy':correct/total if total else None,'default_accuracy':default_correct/total if total else None,
            'state_ce':loss_sum/cells if cells else None,'evaluated_cells':total}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--renders',required=True);p.add_argument('--tasks',required=True)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=5000);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--save-every',type=int,default=500);p.add_argument('--resume',action='store_true');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    tr=StateDataset(a.dataset,a.renders,a.tasks,'train');va=StateDataset(a.dataset,a.renders,a.tasks,'validation')
    if tr.groups&va.groups:raise ValueError('Source group leakage')
    if tr.render_contracts!=va.render_contracts:raise ValueError('Training/validation renderer mismatch')
    # Each step's examples/masks are deterministic, including after resume.
    val=DataLoader(va,batch_size=1,shuffle=False,num_workers=1)
    from transformers import AutoModel
    dino=AutoModel.from_pretrained('facebook/dinov2-large',local_files_only=True).to(device).eval()
    model=StateRefiner(tr.contract).to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    step=0;attempt=0;contract={'registry_hash':tr.r.hash,'class_contract_hash':tr.contract['hash'],
                    'dataset_manifest_sha256':digest(Path(a.dataset)/'manifest.json'),
                    'render_tasks_sha256':digest(a.tasks),'image_contract':'imagenet-518-center-v1',
                    'model':'StateRefiner-v1','implementation_sha256':digest(Path(__file__).with_name('state_model.py')),'renderer_contracts':sorted(tr.render_contracts),'train_groups':sorted(tr.groups),'validation_groups':sorted(va.groups)}
    latest=out/'latest.pt'
    if a.resume and latest.exists():
        ck=torch.load(latest,map_location=device,weights_only=False)
        if ck['contract']!=contract:raise ValueError('Checkpoint/data contract mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];attempt=ck.get('attempt',step)
        torch.set_rng_state(ck['torch_rng'].cpu());np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
        if device=='cuda' and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    if step>=a.steps:
        if not a.resume:raise ValueError('steps must be positive')
        atomic_json(out/'complete.json',{'step':step,'checkpoint_sha256':digest(latest),'validation':ck['validation']})
        print(json.dumps({'already_complete':step}),flush=True);return
    atomic_json(out/'run.json',{'args':vars(a),'contract':contract,'train_images':len(tr),'validation_images':len(va)})
    start=time.time();log=open(out/'train.jsonl','a',buffering=1)
    best_score=json.loads((out/'best_validation.json').read_text())['state_accuracy'] if (out/'best_validation.json').exists() and a.resume else -1
    baseline=evaluate(model,val,dino,device);atomic_json(out/'before.json',baseline);print(json.dumps({'baseline':baseline}),flush=True)
    while step<a.steps:
        rng=np.random.RandomState(a.seed+attempt)
        indices=rng.randint(0,len(tr),size=a.batch)
        b=default_collate([tr.get_sample(int(index),seed=a.seed+attempt*a.batch+j) for j,index in enumerate(indices)])
        b={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
        with torch.no_grad():tokens=encode_image(dino,b['image'])
        drop=torch.rand(tokens.shape[0],device=device)<.15;tokens=tokens.masked_fill(drop[:,None,None],0)
        opt.zero_grad(set_to_none=True)
        logits=model(b['blocks'],b['states'],b['known_mask'],tokens)
        loss,count=model.loss(logits,b['blocks'],b['states'],b['known_mask'],b['valid'])
        attempt+=1
        if not count:
            if attempt>max(100,a.steps*100):raise RuntimeError('No sufficient supervised state batches')
            continue
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite state loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
        if step%10==0:
            row={'step':step,'loss':float(loss),'cells':count,'seconds':round(time.time()-start,2)};log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        if step%a.save_every==0 or step==a.steps:
            metrics=evaluate(model,val,dino,device)
            ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'attempt':attempt,'contract':contract,'classes':tr.contract,
                'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None,
                'validation':metrics}
            tmp=out/'latest.tmp';torch.save(ck,tmp);tmp.replace(latest)
            if not metrics['evaluated_cells']:raise RuntimeError('Validation has no observed state labels')
            if metrics['state_accuracy']>best_score:
                import shutil
                shutil.copy2(latest,out/'best.tmp');(out/'best.tmp').replace(out/'best.pt');best_score=metrics['state_accuracy']
                atomic_json(out/'best_validation.json',{'step':step,**metrics})
            atomic_json(out/'validation.json',{'step':step,**metrics});print(json.dumps({'step':step,'validation':metrics}),flush=True)
    atomic_json(out/'complete.json',{'step':step,'checkpoint_sha256':digest(latest),'validation':metrics})
if __name__=='__main__':main()
