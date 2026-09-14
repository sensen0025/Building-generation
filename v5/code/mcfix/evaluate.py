"""Held-out source-group completion evaluation, distinct from training diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
from .learning_data import StateDataset
from .pipeline import CompletionPipeline
from .metrics import completion_metrics
from .data import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--renders',required=True);p.add_argument('--tasks',required=True)
    p.add_argument('--state-checkpoint',required=True);p.add_argument('--geometry-config',required=True);p.add_argument('--geometry-checkpoint',required=True);p.add_argument('--decoder-checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--limit',type=int,default=24);p.add_argument('--steps',type=int,default=50);p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args();ds=StateDataset(a.dataset,a.renders,a.tasks,a.split)
    smoke=ds.manifest.get('smoke_only',False)
    if smoke and not a.allow_smoke:raise ValueError('Smoke fixtures are not held-out geometry evaluation')
    pipe=CompletionPipeline(a.geometry_config,a.geometry_checkpoint,a.state_checkpoint,a.decoder_checkpoint)
    if pipe.classes['hash']!=ds.contract['hash']:raise ValueError('Model/dataset class contract mismatch')
    if not smoke and pipe.training_contract['dataset_manifest_sha256']!=digest(Path(a.dataset)/'manifest.json'):raise ValueError('State checkpoint belongs to another dataset')
    rows=[]
    seen=set();first=[];rest=[]
    for index,(item,task,image) in enumerate(ds.items):
        if item['group'] in seen:rest.append(index)
        else:first.append(index);seen.add(item['group'])
    for i,index in enumerate((first+rest)[:a.limit]):
        sample=ds[index];g=sample['blocks'].numpy();s=sample['states'].numpy();known=sample['known_mask'].numpy()
        if not known.any():known[:,:32,:]=True
        b,st=pipe(sample['image'],g,s,known,steps=a.steps,seed=42+i)
        metrics=completion_metrics(g,b,s,st,known,sample['valid'].numpy())
        item,task,image=ds.items[index]
        rows.append({'slot':sample['slot'],'sample_id':item['id'],'source_sha256':item['source_sha256'],'group':item['group'],'seed':42+i,**metrics});print(json.dumps(rows[-1]),flush=True)
        atomic_json(a.out,{'evaluation_kind':'end_to_end_smoke' if smoke else 'held_out_source_groups','split':a.split,'steps':a.steps,'provenance':pipe.provenance,'dataset_manifest_sha256':digest(Path(a.dataset)/'manifest.json'),'samples':rows})
if __name__=='__main__':main()
