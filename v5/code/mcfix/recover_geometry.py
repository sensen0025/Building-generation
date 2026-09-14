"""Rerun corrective geometry training after fixing EMA finetune initialization."""
import json,fcntl,time,subprocess
from pathlib import Path
from .orchestrate import BASE,DATA,PY,TRELLIS,VAE,BASECFG,BASECK,ENV,phase,module
from .data import atomic_json,digest

def main():
    lock=open(BASE/'geometry_recovery.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output=BASE/'geometry_ema_fixed'
    if output.exists():raise RuntimeError('Recovery directory exists; inspect it before starting a new run')
    active=json.loads((BASE/'active.json').read_text());active.update(status='experimental_generation',geometry_quality='not_accepted',repair_status='correcting_ema_initialization')
    atomic_json(BASE/'active.json',active)
    phase('train_geometry_ema_fixed',[PY,'train.py','--config',str(BASE/'geometry_data/config.json'),'--output_dir',str(output),'--data_dir',TRELLIS+'/data/mc','--auto_retry','1'],cwd=TRELLIS)
    state=BASE/'state/best.pt'
    common=['--dataset',DATA,'--renders',BASE/'renders','--tasks',BASE/'litematics/tasks.json','--state-checkpoint',state,'--decoder-checkpoint',VAE/'decoder_step0050000.pt','--limit',8,'--steps',50]
    candidates=[{'name':'baseline','config':BASECFG,'checkpoint':BASECK,'evaluation':str(BASE/'geometry_base_eval.json')}]
    for kind,suffix in [('ema','_ema0.9999'),('raw','')]:
        ck=output/'ckpts'/f'denoiser{suffix}_step0001000.pt';report=BASE/f'geometry_fixed_{kind}_eval.json'
        phase(f'eval_geometry_fixed_{kind}',module('evaluate',*common,'--geometry-config',BASE/'geometry_data/config.json','--geometry-checkpoint',ck,'--out',report))
        candidates.append({'name':kind,'config':str(BASE/'geometry_data/config.json'),'checkpoint':str(ck),'evaluation':str(report)})
    for c in candidates:
        rows=json.loads(Path(c['evaluation']).read_text())['samples']
        if len(rows)!=8 or not all(s['known_blocks_exact'] and s['known_states_exact'] for s in rows):raise RuntimeError('Incomplete evaluation or known-space preservation failure')
        c['class_accuracy']=sum(s['unknown_class_accuracy_on_gt_solid'] or 0 for s in rows)/len(rows)
        c['occupancy_iou']=sum(s['unknown_occupancy_iou'] or 0 for s in rows)/len(rows)
    selected=max(candidates,key=lambda c:c['class_accuracy'])
    phase('test_final_ema_fixed',module('evaluate',*common,'--geometry-config',selected['config'],'--geometry-checkpoint',selected['checkpoint'],'--split','test','--out',BASE/'test_final_eval.json'))
    active.update(geometry_config=selected['config'],geometry_checkpoint=selected['checkpoint'],
                  geometry_candidates=candidates,
                  geometry_validation_class_accuracy={'baseline':candidates[0]['class_accuracy'],'corrected':selected['class_accuracy']},
                  previous_incorrect_ema_evaluation=str(BASE/'geometry_new_eval.json'),geometry_selection='highest mean unknown class accuracy on eight validation source groups',
                  test_evaluation=str(BASE/'test_final_eval.json'),repair_status='complete',
                  state_quality='validated_on_given_geometry',geometry_quality='experimental; end-to-end metrics must be reviewed separately',
                  ema_finetune_implementation_sha256=digest(Path(TRELLIS)/'trellis/trainers/basic.py'))
    atomic_json(BASE/'active.json',active);atomic_json(BASE/'pipeline_status.json',{'phase':'complete','status':'complete','generation_quality':'experimental','active':str(BASE/'active.json'),'time':time.time()})
    print(json.dumps({'selected':selected,'candidates':candidates}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(BASE/'pipeline_status.json',{'status':'failed','error':str(e),'time':time.time()});raise
