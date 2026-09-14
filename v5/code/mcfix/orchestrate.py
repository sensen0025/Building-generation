"""Finite repair run: validate -> render -> train -> compare -> publish manifest.

Every phase is journaled. Failure stops dependent work; no unbounded restarts.
"""
import os,json,time,subprocess,sys
from pathlib import Path
from .data import atomic_json,digest

BASE=Path('/data/redstone_v5');CODE=BASE/'code';DATA=BASE/'dataset'
PY='/home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python';NODE='/home/linux/node-v22.14.0-linux-x64/bin/node'
TRELLIS='/home/linux/redstone_ai/TRELLIS';VAE=Path(TRELLIS)/'outputs/ss_vae_mc_v2/ckpts'
BASECFG='/data/redstone_v4/work/ss_flow_mc_v4_rotfix_cont.json'
BASECK=str(BASE/'backup_20260913/ckpts/denoiser_ema0.9999_step0032000.pt')
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',ATTN_BACKEND='sdpa',HF_HUB_OFFLINE='1',PYTHONPATH=TRELLIS+':'+str(CODE))


def phase(name,args,cwd=CODE,env=None):
    log=BASE/(name+'.log');atomic_json(BASE/'pipeline_status.json',{'phase':name,'status':'running','log':str(log),'time':time.time()})
    with open(log,'a') as f:
        p=subprocess.run(args,cwd=cwd,env=env or ENV,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'{name} failed ({p.returncode}): {log}')


def module(name,*args):return [PY,'-u','-m','mcfix.'+name,*map(str,args)]


def main():
    # Existing source build owns its own workers; wait for its atomic final manifest.
    atomic_json(BASE/'pipeline_status.json',{'phase':'source_rebuild','status':'waiting','time':time.time()})
    deadline=time.time()+6*3600
    while not (DATA/'manifest.json').exists():
        if time.time()>deadline:raise RuntimeError('Source rebuild exceeded six-hour bound')
        time.sleep(20)
    phase('normalize_versions',module('migrate_versions','--dataset',DATA))
    phase('verify_full',module('verify_data','--dataset',DATA,'--limit',200,'--out',BASE/'data_verification.json'))
    phase('prepare_renders',module('render_inputs','--dataset',DATA,'--out',BASE/'litematics','--limit',96))
    render_env=dict(ENV,MCFIX_LIT_DIR=str(BASE/'litematics'),MCFIX_RENDER_DIR=str(BASE/'renders'),MCFIX_SHARDS='3')
    atomic_json(BASE/'pipeline_status.json',{'phase':'render','status':'running','time':time.time()})
    procs=[]
    for i in range(3):
        log=open(BASE/f'render_{i}.log','a');env=dict(render_env,MCFIX_SHARD_ID=str(i))
        p=subprocess.Popen([NODE,'/home/linux/schem_render/render_v5.mjs','0','99999999','100',str(9210+i*2),str(9211+i*2)],cwd='/home/linux/schem_render',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        procs.append((p,log))
    codes=[p.wait() for p,log in procs]
    for p,log in procs:log.close()
    if any(codes):raise RuntimeError(f'Render shards failed: {codes}')
    phase('train_state',module('train_state','--dataset',DATA,'--renders',BASE/'renders','--tasks',BASE/'litematics/tasks.json','--out',BASE/'state','--steps',3000,'--save-every',500))
    state=BASE/'state/best.pt';metrics=json.loads((BASE/'state/best_validation.json').read_text())
    if metrics['state_accuracy']<metrics['default_accuracy']:
        phase('train_state_extend',module('train_state','--dataset',DATA,'--renders',BASE/'renders','--tasks',BASE/'litematics/tasks.json','--out',BASE/'state','--steps',6000,'--save-every',500,'--resume'))
        metrics=json.loads((BASE/'state/best_validation.json').read_text())
    if metrics['state_accuracy']<metrics['default_accuracy']:raise RuntimeError('State refiner did not beat default-state baseline; no promotion')
    phase('encode_geometry',module('encode_geometry','--dataset',DATA,'--tasks',BASE/'litematics/tasks.json','--renders',BASE/'renders','--out',BASE/'geometry_data','--encoder-checkpoint',VAE/'encoder_step0050000.pt','--base-config',BASECFG,'--base-checkpoint',BASECK,'--steps',1000))
    phase('train_geometry',[PY,'train.py','--config',str(BASE/'geometry_data/config.json'),'--output_dir',str(BASE/'geometry'),'--data_dir',TRELLIS+'/data/mc','--auto_retry','1'],cwd=TRELLIS)
    newck=BASE/'geometry/ckpts/denoiser_ema0.9999_step0001000.pt'
    if not newck.exists():raise RuntimeError('Missing completed geometry checkpoint')
    common=['--dataset',DATA,'--renders',BASE/'renders','--tasks',BASE/'litematics/tasks.json','--state-checkpoint',state,'--decoder-checkpoint',VAE/'decoder_step0050000.pt','--limit',8,'--steps',50]
    phase('eval_geometry_base',module('evaluate',*common,'--geometry-config',BASECFG,'--geometry-checkpoint',BASECK,'--out',BASE/'geometry_base_eval.json'))
    phase('eval_geometry_new',module('evaluate',*common,'--geometry-config',BASE/'geometry_data/config.json','--geometry-checkpoint',newck,'--out',BASE/'geometry_new_eval.json'))
    def score(path):
        samples=json.loads(Path(path).read_text())['samples']
        if not all(s['known_blocks_exact'] and s['known_states_exact'] for s in samples):raise RuntimeError('Known-space preservation failed')
        return sum(s['unknown_class_accuracy_on_gt_solid'] or 0 for s in samples)/len(samples)
    oldscore,newscore=score(BASE/'geometry_base_eval.json'),score(BASE/'geometry_new_eval.json')
    cfg,ck=(str(BASE/'geometry_data/config.json'),str(newck)) if newscore>=oldscore else (BASECFG,BASECK)
    phase('test_final',module('evaluate',*common,'--geometry-config',cfg,'--geometry-checkpoint',ck,'--split','test','--out',BASE/'test_eval.json'))
    score(BASE/'test_eval.json')
    active={'status':'experimental_generation','state_quality':'validated_on_given_geometry','geometry_quality':'experimental; review end-to-end metrics separately','geometry_config':cfg,'geometry_checkpoint':ck,'state_checkpoint':str(state),
            'decoder_checkpoint':str(VAE/'decoder_step0050000.pt'),'dataset_manifest':str(DATA/'manifest.json'),
            'dataset_sha256':digest(DATA/'manifest.json'),'state_validation':metrics,
            'geometry_validation_class_accuracy':{'baseline':oldscore,'corrected':newscore},
            'test_evaluation':str(BASE/'test_eval.json'),'scope':'128-class geometry with independent full registered states; source unsupported states retained separately'}
    atomic_json(BASE/'active.json',active);atomic_json(BASE/'pipeline_status.json',{'phase':'complete','status':'complete','active':str(BASE/'active.json'),'time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(BASE/'pipeline_status.json',{'status':'failed','error':str(e),'time':time.time()});raise
