"""Exercise the public active-manifest CLI and re-read its actual Litematic."""
import os,json,subprocess,sys
from pathlib import Path
import numpy as np
from .codec import registry
from .containers import decode_path
from .data import atomic_json,digest
from .learning_data import StateDataset

def main():
    base=Path('/data/redstone_v5');out=base/'public_example';out.mkdir(exist_ok=True)
    ds=StateDataset(base/'dataset',base/'renders',base/'litematics/tasks.json','test')
    sample=ds[0];item,task,image=ds.items[0];mask=sample['known_mask'].numpy()
    if not mask.any():mask[:,:32,:]=True
    np.save(out/'known_mask.npy',mask)
    output=out/'completion.litematic'
    command=[sys.executable,'-m','mcfix.generate','--active',str(base/'active.json'),'--image',str(image),
             '--known-litematic',task['litematic'],'--known-mask',str(out/'known_mask.npy'),'--out',str(output),'--steps','50']
    subprocess.run(command,check=True,env=os.environ)
    r=registry()
    def states(path):
        decoded=decode_path(path);ids=np.array([r.normalize_source(s,data_version=decoded['meta']['data_version'])[0] for s in decoded['names']])
        return ids[decoded['vol'].transpose(2,0,1)],decoded
    expected,_=states(task['litematic']);actual,decoded=states(output)
    assert actual.shape==expected.shape==(64,64,64)
    assert np.array_equal(actual[mask],expected[mask])
    report={'command':command,'source_sample':item['id'],'known_cells':int(mask.sum()),'known_state_mismatches':int((actual[mask]!=expected[mask]).sum()),
            'registered_output_cells':int(actual.size),'palette_size':len(decoded['names']),'output_sha256':digest(output),'target_data_version':decoded['meta']['data_version'],
            'scope':'CLI/export contract check on one test example; no additional model selection'}
    atomic_json(base/'public_cli_verification.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
