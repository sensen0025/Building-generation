import unittest,tempfile
from pathlib import Path
import numpy as np
from mcfix.codec import registry,rotate_volume
from mcfix.data import finalize_dataset,digest

class SplitTests(unittest.TestCase):
    def test_duplicate_rotation_of_historical_source_never_enters_holdout(self):
        r=registry()
        contract={'hash':'fixture','classes':[{'count':1,'proxy':False},{'count':2,'proxy':False}]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=[]
            base=np.zeros((8,8,8),np.uint32);base[1:5,2:6,3:5]=1
            different=base.copy();different[5,2,3]=1
            another=base.copy();another[6,3,3]=1
            for i,raw in enumerate([base,rotate_volume(base,1),different,another]):
                path=root/f'{i}.npz'
                np.savez_compressed(path,raw=raw,palette=np.array(['minecraft:air','minecraft:lever']),
                                    blocks=(raw!=0).astype(np.uint8),valid=np.ones(raw.shape,bool),observed=np.ones(raw.shape,bool))
                sample={'id':str(i),'path':str(path),'group':f'g{i}','source_sha256':str(i),'supported_fraction':1.,'sample_sha256':digest(path)}
                records.append({'sha256':str(i),'status':'ok','group':f'g{i}','historic_seen':i==0,'meta':{'data_version':4189},'samples':[sample]})
            result=finalize_dataset(root,records,contract)
            items=result['samples']
            self.assertEqual(items[0]['group'],items[1]['group'])
            self.assertTrue(all(s['split']=='train' and s['historic_seen'] for s in items[:2]))
            self.assertEqual({s['split'] for s in items[2:]},{'validation','test'})
            self.assertEqual(result['split_counts'],{'train':2,'validation':1,'test':1})
