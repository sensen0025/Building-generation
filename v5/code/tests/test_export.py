import unittest,tempfile
from pathlib import Path
import numpy as np
from mcfix.pipeline import CompletionPipeline
from mcfix.codec import registry
from mcfix.containers import decode_path

class ExportTests(unittest.TestCase):
    def pipe(self):
        p=CompletionPipeline.__new__(CompletionPipeline);p.r=registry();return p
    def test_registered_known_uses_target_version_name(self):
        p=self.pipe();sid,_=p.r.normalize_source('minecraft:grass',data_version=3700)
        a=np.full((2,2,2),sid,np.int32)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'out.litematic';p.export(path,a,np.ones(a.shape,bool),np.full(a.shape,'minecraft:grass',object))
            decoded=decode_path(path)
            self.assertIn('minecraft:short_grass',decoded['names'])
            self.assertNotIn('minecraft:grass',decoded['names'])
    def test_unregistered_only_allowed_in_known_space(self):
        p=self.pipe();a=np.full((2,2,2),-1,np.int32);mask=np.ones(a.shape,bool);raw=np.full(a.shape,'mod:custom',object)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'out.litematic';p.export(path,a,mask,raw)
            self.assertIn('mod:custom',decode_path(path)['names'])
            mask[0,0,0]=False
            with self.assertRaises(ValueError):p.export(path,a,mask,raw)
            with self.assertRaises(ValueError):p.export(path,a)
