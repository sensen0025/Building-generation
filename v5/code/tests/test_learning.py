import unittest
import numpy as np
import torch
from mcfix.state_model import StateRefiner
from mcfix.images import normalize_image,encode_image
from mcfix.metrics import completion_metrics

class LearningTests(unittest.TestCase):
    def model(self):
        c={'hash':'test','registry_hash':'test','classes':[
           {'count':1,'min_state':0,'default_local':0,'proxy':False},
           {'count':2,'min_state':1,'default_local':0,'proxy':False}]}
        return StateRefiner(c,registry_size=3,width=8,image_channels=16)
    def test_unknown_state_values_cannot_leak(self):
        m=self.model().eval();b=torch.ones((1,8,8,8),dtype=torch.long);mask=torch.zeros_like(b,dtype=torch.bool);mask[:,:4]=True
        s=torch.ones_like(b);dirty=s.clone();dirty[~mask]=99999;image=torch.randn(1,5,16)
        with torch.no_grad():self.assertTrue(torch.equal(m(b,s,mask,image),m(b,dirty,mask,image)))
        dirty[mask]=-1
        with self.assertRaises(ValueError):m(b,dirty,mask,image)
    def test_full_state_loss_and_known_preservation(self):
        m=self.model();b=torch.ones((1,8,8,8),dtype=torch.long);s=torch.ones_like(b)*2;mask=torch.zeros_like(b,dtype=torch.bool);mask[:,:4]=True;image=torch.randn(1,5,16)
        logits=m(b,s,mask,image);loss,count=m.loss(logits,b,s,mask,torch.ones_like(mask));self.assertEqual(count,256);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertGreater(m.head[-1].weight.grad.abs().sum().item(),0)
        pred=m.predict(b,s,mask,image);self.assertTrue(torch.equal(pred[mask],s[mask]));self.assertTrue(((pred>=1)&(pred<=2)).all())
    def test_empty_supervision_loss_finite(self):
        m=self.model();b=torch.zeros((1,8,8,8),dtype=torch.long);s=b.clone();mask=torch.zeros_like(b,dtype=torch.bool);image=torch.zeros(1,5,16)
        logits=m(b,s,mask,image);loss,count=m.loss(logits,b,s,mask,mask);self.assertEqual(count,0);self.assertTrue(torch.isfinite(loss));loss.backward()
    def test_water_class_outside_legacy_state_range(self):
        from mcfix.codec import registry
        r=registry();water=r.blocks['minecraft:water'];start=water['minStateId']
        rows=[{'count':1,'min_state':0,'default_local':0,'proxy':False} for _ in range(74)]
        rows[73]={'count':16,'min_state':start,'default_local':0,'proxy':False}
        m=StateRefiner({'classes':rows,'hash':'test','registry_hash':r.hash},width=8,image_channels=16)
        b=torch.full((1,8,8,8),73,dtype=torch.long);s=torch.full_like(b,start+7);mask=torch.zeros_like(b,dtype=torch.bool);mask[:,:4]=True
        pred=m.predict(b,s,mask,torch.zeros(1,5,16))
        self.assertTrue(torch.equal(pred[mask],s[mask]));self.assertTrue(((pred>=start)&(pred<start+16)).all())
    def test_same_preprocess_for_training_and_inference(self):
        class Fake:
            def __call__(self,x):
                class Output:pass
                o=Output();o.last_hidden_state=x;return o
        x=torch.rand(2,3,8,8)
        self.assertTrue(torch.equal(encode_image(Fake(),x),normalize_image(x)))
        with self.assertRaises(ValueError):normalize_image(x+1)
    def test_full_state_accuracy_uses_canonical_identity(self):
        gt=np.ones((2,2,2),int);pred=np.full_like(gt,2)
        states=np.full_like(gt,17);known=np.zeros_like(gt,bool)
        metrics=completion_metrics(gt,pred,states,states,known)
        self.assertEqual(metrics['unknown_class_accuracy_on_gt_solid'],0)
        self.assertEqual(metrics['unknown_full_state_accuracy'],1)

    def test_six_faces_without_wrap(self):
        for axis in range(3):
            g=np.ones((4,4,4),int);p=g.copy();known=np.zeros_like(g,bool)
            sl=[slice(None)]*3;sl[axis]=slice(0,2);known[tuple(sl)]=True
            result=completion_metrics(g,p,g,p,known)
            self.assertEqual(result['boundary_edge_count'],16)
            self.assertEqual(result['boundary_adjacency_accuracy'],1)
            p[~known]=2
            result=completion_metrics(g,p,g,p,known)
            self.assertEqual(result['unknown_occupancy_iou'],1);self.assertEqual(result['unknown_class_accuracy_on_gt_solid'],0)

if __name__=='__main__':unittest.main()
