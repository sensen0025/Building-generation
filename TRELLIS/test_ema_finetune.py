import copy,tempfile,unittest
from pathlib import Path
import torch
from trellis.trainers.basic import BasicTrainer

class EmaFinetuneTests(unittest.TestCase):
    def test_finetune_initializes_every_ema_from_loaded_weights(self):
        model=torch.nn.Linear(3,2)
        trainer=BasicTrainer.__new__(BasicTrainer)
        trainer.models={'denoiser':model};trainer.is_master=True;trainer.world_size=1;trainer.fp16_mode=None
        trainer.master_params=list(model.parameters());trainer.ema_rate=[.9,.9999]
        trainer.ema_params=[copy.deepcopy(trainer.master_params) for _ in trainer.ema_rate]
        target={k:torch.full_like(v,7.) for k,v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'weights.pt';torch.save(target,path)
            trainer.finetune_from({'denoiser':str(path)})
        for ema in trainer.ema_params:
            for actual,expected in zip(ema,trainer.master_params):
                self.assertTrue(torch.equal(actual,expected))
                self.assertNotEqual(actual.data_ptr(),expected.data_ptr())
        trainer.update_ema()
        for ema in trainer.ema_params:
            for actual,expected in zip(ema,trainer.master_params):self.assertTrue(torch.allclose(actual,expected))

if __name__=='__main__':unittest.main()
