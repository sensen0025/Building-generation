"""One real-data optimizer step; does not save or change training checkpoints."""
import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['HF_HUB_OFFLINE'] = '1'
import json
import torch
from trellis.models import SparseStructureFlowModel
from trellis.datasets.mc_flow import MCFlowImageDataset
from trellis.trainers.flow_matching.dense_flow import DenseImageFlowMatchingCFGTrainer


def main():
    torch.manual_seed(123)
    with open('configs/flow/ss_flow_mc_inpaint_v4.json') as handle:
        config = json.load(handle)
    args = config['models']['denoiser']['args']
    model = SparseStructureFlowModel(**args).cuda()
    checkpoint = 'outputs/ss_flow_mc_img_v3_lr5e5_165k/ckpts/denoiser_ema0.9999_step0164000.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys
    assert all(k.startswith('voxel_conditioner.') for k in incompatible.missing_keys)
    print('Real checkpoint migration:', len(incompatible.missing_keys), 'new tensors', flush=True)
    dataset = MCFlowImageDataset('', **config['dataset']['args'])
    samples = []
    for i in range(4):
        dataset.inpaint_probability = 1.0 if i < 2 else 0.0
        samples.append(dataset[i])
    data = {k: v.cuda() for k, v in dataset.collate_fn(samples).items()}
    trainer = object.__new__(DenseImageFlowMatchingCFGTrainer)
    trainer.training_models = {'denoiser': model}
    trainer.sigma_min = 1e-5
    trainer.t_schedule = config['trainer']['args']['t_schedule']
    trainer.p_uncond = 0.1
    trainer.inpaint_image_drop_probability = 0.25
    trainer.image_cond_model_name = config['trainer']['args']['image_cond_model']
    trainer.image_cond_model = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses, _ = trainer.training_losses(**data)
        assert torch.isfinite(losses['loss'])
        losses['loss'].backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        if step == 0:
            assert model.voxel_conditioner.projection.weight.grad.abs().sum() > 0
        else:
            assert model.voxel_conditioner.blocks.weight.grad.abs().sum() > 0
        optimizer.step()
        print('Step', step, 'loss', losses['loss'].item(), 'inpaint_fraction',
              losses['inpaint_fraction'].item(), flush=True)
    print('Peak allocated GiB:', round(torch.cuda.max_memory_allocated() / 2**30, 3), flush=True)
    print('GPU validation passed; no checkpoint saved', flush=True)


if __name__ == '__main__':
    main()
