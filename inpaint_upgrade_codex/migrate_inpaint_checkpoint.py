"""Expand a flow state dict for v4, rejecting unrelated incompatibilities."""
import argparse
import json
from pathlib import Path
import torch
from trellis.models import SparseStructureFlowModel


def migrate(source, config, destination):
    if Path(destination).exists():
        raise FileExistsError(destination)
    with open(config) as handle:
        args = json.load(handle)['models']['denoiser']['args']
    model = SparseStructureFlowModel(**args)
    state = torch.load(source, map_location='cpu', weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not k.startswith('voxel_conditioner.') for k in result.missing_keys):
        raise ValueError(f'Incompatible source checkpoint: {result}')
    torch.save(model.state_dict(), destination)
    print(f'Saved {destination}; initialized {len(result.missing_keys)} condition tensors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--config', default='configs/flow/ss_flow_mc_inpaint_v4.json')
    options = parser.parse_args()
    migrate(options.source, options.config, options.destination)
