"""Explicit known-space conditioning; unknown voxels never enter the encoder."""
import math
import torch
from torch import nn
from .spatial import patchify


class VoxelConditioner(nn.Module):
    def __init__(self, classes, channels, model_channels, scale, patch_size,
                 state_dims=(7, 3, 6, 16, 3)):
        super().__init__()
        if scale < 1 or scale & (scale - 1):
            raise ValueError('voxel_cond_scale must be a power of two')
        self.scale, self.patch_size = scale, patch_size
        self.state_dims = tuple(state_dims)
        self.blocks = nn.Embedding(classes, channels)
        self.states = nn.ModuleList(nn.Embedding(d, channels) for d in state_dims)
        layers = [nn.Conv3d(channels + 1, channels, 3, padding=1), nn.SiLU()]
        for _ in range(int(math.log2(scale))):
            layers.extend([nn.Conv3d(channels, channels, 4, stride=2, padding=1), nn.SiLU()])
        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Linear(channels * patch_size ** 3, model_channels)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x, known_blocks=None, known_mask=None, known_states=None):
        shape = (x.shape[0], *(n * self.scale for n in x.shape[2:]))
        if (known_blocks is None) != (known_mask is None):
            raise ValueError('known_blocks and known_mask must be supplied together')
        if known_blocks is None:
            if known_states is not None:
                raise ValueError('known_states requires blocks and mask')
            known_blocks = torch.zeros(shape, device=x.device, dtype=torch.long)
            known_mask = torch.zeros(shape, device=x.device, dtype=torch.bool)
        if tuple(known_blocks.shape) != shape or tuple(known_mask.shape) != shape:
            raise ValueError(f'Block conditions must have shape {shape}')
        if known_mask.dtype != torch.bool:
            raise ValueError('known_mask must be boolean: True means known, including air')
        if known_blocks.device != x.device or known_mask.device != x.device:
            raise ValueError('Block conditions must be on the model device')
        if known_blocks.is_floating_point():
            raise ValueError('known_blocks must contain integer class IDs')
        ids = torch.where(known_mask, known_blocks.long(), 0)
        if ((ids < 0) | (ids >= self.blocks.num_embeddings)).any():
            raise ValueError('Known block ID out of range')
        features = self.blocks(ids)
        if known_states is None:
            known_states = torch.zeros_like(ids)
        if known_states.shape != ids.shape or known_states.is_floating_point():
            raise ValueError('known_states must be integer packed states matching blocks')
        if known_states.device != x.device:
            raise ValueError('Known states must be on model device')
        packed = torch.where(known_mask, known_states.long(), 0)
        if ((packed < 0) | (packed >= math.prod(self.state_dims))).any():
            raise ValueError('Known packed state out of range')
        # States use the same mixed-radix encoding as SparseStructureVaeTrainer.
        multiplier = 1
        for dim, embedding in zip(self.state_dims, self.states):
            features = features + embedding((packed // multiplier) % dim)
            multiplier *= dim
        mask = known_mask.unsqueeze(1)
        features = features.permute(0, 4, 1, 2, 3) * mask
        features = self.encoder(torch.cat([features, mask.to(features.dtype)], dim=1))
        tokens = patchify(features, self.patch_size).flatten(2).transpose(1, 2)
        # Keep an empty mask exactly equivalent to the original model, even after training.
        present = known_mask.flatten(1).any(1).view(-1, 1, 1)
        return self.projection(tokens) * present
