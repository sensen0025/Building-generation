"""Minecraft masks and voxel-space preservation shared by training and inference."""
import numpy as np
import torch


def sample_known_mask(blocks, up_axis=1, rng=None):
    from mcfix.learning_data import sample_known_mask as shared_mask
    return shared_mask(blocks, up_axis, rng)


def preserve_known(pred_blocks, pred_states, known_blocks, known_states, known_mask):
    """Apply after decoding; preserve known air and packed states exactly."""
    if known_mask.dtype != torch.bool:
        raise ValueError('known_mask must be boolean')
    if any(v.shape != known_mask.shape for v in (pred_blocks, pred_states, known_blocks, known_states)):
        raise ValueError('All voxel grids must have identical shapes')
    return (torch.where(known_mask, known_blocks, pred_blocks),
            torch.where(known_mask, known_states, pred_states))


@torch.no_grad()
def sample_completion(model, decoder, noise, cond, known_blocks, known_mask,
                      known_states, latent_mean, latent_std, steps=50,
                      cfg_strength=1.5, num_classes=128,
                      state_dims=(7, 3, 6, 16, 3), state_classes=None):
    """cond is encoded image tokens; pass zero tokens for block-only generation."""
    if state_classes is None:
        raise ValueError('Pass state_classes from the decoder training metadata; use mcfix.pipeline for full states')
    from ..pipelines.samplers.flow_euler import FlowEulerCfgSampler
    result = FlowEulerCfgSampler(1e-5).sample(
        model, noise, cond, torch.zeros_like(cond), steps=steps,
        cfg_strength=cfg_strength, verbose=False, known_blocks=known_blocks,
        known_mask=known_mask, known_states=known_states)
    mean = torch.as_tensor(latent_mean, device=noise.device).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(latent_std, device=noise.device).view(1, -1, 1, 1, 1)
    logits = decoder(result.samples * std + mean)
    if logits.shape[1] != num_classes + sum(state_dims):
        raise ValueError('Decoder channel count does not match class/state configuration')
    blocks = logits[:, :num_classes].argmax(1)
    states = torch.zeros_like(blocks)
    offset, multiplier = num_classes, 1
    for dim in state_dims:
        states += logits[:, offset:offset + dim].argmax(1) * multiplier
        offset += dim
        multiplier *= dim
    valid = torch.zeros_like(blocks, dtype=torch.bool)
    for cls in state_classes:
        valid |= blocks == cls
    states = torch.where(valid, states, 0)
    return preserve_known(blocks, states, known_blocks, known_states, known_mask)
