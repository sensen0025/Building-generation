"""Minecraft masks and voxel-space preservation shared by training and inference."""
import numpy as np
import torch


def sample_known_mask(blocks, up_axis=1, rng=None):
    """Mix foundations, side overlaps and box holes; True includes known air."""
    rng = rng if rng is not None else np.random
    if blocks.ndim != 3 or min(blocks.shape) < 2 or up_axis not in (0, 1, 2):
        raise ValueError('Expected a 3D block grid and up_axis in 0..2')
    mask = np.zeros(blocks.shape, dtype=bool)
    coords = np.argwhere(blocks != 0)
    low = coords.min(0) if len(coords) else np.zeros(3, dtype=int)
    high = coords.max(0) + 1 if len(coords) else np.array(blocks.shape)
    mode = int(rng.choice(3))
    if mode < 2:
        axis = up_axis if mode == 0 else int(rng.choice([a for a in range(3) if a != up_axis]))
        cut = int(low[axis] + rng.uniform(0.25, 0.7) * (high[axis] - low[axis]))
        cut = int(np.clip(cut, 1, blocks.shape[axis] - 1))
        region = [slice(None)] * 3
        region[axis] = slice(0, cut) if mode == 0 or rng.random() < 0.5 else slice(cut, None)
        mask[tuple(region)] = True
    else:
        mask[:] = True
        hole = []
        for axis, size in enumerate(blocks.shape):
            width = max(1, int((high[axis] - low[axis]) * rng.uniform(0.35, 0.75)))
            start = int(rng.randint(int(low[axis]), max(int(low[axis]) + 1, int(high[axis]) - width + 1)))
            hole.append(slice(start, min(size, start + width)))
        mask[tuple(hole)] = False
    return mask


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
                      state_dims=(7, 3, 6, 16, 3), state_classes=tuple(range(1, 63))):
    """cond is encoded image tokens; pass zero tokens for block-only generation."""
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
