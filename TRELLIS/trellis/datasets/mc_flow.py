"""Minecraft flow dataset: dense latent + 渲染图条件."""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class MCFlowImageDataset(Dataset):
    """
    Dense latent + image condition dataset for Minecraft flow training.

    Args:
        roots (str): data dir (unused placeholder).
        latent_npy (str): path to [N,8,16,16,16] latent array.
        image_dir (str): directory containing bld_XXXX.png renders.
        image_size (int): resize target for condition images.
        normalize (dict): {'mean': [...], 'std': [...]} applied to latents.
    """

    def __init__(self, roots, latent_npy, image_dir, image_size=518, normalize=None,
                 blocks_npy=None, states_npy=None, inpaint_probability=0.0,
                 up_axis=1, indices_path=None):
        self.latents = np.load(latent_npy, mmap_mode='r')
        self.image_dir = image_dir
        self.image_size = image_size
        self.normalize = normalize
        self.value_range = (0, 1)
        if not 0 <= inpaint_probability <= 1:
            raise ValueError('inpaint_probability must be in [0, 1]')
        if up_axis not in (0, 1, 2):
            raise ValueError('up_axis must be 0, 1 or 2')
        self.inpaint_probability = inpaint_probability
        self.up_axis = up_axis
        self.blocks = np.load(blocks_npy, mmap_mode='r') if blocks_npy else None
        self.states = np.load(states_npy, mmap_mode='r') if states_npy else None
        if inpaint_probability > 0 and (self.blocks is None or self.states is None):
            raise ValueError('Inpainting requires aligned blocks_npy and states_npy')
        if (self.blocks is None) != (self.states is None):
            raise ValueError('Supply both blocks_npy and states_npy')
        if self.blocks is not None:
            if self.blocks.ndim != 4 or self.blocks.shape != self.states.shape or len(self.blocks) != len(self.latents):
                raise ValueError('Block/state arrays must be [N,X,Y,Z], aligned with latents')

        # 可用索引: 有 latent 且有渲染图
        if not image_dir or not os.path.isdir(image_dir):
            raise ValueError('An existing image directory is required')
        self.image_paths = {}
        ids = list(range(self.latents.shape[0]))
        if image_dir:
            have = set()
            for f in glob.glob(os.path.join(image_dir, 'bld_*.png')):
                try:
                    image_id = int(os.path.basename(f)[4:-4])
                    if not os.path.isfile(f):
                        continue
                    if image_id in self.image_paths and os.path.realpath(self.image_paths[image_id]) != os.path.realpath(f):
                        raise RuntimeError(f'Ambiguous images for sample {image_id}')
                    self.image_paths[image_id] = f
                    have.add(image_id)
                except ValueError:
                    pass
            ids = [i for i in ids if i in have]
        if indices_path is not None:
            import json
            selected = json.load(open(indices_path))
            if not isinstance(selected, list) or len(set(selected)) != len(selected) or any(type(v) is not int or not 0 <= v < len(self.latents) for v in selected):
                raise ValueError('Invalid split indices')
            allowed = set(selected)
            ids = [i for i in ids if i in allowed]
        ids = [i for i in ids if i in self.image_paths]
        if not ids:
            raise ValueError('No valid images in requested dataset split')
        self.ids = ids

        if normalize is not None:
            self.mean = torch.tensor(normalize['mean']).reshape(-1, 1, 1, 1).float()
            self.std = torch.tensor(normalize['std']).reshape(-1, 1, 1, 1).float()
            if self.mean.shape[0] != self.latents.shape[1] or self.std.shape != self.mean.shape or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
                raise ValueError('Invalid latent normalization')
        print(f'[MCFlowImageDataset] {len(self.ids)} samples (of {self.latents.shape[0]})', flush=True)

    def __deepcopy__(self, memo):
        """Shallow copy: mmap arrays must not be deep-copied (would read GBs into RAM)."""
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, (np.ndarray, np.memmap)):
                setattr(result, k, v)
            elif k in ("ids",):
                setattr(result, k, list(v))
            else:
                setattr(result, k, copy.copy(v))
        return result

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        z = torch.from_numpy(np.asarray(self.latents[idx]).copy()).float()
        if self.normalize is not None:
            z = (z - self.mean) / self.std
        from mcfix.images import prepare_image
        cond = prepare_image(self.image_paths[idx], self.image_size)
        sample = {'x_0': z, 'cond': cond}
        if self.blocks is not None:
            from ..utils.mc_inpaint import sample_known_mask
            blocks = np.asarray(self.blocks[idx])
            active = np.random.random() < self.inpaint_probability
            mask = sample_known_mask(blocks, self.up_axis) if active else np.zeros(blocks.shape, dtype=bool)
            sample.update(
                known_blocks=torch.from_numpy(np.where(mask, blocks, 0).astype(np.int64)),
                known_states=torch.from_numpy(np.where(mask, self.states[idx], 0).astype(np.int64)),
                known_mask=torch.from_numpy(mask),
            )
        return sample

    def collate_fn(self, batch, split_size=None):
        def stack(items):
            return {k: torch.stack([b[k] for b in items], dim=0) for k in items[0]}
        if split_size is not None:
            return [stack(batch[i:i + split_size]) for i in range(0, len(batch), split_size)]
        return stack(batch)

    @torch.no_grad()
    def visualize_sample(self, sample):
        """snapshot 只保存 cond 图像."""
        return sample['cond']
