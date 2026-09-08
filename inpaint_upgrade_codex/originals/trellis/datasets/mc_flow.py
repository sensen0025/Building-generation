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

    def __init__(self, roots, latent_npy, image_dir, image_size=518, normalize=None):
        self.latents = np.load(latent_npy, mmap_mode='r')
        self.image_dir = image_dir
        self.image_size = image_size
        self.normalize = normalize
        self.value_range = (0, 1)

        # 可用索引: 有 latent 且有渲染图
        ids = list(range(self.latents.shape[0]))
        if image_dir:
            have = set()
            for f in glob.glob(os.path.join(image_dir, 'bld_*.png')):
                try:
                    have.add(int(os.path.basename(f)[4:-4]))
                except ValueError:
                    pass
            ids = [i for i in ids if i in have]
        self.ids = ids

        if normalize is not None:
            self.mean = torch.tensor(normalize['mean']).reshape(-1, 1, 1, 1).float()
            self.std = torch.tensor(normalize['std']).reshape(-1, 1, 1, 1).float()
        print(f'[MCFlowImageDataset] {len(self.ids)} samples (of {self.latents.shape[0]})', flush=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        z = torch.from_numpy(np.asarray(self.latents[idx]).copy()).float()
        if self.normalize is not None:
            z = (z - self.mean) / self.std
        img = Image.open(os.path.join(self.image_dir, f'bld_{idx:04d}.png')).convert('RGB')
        w, h = img.size
        s = min(w, h)
        l = (w - s) // 2
        t = (h - s) // 2
        img = img.crop((l, t, l + s, t + s)).resize(
            (self.image_size, self.image_size), Image.LANCZOS)
        cond = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return {'x_0': z, 'cond': cond}

    def collate_fn(self, batch, split_size=None):
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}

    @torch.no_grad()
    def visualize_sample(self, sample):
        """snapshot 只保存 cond 图像."""
        return sample['cond']
