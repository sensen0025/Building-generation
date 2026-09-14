"""Dense flow matching trainer with image conditioning (Minecraft).

Reuses FlowMatchingTrainer (dense-compatible diffuse/get_v/sample_t) with
DINOv2 image conditioning loaded offline from HF cache, plus CFG.
"""
import os
from typing import *
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin


class DenseImageFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Dense (non-sparse) image-conditioned flow matching trainer with CFG.

    Args:
        image_cond_model (str): HF model id for DINOv2 image encoder (offline).
        p_uncond (float): probability of dropping condition (CFG).
    """

    def __init__(self, *args, image_cond_model='facebook/dinov2-large',
                 inpaint_image_drop_probability=0.25, **kwargs):
        if not 0 <= inpaint_image_drop_probability <= 1:
            raise ValueError('inpaint_image_drop_probability must be in [0, 1]')
        self.inpaint_image_drop_probability = inpaint_image_drop_probability
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None  # lazy init

    def _init_image_cond_model(self):
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        from transformers import AutoModel
        print(f'[DenseImageFlow] loading image encoder {self.image_cond_model_name}...', flush=True)
        self.image_cond_model = AutoModel.from_pretrained(
            self.image_cond_model_name, local_files_only=True).cuda().eval()
        print('[DenseImageFlow] image encoder ready.', flush=True)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """image: [B,3,H,W] in [0,1] -> patch tokens [B,S,1024]."""
        if self.image_cond_model is None:
            self._init_image_cond_model()
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        from mcfix.images import normalize_image
        x = normalize_image(image)
        feats = self.image_cond_model(x).last_hidden_state  # [B, 1+S, 1024]
        return feats

    def get_cond(self, cond, **kwargs):
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        return super().get_cond(cond, **kwargs)

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        cond = self.encode_image(cond)
        neg_cond = torch.zeros_like(cond) if neg_cond is None else neg_cond
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}

    def vis_cond(self, cond, **kwargs):
        return {}

    def training_losses(self, x_0: torch.Tensor, cond=None, **kwargs) -> Tuple[Dict, Dict]:
        """
        Dense flow matching loss.

        Args:
            x_0: [N, C, res, res, res] dense latent.
            cond: [N, 3, H, W] images.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        known = {k: kwargs[k] for k in ('known_blocks', 'known_mask', 'known_states') if k in kwargs}
        cond = self.get_cond(cond)
        active = None
        if 'known_mask' in known:
            active = known['known_mask'].flatten(1).any(1)
            drop = active & (torch.rand(x_0.shape[0], device=x_0.device) < self.inpaint_image_drop_probability)
            cond = torch.where(drop[:, None, None], torch.zeros_like(cond), cond)

        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **known)
        target = self.get_v(x_0, noise, t)

        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]
        if active is not None:
            per_sample = (pred.detach() - target).square().flatten(1).mean(1)
            terms['inpaint_fraction'] = active.float().mean()
            for name, selected in [('inpaint_mse', active), ('generation_mse', ~active)]:
                if selected.any():
                    terms[name] = per_sample[selected].mean()

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        conds = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            cond = data['cond'][:batch].cuda()
            if self.image_cond_model is None:
                self._init_image_cond_model()
            mean = torch.tensor([0.485, 0.456, 0.406], device=cond.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=cond.device).view(1, 3, 1, 1)
            feats = self.image_cond_model(__import__('mcfix.images', fromlist=['normalize_image']).normalize_image(cond)).last_hidden_state
            # 用 cond 图像本身作为可视化 (samples 存 cond 图)
            conds.append(cond)
        sample_dict = {
            'cond': {'value': torch.cat(conds, dim=0), 'type': 'image'},
        }
        return sample_dict
