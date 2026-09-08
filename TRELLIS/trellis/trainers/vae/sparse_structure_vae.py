from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer


class SparseStructureVaeTrainer(BasicTrainer):
    """
    Trainer for Sparse Structure VAE.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        loss_type (str): Loss type. 'bce' for binary cross entropy, 'l1' for L1 loss, 'dice' for Dice loss.
        lambda_kl (float): KL divergence loss weight.
    """
    
    def __init__(
        self,
        *args,
        loss_type='bce',
        lambda_kl=1e-6,
        num_classes=66,
        class_weights=None,
        state_field_dims=(7, 3, 6, 16, 3),
        lambda_state=1.0,
        state_class_list=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl
        self.num_classes = num_classes
        if isinstance(class_weights, str):
            import numpy as _np
            cw = _np.load(class_weights)
            self.class_weights = torch.from_numpy(cw).float().to(self.device)
            print(f"[Trainer] class_weights loaded: {self.class_weights[:6].tolist()}...")
        else:
            self.class_weights = class_weights
        # ---- blockstate 分层 loss 配置 ----
        self.state_field_dims = tuple(int(d) for d in state_field_dims)
        self.lambda_state = lambda_state
        self.state_num_channels = sum(self.state_field_dims)
        if state_class_list is None:
            self.state_class_list = None
        else:
            self.state_class_list = set(int(c) for c in state_class_list)
            print(f"[Trainer] state loss on {len(self.state_class_list)} classes, "
                  f"fields={self.state_field_dims}, lambda_state={lambda_state}, "
                  f"state_channels={self.state_num_channels}", flush=True)
        # uint16 打包基数: st = o + 7*(v + 3*(sh + 6*(hz + 16*ac)))
        self._field_mult = [1]
        for d in self.state_field_dims[:-1]:
            self._field_mult.append(self._field_mult[-1] * d)

    @staticmethod
    def decode_state(st: torch.Tensor, field_dims, field_mult):
        """uint16 -> 各字段 long [N,H,W,D] 列表."""
        fields = []
        for f, dim in enumerate(field_dims):
            fields.append((st // field_mult[f]) % dim)
        return fields

    def training_losses(
        self,
        ss: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            ss: The [N x 1 x H x W x D] tensor of binary sparse structure.
            kwargs: optional 'state' [N x 1 x H x W x D] uint16 labels.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        # 多类别: ss 是 [N,1,H,W,D] long 标签(0..num_classes-1)
        if self.loss_type == 'ce':
            ss_label = ss.squeeze(1).long()
            ss_input = F.one_hot(ss_label, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        else:
            ss_label = None
            ss_input = ss.float()

        z, mean, logvar = self.training_models['encoder'](ss_input, sample_posterior=True, return_raw=True)
        logits = self.training_models['decoder'](z)

        terms = edict(loss = 0.0)
        if self.loss_type == 'ce':
            ce = F.cross_entropy(logits[:, :self.num_classes], ss_label, reduction='mean', weight=self.class_weights if hasattr(self, 'class_weights') else None)
            terms["ce"] = ce
            terms["loss"] = terms["loss"] + terms["ce"]
        elif self.loss_type == 'bce':
            terms["bce"] = F.binary_cross_entropy_with_logits(logits, ss.float(), reduction='mean')
            terms["loss"] = terms["loss"] + terms["bce"]
        elif self.loss_type == 'l1':
            terms["l1"] = F.l1_loss(F.sigmoid(logits), ss.float(), reduction='mean')
            terms["loss"] = terms["loss"] + terms["l1"]
        elif self.loss_type == 'dice':
            logits = F.sigmoid(logits)
            terms["dice"] = 1 - (2 * (logits * ss.float()).sum() + 1) / (logits.sum() + ss.float().sum() + 1)
            terms["loss"] = terms["loss"] + terms["dice"]
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

        # ---- blockstate 分层 loss (masked, 条件于 GT id) ----
        state_loss = None
        if (self.loss_type == 'ce' and getattr(self, 'state_class_list', None) is not None
                and logits.shape[1] >= self.num_classes + self.state_num_channels
                and 'state' in kwargs and kwargs['state'] is not None):
            st = kwargs['state'].squeeze(1).long()  # [N,H,W,D] uint16
            # mask: GT id 属于有 state 的类
            mask = torch.zeros_like(ss_label, dtype=torch.bool)
            for c in self.state_class_list:
                mask |= (ss_label == c)
            if mask.any():
                fields = self.decode_state(st, self.state_field_dims, self._field_mult)
                ls = logits[:, self.num_classes:self.num_classes + self.state_num_channels]
                off = 0
                state_loss = 0.0
                for f, dim in enumerate(self.state_field_dims):
                    fl = ls[:, off:off + dim].permute(0, 2, 3, 4, 1)[mask]
                    ft = fields[f][mask]
                    state_loss = state_loss + F.cross_entropy(fl, ft, reduction='mean')
                    off += dim
                terms["state_ce"] = state_loss
                terms["loss"] = terms["loss"] + self.lambda_state * state_loss
        if state_loss is None:
            terms["state_ce"] = terms["loss"].detach() * 0.0  # 保持 key 存在

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
            
        return terms, {}
    
    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=1, verbose=False):
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)
    
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

        # inference
        gts = []
        recons = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            if self.loss_type == 'ce':
                ss_label = args['ss'].squeeze(1).long()
                ss_input = F.one_hot(ss_label, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
            else:
                ss_input = args['ss'].float()
            z = self.models['encoder'](ss_input, sample_posterior=False)
            logits = self.models['decoder'](z)
            logits = logits[:, :self.num_classes]  # id 部分 (state 通道不参与 argmax)
            recon = logits.argmax(1, keepdim=True).long() if self.loss_type == 'ce' else (logits > 0).long()
            gts.append(args['ss'])
            recons.append(recon)

        sample_dict = {
            'gt': {'value': torch.cat(gts, dim=0), 'type': 'sample'},
            'recon': {'value': torch.cat(recons, dim=0), 'type': 'sample'},
        }
        return sample_dict
