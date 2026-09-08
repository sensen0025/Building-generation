import os
import json
from typing import Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import utils3d
from .components import StandardDatasetBase
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer


class SparseStructure(StandardDatasetBase):
    """
    Sparse structure dataset

    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the voxel grid
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
    """

    def __init__(self,
        roots,
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        num_classes: int = 66,
        state_npy: str = None,
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.num_classes = num_classes
        self.value_range = (0, num_classes - 1)
        self._ss_mmap = None
        self._state_mmap = None

        super().__init__(roots)
        # 加载多类别体素 memmap (ss64.npy)
        for root in self.roots:
            p = os.path.join(root, 'ss64.npy')
            if os.path.exists(p):
                self._ss_mmap = np.load(p, mmap_mode='r')
                print(f"[SparseStructure] loaded ss64.npy shape={self._ss_mmap.shape} num_classes={num_classes}", flush=True)
                break
        # 可选: 加载 blockstate memmap (ss64_state.npy, uint16 5字段)
        if state_npy is not None and os.path.exists(state_npy):
            self._state_mmap = np.load(state_npy, mmap_mode='r')
            print(f"[SparseStructure] loaded state_npy shape={self._state_mmap.shape}", flush=True)
        
    def filter_metadata(self, metadata):
        # Minecraft 数据无 voxelized/aesthetic_score 列, 直接返回全部
        stats = {'Total': len(metadata)}
        return metadata, stats

    def get_instance(self, root, instance):
        if self._ss_mmap is not None:
            idx = int(instance)
            ss = np.array(self._ss_mmap[idx])
            pack = {'ss': torch.from_numpy(ss).long().unsqueeze(0)}
            if self._state_mmap is not None and idx < self._state_mmap.shape[0]:
                st = np.array(self._state_mmap[idx])
                pack['state'] = torch.from_numpy(st).long().unsqueeze(0)
            return pack
        # fallback 原逻辑(ply)
        position = utils3d.io.read_ply(os.path.join(root, 'voxels', f'{instance}.ply'))[0]
        coords = ((torch.tensor(position) + 0.5) * self.resolution).int().contiguous()
        ss = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        return {'ss': ss}

    @torch.no_grad()
    def visualize_sample(self, ss: Union[torch.Tensor, dict]):
        ss = ss if isinstance(ss, torch.Tensor) else ss['ss']
        
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        
        # Build each representation
        ss = ss.cuda()
        for i in range(ss.shape[0]):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(ss[i, 0], as_tuple=False)
            representation.position = coords.float() / self.resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(self.resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)
       