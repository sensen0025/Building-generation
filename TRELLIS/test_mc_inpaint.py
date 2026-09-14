from mcfix.legacy import legacy_state_classes
import os
os.environ.setdefault('ATTN_BACKEND', 'sdpa')
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from trellis.models import SparseStructureFlowModel
from trellis.modules.voxel_conditioning import VoxelConditioner
from trellis.utils.mc_inpaint import sample_known_mask, preserve_known, sample_completion
from trellis.datasets.mc_flow import MCFlowImageDataset


def _radix_decode(packed, dims):
    """Decode a mixed-radix packed integer back into per-field integers.

    Mirrors VoxelConditioner.forward and SparseStructureVaeTrainer.decode_state:
    field_f = (packed // multiplier_f) % dim_f, multiplier accumulates dims.
    """
    fields = []
    multiplier = 1
    for dim in dims:
        fields.append((packed // multiplier) % dim)
        multiplier *= dim
    return fields


def _detect_half_cut(mask):
    """Return (True, axis) if mask is a half-space cut along one axis
    (known = a full slab on one side, unknown = full slab on the other),
    else (False, None) which implies a box-hole mask."""
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        proj = mask.all(axis=other)       # slice along axis entirely known?
        inv = (~mask).all(axis=other)     # slice along axis entirely unknown?
        if proj.any() and inv.any():
            idx = np.flatnonzero(proj)
            # True positions must be contiguous (a prefix or suffix of the axis)
            if idx.max() - idx.min() + 1 == proj.sum():
                return True, axis
    return False, None


class InpaintTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 8, 4, 4, 4)
        self.blocks = torch.ones(2, 16, 16, 16, dtype=torch.long)
        self.mask = torch.zeros_like(self.blocks, dtype=torch.bool)
        self.mask[:, :8] = True

    def conditioner(self):
        module = VoxelConditioner(8, 8, 32, 4, 2)
        torch.nn.init.normal_(module.projection.weight, std=0.1)
        return module

    def test_unknown_values_do_not_leak(self):
        module = self.conditioner()
        a = module(self.x, self.blocks, self.mask)
        dirty = self.blocks.clone()
        dirty[~self.mask] = 9999
        states = torch.zeros_like(dirty)
        states[~self.mask] = 5000
        self.assertTrue(torch.equal(a, module(self.x, dirty, self.mask, states)))

    def test_known_air_differs_from_unknown(self):
        module = self.conditioner()
        empty = torch.zeros_like(self.blocks)
        a = module(self.x, empty, self.mask)
        b = module(self.x, empty, torch.zeros_like(self.mask))
        self.assertGreater(a.abs().sum().item(), 0)
        self.assertEqual(b.abs().sum().item(), 0)

    def test_state_condition_changes_features(self):
        module = self.conditioner()
        a = module(self.x, self.blocks, self.mask)
        b = module(self.x, self.blocks, self.mask, torch.ones_like(self.blocks))
        self.assertFalse(torch.equal(a, b))

    def test_invalid_mask_rejected(self):
        with self.assertRaises(ValueError):
            self.conditioner()(self.x, self.blocks, self.mask.float())

    def test_legacy_equivalence_and_gradients(self):
        args = dict(resolution=4, in_channels=8, out_channels=8, model_channels=32,
                    cond_channels=16, num_blocks=1, num_heads=4)
        original = SparseStructureFlowModel(**args)
        torch.nn.init.normal_(original.out_layer.weight, std=0.1)
        upgraded = SparseStructureFlowModel(**args, voxel_cond_classes=8, voxel_cond_channels=8)
        result = upgraded.load_state_dict(original.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(all(k.startswith('voxel_conditioner.') for k in result.missing_keys))
        t, cond = torch.ones(2), torch.randn(2, 3, 16)
        expected = original(self.x, t, cond)
        actual = upgraded(self.x, t, cond, self.blocks, self.mask)
        self.assertTrue(torch.equal(expected, actual))
        actual.square().mean().backward()
        self.assertGreater(upgraded.voxel_conditioner.projection.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in upgraded.parameters()))
        torch.nn.init.normal_(upgraded.voxel_conditioner.projection.weight, std=0.1)
        self.assertTrue(torch.equal(expected, upgraded(self.x, t, cond)))

    def test_preserves_air_blocks_and_states(self):
        blocks = self.blocks.clone()
        blocks[:, 0] = 0
        states = torch.full_like(blocks, 321)
        pred = torch.full_like(blocks, 7)
        b, s = preserve_known(pred, pred, blocks, states, self.mask)
        self.assertTrue(torch.equal(b[self.mask], blocks[self.mask]))
        self.assertTrue(torch.equal(s[self.mask], states[self.mask]))
        self.assertTrue(torch.equal(b[~self.mask], pred[~self.mask]))

    def test_masks_have_known_and_unknown_regions(self):
        blocks = np.ones((16, 16, 16), dtype=np.uint8)
        rng = np.random.RandomState(4)
        for _ in range(100):
            mask = sample_known_mask(blocks, rng=rng)
            self.assertTrue(mask.any())
            self.assertFalse(mask.all())

    def test_dataset_ratios_and_no_target_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            np.save(p / 'z.npy', np.zeros((1, 8, 4, 4, 4), dtype=np.float32))
            np.save(p / 'b.npy', np.ones((1, 16, 16, 16), dtype=np.uint8))
            np.save(p / 's.npy', np.ones((1, 16, 16, 16), dtype=np.uint16))
            Image.new('RGB', (8, 8)).save(p / 'bld_0000.png')
            dataset = MCFlowImageDataset('', str(p / 'z.npy'), directory, image_size=8,
                                        blocks_npy=p / 'b.npy', states_npy=p / 's.npy',
                                        inpaint_probability=0.3)
            np.random.seed(42)
            count = 0
            for _ in range(1000):
                sample = dataset[0]
                mask = sample['known_mask']
                count += int(mask.any())
                self.assertEqual(sample['known_blocks'][~mask].sum().item(), 0)
                self.assertEqual(sample['known_states'][~mask].sum().item(), 0)
            self.assertTrue(250 < count < 350, count)
            dataset.inpaint_probability = 0
            self.assertFalse(dataset[0]['known_mask'].any())
            dataset.inpaint_probability = 1
            self.assertTrue(dataset[0]['known_mask'].any())

    def test_sampler_cfg_keeps_block_condition_and_final_clamp(self):
        calls = []
        class Model:
            def __call__(self, x, t, cond, **kwargs):
                calls.append(kwargs)
                return torch.zeros_like(x)
        class Decoder:
            def __call__(self, x):
                return torch.zeros(2, 163, 16, 16, 16)
        states = torch.ones_like(self.blocks)
        b, s = sample_completion(Model(), Decoder(), self.x, torch.ones(2, 3, 16),
                                 self.blocks, self.mask, states, [0] * 8, [1] * 8, steps=2, state_classes=legacy_state_classes())
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(c['known_mask'] is self.mask for c in calls))
        self.assertTrue(torch.equal(b[self.mask], self.blocks[self.mask]))
        self.assertTrue(torch.equal(s[self.mask], states[self.mask]))

    def test_state_radix_roundtrip(self):
        # Guard: if the conditioner's state_dims order / radix multipliers ever
        # drift from the VAE's canonical pack, the decode must stop round-tripping.
        module = VoxelConditioner(8, 8, 32, 4, 2)
        self.assertEqual(module.state_dims, (7, 3, 6, 16, 3))
        mults = []
        mult = 1
        for d in module.state_dims:
            mults.append(mult)
            mult *= d
        self.assertEqual(mults, [1, 7, 21, 126, 2016])

        # Canonical pack, mirrors /home/linux/redstone_ai/_voxelize.py::pack_state
        # and the VAE trainer's state_field_dims=(7,3,6,16,3).
        def pack_state(o, v, sh, hz, ac):
            return o + 7 * (v + 3 * (sh + 6 * (hz + 16 * ac)))

        for o in range(7):
            for v in range(3):
                for sh in range(6):
                    for hz in range(16):
                        for ac in range(3):
                            st = pack_state(o, v, sh, hz, ac)
                            self.assertEqual(_radix_decode(st, module.state_dims),
                                             [o, v, sh, hz, ac])

    def test_sample_known_mask_bbox_and_modes(self):
        # Non-trivial structure: occupies a sub-box, so bbox != full cube, and the
        # cut must land inside the bbox and leave structure on BOTH sides.
        blocks = np.zeros((16, 16, 16), dtype=np.uint8)
        blocks[4:12, 5:13, 3:11] = 1
        non_air = blocks != 0
        rng = np.random.RandomState(7)
        seen = {'up': 0, 'side': 0, 'hole': 0}
        for _ in range(400):
            mask = sample_known_mask(blocks, up_axis=1, rng=rng)
            self.assertTrue(mask.any())
            self.assertFalse(mask.all())
            # A real completion task: structure on both the known and unknown sides.
            self.assertTrue((mask & non_air).any())
            self.assertTrue((~mask & non_air).any())
            cut, axis = _detect_half_cut(mask)
            if cut:
                seen['up' if axis == 1 else 'side'] += 1
            else:
                seen['hole'] += 1
        # All three modes (foundation / side overlap / box hole) must appear.
        for key in ('up', 'side', 'hole'):
            self.assertGreater(seen[key], 0, key)
        # Negative / invalid input checks.
        with self.assertRaises(ValueError):
            sample_known_mask(np.zeros((16, 16)), rng=np.random.RandomState(0))
        with self.assertRaises(ValueError):
            sample_known_mask(np.zeros((16, 16, 16)), up_axis=3, rng=np.random.RandomState(0))


if __name__ == '__main__':
    unittest.main()
