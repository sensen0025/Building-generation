# Minecraft Flow Inpainting v4

## Training

New config: `configs/flow/ss_flow_mc_inpaint_v4.json`.
The old v3 config and default model constructor remain usable.

The v4 dataset samples 30% completion tasks in expectation, not exactly in each
batch. Masks mix bottom foundations, horizontal side overlaps, and box holes.
The vertical axis is array axis 1 (X,Y,Z). Known space includes air. These masks
are geometric augmentation, not terrain/building segmentation.

Inputs are aligned arrays: normalized latent [N,8,16,16,16], block class IDs and
packed states [N,64,64,64], plus images. The current config points to the existing
non-rotated v2 arrays. Replace all aligned paths and latent normalization together
when adopting a new dataset. Shape checks cannot detect reordered samples.

The model embeds block classes and five mixed-radix state fields, masks unknown
values before convolution, downsamples 4x and adds projected condition tokens to
the transformer input. The projection starts at zero. An empty mask always adds
zero, including after fine-tuning. Transformer activation checkpointing is enabled.

Image CFG dropout remains 10%. Completion examples have an additional independent
25% image dropout, so 32.5% of completion examples have no image condition overall.
Block conditions are retained in both CFG branches. Loss remains dense latent
velocity MSE; masking latent loss as if it were voxel-local would be incorrect.

Launch from the TRELLIS directory after preparing the intended dataset:

```sh
ATTN_BACKEND=sdpa /home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python train.py \
  --config configs/flow/ss_flow_mc_inpaint_v4.json \
  --output_dir outputs/ss_flow_mc_inpaint_v4 --data_dir data/mc
```

No long training job is launched by this upgrade.

## Existing Weights

Old checkpoints load strictly into the default (v3) constructor. To warm-start
v4, expand a checkpoint first; do not resume an old optimizer state:

```sh
ATTN_BACKEND=sdpa /home/linux/anaconda3/envs/torch2.4_cuda12.1/bin/python \
  migrate_inpaint_checkpoint.py OLD_DENOISER.pt NEW_DENOISER.pt
```

Then set `trainer.args.finetune_ckpt` to `{"denoiser": "NEW_DENOISER.pt"}`
in a training config, using an absolute checkpoint path. The migration rejects
unrelated missing/unexpected keys and incompatible tensor sizes, and refuses to
overwrite an existing destination. Use a checkpoint trained with the same VAE,
class map, orientation and normalization as the selected training data.

## Inference

`trellis.utils.mc_inpaint.sample_completion` accepts the model, decoder, latent
noise, encoded DINO tokens, known block IDs, boolean mask, packed states and latent
normalization. All inputs must be on the same device with matching batch sizes.
Use eval mode for both networks. Pass zero image tokens for block-only inference.
Unknown block/state values are ignored before feature extraction.

The helper samples the latent and decodes block/state logits, then copies known
blocks AND states back exactly, including known air. It returns two [B,X,Y,Z]
integer tensors. This API uses the current 128-class map and default state classes
1..62; override those arguments if the decoder vocabulary changes. States use
multipliers [1,7,21,126,2016], matching the existing VAE.

Calling the generic sampler directly forwards block conditions but does NOT apply
the final voxel preservation step; use this helper or `preserve_known` yourself.
The old `_gen_v3.py` remains a legacy ordinary-generation script.

## Verification and Limits

`python -m unittest test_mc_inpaint -v` tests leakage isolation, known air,
state conditioning, legacy equivalence, gradients, mask sampling, the 30% mixture,
CFG forwarding and final voxel preservation.

`python verify_inpaint_gpu.py` performs two disposable optimizer steps using the
real 64-cube dataset, cached DINO encoder and a real v3 checkpoint. It checks finite
losses/gradients and that gradients reach the condition encoder after its initial
zero projection is updated. It does not write a training checkpoint.

This adds local completion, not global large-building planning or automatic
terrain segmentation. Output is still 64 cubed. Exact known-voxel preservation
does not establish continuity or functional correctness at the boundary. Quality
and the best completion ratio require held-out training experiments.

Original changed source files are backed up on the server in
`/home/linux/redstone_ai/inpaint_upgrade_codex/originals.tar`.

Verified on the server: 9/9 unit tests passed. The real checkpoint gained 14 new
condition tensors. Two batch-4 GPU steps passed finite-gradient checks with peak
allocated GPU memory 3.029 GiB (not total process memory). No quality claim follows
from this smoke test; full training has not been run.
