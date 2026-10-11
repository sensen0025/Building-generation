"""Minimal Sam3VideoVideoProcessor shim.

transformers 5.2.0 registers ("sam3_video", "Sam3VideoVideoProcessor") in
video_processing_auto.py but ships NO video_processing_sam3_video.py, so
Sam3VideoProcessor.from_pretrained() can never load its video component.
The only consumer in HY-World 2.0 is retrieval_wm.py outdoor sky removal,
which calls init_video_session -> video_processor(videos=(T,H,W,3), ...) and
reads pixel_values_videos[0] (T,C,H,W) + original_sizes[0] (H,W).

This shim implements exactly that contract by applying the real SAM3 image
processor frame by frame (same resize/normalise the tracker expects).
"""
import numpy as np
import torch
from PIL import Image

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.video_processing_utils import BaseVideoProcessor


class Sam3VideoVideoProcessor(BaseVideoProcessor):
    main_input_name = "videos"
    model_input_names = ["pixel_values_videos"]

    model_input_names = ["pixel_values_videos"]

    def __init__(self, image_processor=None, **kwargs):
        # BaseVideoProcessor.__init__ validates kwargs against VideosKwargs;
        # we carry no video-decoding config of our own.
        super().__init__()
        self.image_processor = image_processor
        self.num_frames = kwargs.pop("num_frames", None)
        self.do_sample_frames = kwargs.pop("do_sample_frames", False)

    def __call__(self, videos=None, device=None, return_tensors="pt", **kwargs):
        if videos is None:
            raise ValueError("Sam3VideoVideoProcessor requires `videos`")
        if isinstance(videos, torch.Tensor):
            arr = videos
            if arr.ndim == 4 and arr.shape[1] in (1, 3):      # (T,C,H,W)
                arr = arr.permute(0, 2, 3, 1)
            videos = arr.cpu().numpy()
        if isinstance(videos, np.ndarray):
            frames = [Image.fromarray(np.asarray(f).astype(np.uint8)) for f in videos]
        elif isinstance(videos, (list, tuple)):
            frames = [f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f).astype(np.uint8))
                      for f in videos]
        else:
            raise TypeError(f"unsupported videos type {type(videos)}")

        if self.num_frames is not None and self.do_sample_frames:
            idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
            frames = [frames[i] for i in idx]

        out = self.image_processor(frames, return_tensors=return_tensors, **kwargs)
        # image processor on a list gives (T,C,H,W) + original_sizes (T,2)
        pix = out["pixel_values"]
        sizes = out["original_sizes"]
        if pix.ndim == 3:
            pix = pix.unsqueeze(0)
        if sizes.ndim == 1:
            sizes = sizes.unsqueeze(0)
        feats = {"pixel_values_videos": pix.unsqueeze(0).to(device) if device is not None else pix.unsqueeze(0),
                 "original_sizes": sizes.unsqueeze(0)}
        return BatchFeature(data=feats, tensor_type=return_tensors)


def build_sam3_video_processor(repo_id, target_size=1008):
    """Construct a Sam3VideoProcessor without from_pretrained (no video config needed)."""
    from transformers import Sam3VideoProcessor, AutoTokenizer
    from transformers.models.sam3 import Sam3ImageProcessorFast
    ip = Sam3ImageProcessorFast.from_pretrained(repo_id, local_files_only=True)
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    except Exception:
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained(repo_id, local_files_only=True)
    vp = Sam3VideoVideoProcessor(image_processor=ip)
    return Sam3VideoProcessor(image_processor=ip, video_processor=vp, tokenizer=tok, target_size=target_size)
