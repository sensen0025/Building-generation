from . import samplers

try:
    from .trellis_image_to_3d import TrellisImageTo3DPipeline
    from .trellis_text_to_3d import TrellisTextTo3DPipeline
except Exception as e:
    print(f"[trellis.pipelines] 3D pipelines skipped: {e}")


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:

    """
    pass
