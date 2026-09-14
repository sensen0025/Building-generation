"""One image preprocessing implementation for training, evaluation and inference."""
import numpy as np
from PIL import Image
MEAN=(0.485,0.456,0.406)
STD=(0.229,0.224,0.225)

def prepare_image(image,size=518):
    import torch
    if not isinstance(image,Image.Image):
        with Image.open(image) as opened: image=opened.convert('RGB')
    else: image=image.convert('RGB')
    w,h=image.size;s=min(w,h);left=(w-s)//2;top=(h-s)//2
    image=image.crop((left,top,left+s,top+s)).resize((size,size),Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255.0

def normalize_image(image):
    import torch
    if image.ndim!=4 or image.shape[1]!=3 or not image.is_floating_point():
        raise ValueError('Expected float [B,3,H,W] RGB in [0,1]')
    if not torch.isfinite(image).all() or image.min()<0 or image.max()>1:
        raise ValueError('Image outside [0,1]')
    mean=image.new_tensor(MEAN).view(1,3,1,1);std=image.new_tensor(STD).view(1,3,1,1)
    return (image-mean)/std

def encode_image(model,image):
    import torch
    with torch.no_grad():return model(normalize_image(image)).last_hidden_state
