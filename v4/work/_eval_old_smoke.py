from mcfix.legacy import legacy_state_classes
import os, sys, json
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/home/linux/redstone_ai/TRELLIS")
import numpy as np, torch
from PIL import Image
from trellis.models import SparseStructureFlowModel, SparseStructureDecoder
from trellis.utils.mc_inpaint import sample_completion
from transformers import AutoModel

cfg = json.load(open("/data/redstone_v4/work/ss_flow_mc_v4_comb.json"))
args = cfg["models"]["denoiser"]["args"]
mean = torch.tensor(cfg["dataset"]["args"]["normalize"]["mean"]).view(1,-1,1,1,1)
std = torch.tensor(cfg["dataset"]["args"]["normalize"]["std"]).view(1,-1,1,1,1)
dev = "cuda"

denoiser = SparseStructureFlowModel(**args).to(dev)
denoiser.load_state_dict(torch.load("/data/redstone_v4/work/v4_init.pt", map_location="cpu", weights_only=True))
denoiser.eval()
decoder = SparseStructureDecoder(out_channels=163, latent_channels=8, num_res_blocks=2,
                                 num_res_blocks_middle=2, channels=[512,128,32], use_fp16=False).to(dev)
decoder.load_state_dict(torch.load("/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/ckpts/decoder_step0050000.pt", map_location="cpu", weights_only=True))
decoder.eval()
dino = AutoModel.from_pretrained("facebook/dinov2-large", local_files_only=True).to(dev).eval()

# old sample idx 100 (has image in render_output_rot)
ss = np.load("/data/redstone_v4/train/ss64_all.npy", mmap_mode="r")
st = np.load("/data/redstone_v4/train/ss64_state_all_rotfix.npy", mmap_mode="r")
gt = np.asarray(ss[100]); gtp = np.asarray(st[100])
img = Image.open("/home/linux/redstone_ai/render_output_rot/bld_0100.png").convert("RGB")
w,h = img.size; s = min(w,h)
img = img.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2)).resize((518,518), Image.LANCZOS)
x = torch.from_numpy(np.array(img)).permute(2,0,1).float().unsqueeze(0).to(dev)/255.0
im_mean = torch.tensor([0.485,0.456,0.406], device=dev).view(1,3,1,1)
im_std = torch.tensor([0.229,0.224,0.225], device=dev).view(1,3,1,1)
with torch.no_grad():
    feats = dino((x-im_mean)/im_std).last_hidden_state
print("cond shape:", feats.shape)
nz = np.argwhere(gt>0)
mask = np.zeros_like(gt, dtype=bool); cut = int(nz[:,1].min()+(nz[:,1].max()-nz[:,1].min())*0.5)
mask[:, cut:, :] = True
known = ~mask
with torch.no_grad():
    vox, fields = sample_completion(denoiser, decoder, torch.randn(1,8,16,16,16,device=dev),
        feats, torch.from_numpy(gt).long().unsqueeze(0).to(dev),
        torch.from_numpy(known).unsqueeze(0).to(dev),
        torch.from_numpy(gtp).long().unsqueeze(0).to(dev),
        mean.to(dev), std.to(dev), steps=20, state_classes=legacy_state_classes())
vox = vox[0].cpu().numpy()
inter = ((gt>0)&(vox>0)&mask).sum(); union = (((gt>0)&mask)|((vox>0)&mask)).sum()
print("image-cond completion IoU:", round(inter/max(union,1),4))
print("known preserved:", round(float(((gt>0)==(vox>0))[known].mean()),4))
print("IMAGE-COND EVAL SMOKE OK")
