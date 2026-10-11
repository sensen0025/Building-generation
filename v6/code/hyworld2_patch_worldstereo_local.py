"""Point stage 3 at a self-contained local WorldStereo directory.

worldstereo_wrapper.WorldStereo.from_pretrained() calls hf_hub_download() for
{subfolder}/config.json and {subfolder}/model.safetensors whenever repo_id is a
Hub id, which needs a live HEAD request (and the HF cache snapshot on this box
has dangling blobs). When repo_id is a *local directory* it instead reads
{repo_id}/{subfolder}/config.json and model.safetensors straight off disk.

So we assemble /dev/shm/WorldStereo/<model_type>/{config.json,model.safetensors}
and let video_gen.py prefer it.
"""
import os
import shutil
import sys

VIDEO_GEN = "/root/autodl-tmp/hyworld2/town_compare/source/hyworld2/worldgen/video_gen.py"
LOCAL_DIR = "/dev/shm/WorldStereo"

OLD = '''    worldstereo = WorldStereo.from_pretrained(
        "hanshanxue/WorldStereo",'''
NEW = '''    worldstereo = WorldStereo.from_pretrained(
        WORLDSTEREO_REPO,'''

ANCHOR = 'SAM3_VIDEO_REPO_ID = "/dev/shm/sam3_video_hf" if os.path.exists("/dev/shm/sam3_video_hf/config.json") else SAM3_REPO_ID\n'
EXTRA = (
    '# Stage 3 weights: prefer a self-contained local directory so from_pretrained() never\n'
    '# touches the Hub (the HF cache snapshot here has dangling blobs and the HEAD call fails).\n'
    'WORLDSTEREO_REPO = "/dev/shm/WorldStereo" if os.path.isdir("/dev/shm/WorldStereo") else "hanshanxue/WorldStereo"\n'
)


def main():
    src = open(VIDEO_GEN).read()
    if "WORLDSTEREO_REPO" in src:
        print("video_gen.py: already patched")
        return
    assert OLD in src, "worldstereo from_pretrained call not found verbatim"
    assert ANCHOR in src, "anchor line missing"
    src = src.replace(ANCHOR, ANCHOR + EXTRA)
    src = src.replace(OLD, NEW)
    if not os.path.exists(VIDEO_GEN + ".orig2"):
        shutil.copy2(VIDEO_GEN, VIDEO_GEN + ".orig2")
    open(VIDEO_GEN, "w").write(src)
    print("video_gen.py: patched for local WorldStereo dir")


if __name__ == "__main__":
    main()
    r = os.popen(f"/root/miniconda3/bin/python -c 'import ast; ast.parse(open(\"{VIDEO_GEN}\").read())' 2>&1").read()
    print("SYNTAX OK" if not r else "SYNTAX FAIL " + r)
