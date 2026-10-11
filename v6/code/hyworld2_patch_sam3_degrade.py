"""Make SAM3 video sky-masking optional in HY-World 2.0 stage 3.

Why: transformers 5.2.0 registers ("sam3_video", "Sam3VideoVideoProcessor") in
models/auto/video_processing_auto.py but ships no video_processing_sam3_video.py,
and facebook/sam3 has no video_preprocessor_config.json. So
Sam3VideoProcessor.from_pretrained() can never succeed. SAM3 video is used for
exactly one thing in this pipeline: dropping sky pixels from the mono-depth mask
for outdoor scenes. Failing that step only leaves sky points in the cloud, which
we filter downstream, so it must not abort the run.
"""
import os
import shutil
import sys

ROOT = "/root/autodl-tmp/hyworld2/town_compare/source/hyworld2/worldgen"
VIDEO_GEN = os.path.join(ROOT, "video_gen.py")
RETRIEVAL = os.path.join(ROOT, "src", "retrieval_wm.py")

LOAD_OLD = """    sam3_model = Sam3VideoModel.from_pretrained(SAM3_REPO_ID).to(device, dtype=torch.bfloat16)
    sam3_processor = Sam3VideoProcessor.from_pretrained(SAM3_REPO_ID)
"""
LOAD_NEW = """    # SAM3 video tracker. transformers 5.2.0 registers a Sam3VideoVideoProcessor that
    # does not exist and facebook/sam3 ships no video_preprocessor_config.json, so the stock
    # Sam3VideoProcessor.from_pretrained() can never succeed. Use the local shim processor and
    # degrade gracefully (sky masking is skipped in retrieval_wm) if the model will not load.
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
        from sam3_video_shim import build_sam3_video_processor
        sam3_model = Sam3VideoModel.from_pretrained(SAM3_VIDEO_REPO_ID, local_files_only=True).to(device, dtype=torch.bfloat16)
        sam3_processor = build_sam3_video_processor(SAM3_VIDEO_REPO_ID)
        rank0_log("SAM3 video tracker loaded (shim video processor).")
    except Exception as _sam3_e:
        rank0_log(f"[WARN] SAM3 video tracker unavailable, sky masking will be skipped: {type(_sam3_e).__name__}: {_sam3_e}")
        sam3_model = None
        sam3_processor = None
"""

FALLBACK_OLD = """        if sam3_model is None or sam3_processor is None:
            from transformers import Sam3VideoModel, Sam3VideoProcessor
            self.sam3_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
            self.sam3_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
"""
FALLBACK_NEW = """        if sam3_model is None or sam3_processor is None:
            try:
                from transformers import Sam3VideoModel, Sam3VideoProcessor
                self.sam3_model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
                self.sam3_processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
            except Exception as _sam3_e:
                rank0_log(f"[WARN] SAM3 video tracker unavailable ({type(_sam3_e).__name__}: {_sam3_e}); sky masking disabled.")
                self.sam3_model = None
                self.sam3_processor = None
"""


def backup(path):
    if not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")


def patch_video_gen():
    src = open(VIDEO_GEN).read()
    if "sam3_video_shim" in src:
        print("video_gen.py: already patched")
        return
    assert LOAD_OLD in src, "video_gen.py: SAM3 load lines not found verbatim"
    src = src.replace(LOAD_OLD, LOAD_NEW)
    anchor = 'SAM3_REPO_ID = "/root/autodl-tmp/hyworld2/town_compare/sam3_hf"\n'
    assert anchor in src, "video_gen.py: SAM3_REPO_ID anchor missing"
    src = src.replace(
        anchor,
        anchor
        + 'SAM3_VIDEO_REPO_ID = "/dev/shm/sam3_video_hf" if os.path.exists("/dev/shm/sam3_video_hf/config.json") else SAM3_REPO_ID\n',
    )
    backup(VIDEO_GEN)
    open(VIDEO_GEN, "w").write(src)
    print("video_gen.py: patched")


def patch_retrieval():
    lines = open(RETRIEVAL).read().split("\n")
    src = "\n".join(lines)
    if "[WARN] SAM3 sky masking failed" in src:
        print("retrieval_wm.py: already patched")
        return
    assert FALLBACK_OLD in src, "retrieval_wm.py: fallback block not found verbatim"
    src = src.replace(FALLBACK_OLD, FALLBACK_NEW)

    lines = src.split("\n")
    # locate the sky-masking if-statement and its terminating comment
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == '# Use SAM3 to remove the sky mask.':
            for j in range(i + 1, min(i + 6, len(lines))):
                if lines[j].strip().startswith('if self.meta_info["scene_type"] == "outdoor"'):
                    start = j
                    break
            break
    assert start is not None, "retrieval_wm.py: sky-mask if-statement not found"
    end = None
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "# Initialize the cache for the current video.":
            end = j
            break
    assert end is not None, "retrieval_wm.py: end of sky-mask block not found"

    head = lines[start]
    body = lines[start + 1:end]
    while body and body[-1].strip() == "":
        body.pop()
    indent = len(head) - len(head.lstrip())
    assert indent == 12, f"unexpected indent {indent}"
    new_head = (
        head.replace(
            'if self.meta_info["scene_type"] == "outdoor":',
            'if (self.meta_info["scene_type"] == "outdoor"\n'
            "                    and self.sam3_model is not None\n"
            "                    and self.sam3_processor is not None):",
        )
    )
    new_block = [new_head, "                try:"]
    new_block += ["    " + b if b.strip() else b for b in body]
    new_block += [
        "                except Exception as _sky_e:",
        '                    rank0_log(f"[WARN] SAM3 sky masking failed ({type(_sky_e).__name__}: {_sky_e}); continuing without it.")',
        "",
    ]
    lines = lines[:start] + new_block + lines[end:]
    backup(RETRIEVAL)
    open(RETRIEVAL, "w").write("\n".join(lines))
    print(f"retrieval_wm.py: patched (sky block re-indented, {len(body)} body lines wrapped)")


if __name__ == "__main__":
    patch_video_gen()
    patch_retrieval()
    for p in (VIDEO_GEN, RETRIEVAL):
        r = os.popen(f"/root/miniconda3/bin/python -c 'import ast; ast.parse(open(\"{p}\").read())' 2>&1").read()
        print(("SYNTAX OK  " if not r else "SYNTAX FAIL ") + p + (" " + r if r else ""))
