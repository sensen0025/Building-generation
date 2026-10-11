"""Fix truthiness tests on torch.compile'd modules in the HY-World 2.0 pipelines.

worldstereo_wrapper.py wraps the VAE (line 400) and the text encoder (line 359) in
torch.compile, turning them into torch._dynamo OptimizedModule objects. Python's
truth testing on an object without __bool__ falls back to __len__, and
OptimizedModule.__len__ raises TypeError. So

    ... if getattr(self, "vae", None) else ...

in _pipeline_common.py:82-83 aborts stage 3 with
    TypeError: AutoencoderKLWan does not support len()

Replacing the truth test with `is not None` is both correct and compile-safe.
"""
import os
import re
import shutil
import sys

PIPE_COMMON = "/root/autodl-tmp/hyworld2/town_compare/source/hyworld2/worldgen/models/pipelines/_pipeline_common.py"
WORLDGEN = "/root/autodl-tmp/hyworld2/town_compare/source/hyworld2/worldgen"

FIXES = [
    (
        'self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4',
        'self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) is not None else 4',
    ),
    (
        'self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8',
        'self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) is not None else 8',
    ),
]


def main():
    src = open(PIPE_COMMON).read()
    changed = 0
    for old, new in FIXES:
        if new in src:
            continue
        if old in src:
            src = src.replace(old, new)
            changed += 1
    if changed:
        if not os.path.exists(PIPE_COMMON + ".orig"):
            shutil.copy2(PIPE_COMMON, PIPE_COMMON + ".orig")
        open(PIPE_COMMON, "w").write(src)
    print(f"_pipeline_common.py: {changed} line(s) fixed")

    # scan for other truthiness tests on modules that may be torch.compile'd
    pat = re.compile(
        r"if\s+(getattr\(self,\s*[\"'][a-z_]+[\"'],\s*None\)|self\.(vae|transformer|text_encoder|image_encoder|controlnet))"
        r"(?!\s+is\s+not\s+None)[^:]*:"
    )
    print("\n=== other truthiness-on-module sites (review) ===")
    hits = 0
    for root, _dirs, files in os.walk(WORLDGEN):
        if "__pycache__" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            try:
                for i, line in enumerate(open(p, errors="ignore"), 1):
                    if pat.search(line):
                        print(f"{p.replace(WORLDGEN + '/', '')}:{i}: {line.strip()[:120]}")
                        hits += 1
            except Exception:
                pass
    print(f"({hits} site(s))")


if __name__ == "__main__":
    main()
    r = os.popen(f"/root/miniconda3/bin/python -c 'import ast; ast.parse(open(\"{PIPE_COMMON}\").read())' 2>&1").read()
    print("SYNTAX OK" if not r else "SYNTAX FAIL " + r)
