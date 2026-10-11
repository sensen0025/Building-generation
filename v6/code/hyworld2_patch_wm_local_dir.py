"""Pass WorldMirror a local snapshot directory instead of the Hub repo id.

retrieval_wm.apply_worldmirror() launches the reconstruction as a subprocess with
HF_HUB_OFFLINE=1, so worldrecon.pipeline._resolve_model_dir() cannot resolve
"tencent/HY-World-2.0" (its snapshot_download fails even with local_files_only, because
offline mode cannot validate the repo revision). When the first argument is a *local
directory* that already contains <subfolder>/{config.json,model.safetensors}, the
resolver returns it immediately and never touches the network.

So we locate the cached snapshot at runtime and pass it as
--pretrained_model_name_or_path. The repo id stays the fallback.
"""
import os
import shutil

RETRIEVAL = "/root/autodl-tmp/hyworld2/town_compare/source/hyworld2/worldgen/src/retrieval_wm.py"

OLD = """                wm_cmd = [
                    "torchrun", f"--nproc_per_node={self.world_size}", "-m", "worldrecon.pipeline",
"""
NEW = """                # Prefer the cached local snapshot: this subprocess runs with HF_HUB_OFFLINE=1
                # and _resolve_model_dir() cannot resolve a bare repo id offline. A local dir
                # containing HY-WorldMirror-2.0/{config.json,model.safetensors} is returned as-is.
                import glob as _glob
                _wm_model = "tencent/HY-World-2.0"
                _hf_home = os.environ.get("HF_HOME", "/root/autodl-tmp/hyworld2/cache/huggingface")
                for _snap in _glob.glob(os.path.join(
                        _hf_home, "hub", "models--tencent--HY-World-2.0", "snapshots", "*")):
                    if os.path.isdir(os.path.join(_snap, "HY-WorldMirror-2.0")):
                        _wm_model = _snap
                        break
                wm_cmd = [
                    "torchrun", f"--nproc_per_node={self.world_size}", "-m", "worldrecon.pipeline",
                    "--pretrained_model_name_or_path", _wm_model,
"""

src = open(RETRIEVAL).read()
if "--pretrained_model_name_or_path" in src:
    print("retrieval_wm.py: already patched")
else:
    assert OLD in src, "wm_cmd block not found verbatim"
    src = src.replace(OLD, NEW, 1)
    if not os.path.exists(RETRIEVAL + ".orig2"):
        shutil.copy2(RETRIEVAL, RETRIEVAL + ".orig2")
    open(RETRIEVAL, "w").write(src)
    print("retrieval_wm.py: local WorldMirror snapshot is now passed explicitly")

r = os.popen(f"/root/miniconda3/bin/python -c 'import ast; ast.parse(open(\"{RETRIEVAL}\").read())' 2>&1").read()
print("SYNTAX OK" if not r else "SYNTAX FAIL " + r)
