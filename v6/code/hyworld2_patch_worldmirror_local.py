"""Let WorldMirror resolve its weights from the local HF cache when the Hub is unreachable.

worldrecon/pipeline.py::_resolve_model_dir() takes three steps: join(path, subfolder),
treat path as a local dir, else snapshot_download() from the Hub. Stage 3's WorldMirror
subprocess runs without proxy env, so step 3 raises and the whole reconstruction aborts,
even though models--tencent--HY-World-2.0 is fully cached on disk.

Fix: keep the download attempt, but on failure retry with local_files_only=True, which
resolves the cached snapshot. Works offline when cached and still downloads when online.
"""
import os
import shutil

PIPE = "/root/autodl-tmp/hyworld2/town_compare/source/hyworld2/worldrecon/pipeline.py"

OLD = """    repo_root = snapshot_download(
        repo_id=model_path,
        allow_patterns=[f"{subfolder}/*"],
    )
"""
NEW = """    try:
        repo_root = snapshot_download(
            repo_id=model_path,
            allow_patterns=[f"{subfolder}/*"],
        )
    except Exception as _hub_error:
        print(f"[Init] Hub download failed ({type(_hub_error).__name__}: {_hub_error}); "
              f"retrying from the local HF cache")
        repo_root = snapshot_download(
            repo_id=model_path,
            allow_patterns=[f"{subfolder}/*"],
            local_files_only=True,
        )
"""

src = open(PIPE).read()
if "retrying from the local HF cache" in src:
    print("pipeline.py: already patched")
else:
    assert OLD in src, "snapshot_download block not found verbatim"
    src = src.replace(OLD, NEW, 1)
    if not os.path.exists(PIPE + ".orig"):
        shutil.copy2(PIPE, PIPE + ".orig")
    open(PIPE, "w").write(src)
    print("pipeline.py: local-cache fallback added")

r = os.popen(f"/root/miniconda3/bin/python -c 'import ast; ast.parse(open(\"{PIPE}\").read())' 2>&1").read()
print("SYNTAX OK" if not r else "SYNTAX FAIL " + r)
