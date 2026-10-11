"""Stop run_generation.sh from starting the 20 GB Qwen3-VL service when stage 1 is skipped.

Only stage 1 (traj_generate.py, WorldNav trajectory planning) talks to the local VLM
service on port 8000. Stages 3-5 do not. Since the box has a single 96 GB H20 and stage 3
needs ~75 GB, leaving a 20.6 GB VLM process resident makes stage 3 die with
`torch.OutOfMemoryError: Tried to allocate 640.00 MiB`.
"""
import os
import shutil

RUNNER = "/root/run_generation.sh"
OLD = "start_vlm || exit 3"
NEW = ('if [[ "${SKIP_STAGES:-}" == *1* ]]; then\n'
       '  log "stage 1 skipped: VLM service not needed, not starting it"\n'
       'else\n'
       '  start_vlm || exit 3\n'
       'fi')

src = open(RUNNER).read()
if "VLM service not needed" in src:
    print("run_generation.sh: already patched")
else:
    assert OLD in src, "start_vlm call not found verbatim"
    src = src.replace(OLD, NEW, 1)
    if not os.path.exists(RUNNER + ".orig"):
        shutil.copy2(RUNNER, RUNNER + ".orig")
    open(RUNNER, "w").write(src)
    print("run_generation.sh: VLM startup is now conditional on stage 1")

r = os.popen("bash -n /root/run_generation.sh 2>&1").read()
print("BASH SYNTAX OK" if not r else "BASH SYNTAX FAIL " + r)
