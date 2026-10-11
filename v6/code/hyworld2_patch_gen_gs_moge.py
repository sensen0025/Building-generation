"""Point gen_gs_data.py (stage 4) at the local MoGe checkpoint.

gen_gs_data.py:98 calls MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal"), a Hub id,
but the stage runs with HF_HUB_OFFLINE=1 and moge/model/v2.py then calls hf_hub_download,
so stage 4 aborted with LocalEntryNotFoundError. traj_generate.py and video_gen.py already
use the local checkpoint at /dev/shm/hyworld2-nav-models/moge/model.pt; stage 4 simply
never got the same treatment.
"""
import os
import shutil

GEN = "/root/autodl-tmp/hyworld2/town_compare/source/hyworld2/worldgen/gen_gs_data.py"
MOGE = "/dev/shm/hyworld2-nav-models/moge/model.pt"

OLD = '    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)'
NEW = ('    # Local checkpoint first: this stage runs with HF_HUB_OFFLINE=1 and MoGe\'s\n'
       '    # from_pretrained() falls back to hf_hub_download() for a bare repo id.\n'
       f'    MOGE_ID = "{MOGE}"\n'
       '    moge_model = MoGeModel.from_pretrained(\n'
       '        MOGE_ID if os.path.exists(MOGE_ID) else "Ruicheng/moge-2-vitl-normal"\n'
       '    ).to(device)')

src = open(GEN).read()
if "MOGE_ID" in src:
    print("gen_gs_data.py: already patched")
else:
    assert OLD in src, "MoGe load line not found verbatim"
    src = src.replace(OLD, NEW, 1)
    if not os.path.exists(GEN + ".orig"):
        shutil.copy2(GEN, GEN + ".orig")
    open(GEN, "w").write(src)
    print("gen_gs_data.py: local MoGe checkpoint wired in")

r = os.popen(f"/root/miniconda3/bin/python -c 'import ast; ast.parse(open(\"{GEN}\").read())' 2>&1").read()
print("SYNTAX OK" if not r else "SYNTAX FAIL " + r)
