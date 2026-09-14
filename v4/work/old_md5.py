# -*- coding: utf-8 -*-
"""Compute md5 of old 5339 valid litematics v2: FS stores mojibake names (utf8->cp866);
map raw list names (== FS names) to repaired names for cross-matching with local dataset."""
import os, hashlib, json
BASE = "/home/linux/redstone_ai/dataset"
SRC = f"{BASE}/litematics"
OUT = "/data/redstone_v4/work/old_valid_md5.json"
raw_names = [l.strip() for l in open(f"{BASE}/valid_files_list.txt", encoding="utf-8") if l.strip()]
valid = []
nmiss = 0
for i, n in enumerate(raw_names):
    p = os.path.join(SRC, n)
    if not os.path.exists(p):
        nmiss += 1
        continue
    try:
        pretty = n.encode("cp866").decode("utf-8")
    except Exception:
        pretty = n
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    valid.append({"file": pretty, "file_fs": n, "md5": h})
    if (i + 1) % 1000 == 0:
        print(f"{i+1}/{len(raw_names)}", flush=True)
json.dump(valid, open(OUT, "w"), ensure_ascii=False)
print("saved", len(valid), "missing", nmiss, "->", OUT, flush=True)
