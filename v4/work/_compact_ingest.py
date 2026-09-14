
# -*- coding: utf-8 -*-
"""Compact ss64_new arrays to only ok samples; save final sample list."""
import json
import numpy as np

WORK = "/data/redstone_v4/work"
m = json.load(open(f"{WORK}/ingest_manifest.json"))
ok = [r for r in m if r.get("ok")]
print("ok:", len(ok))
ss = np.load(f"{WORK}/ss64_new.npy", mmap_mode="r")
st = np.load(f"{WORK}/ss64_state_new.npy", mmap_mode="r")
idxs = [r["idx"] for r in ok]
out_s = np.lib.format.open_memmap(f"{WORK}/ss64_new_c.npy", mode="w+", dtype=np.uint8, shape=(len(idxs), 64, 64, 64))
out_t = np.lib.format.open_memmap(f"{WORK}/ss64_state_new_c.npy", mode="w+", dtype=np.uint16, shape=(len(idxs), 64, 64, 64))
for j, i in enumerate(idxs):
    out_s[j] = ss[i]
    out_t[j] = st[i]
    if (j+1) % 1500 == 0:
        print(j+1, flush=True)
        out_s.flush(); out_t.flush()
out_s.flush(); out_t.flush()
for j, r in enumerate(ok):
    r["nidx"] = j
json.dump(ok, open(f"{WORK}/ingest_manifest_ok.json", "w"), ensure_ascii=False)
print("saved compact arrays:", len(idxs))
