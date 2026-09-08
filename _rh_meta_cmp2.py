
import json
IDX = [0,6,46,250,600,1200,1800,2200]
nm = json.load(open("/home/linux/redstone_ai/render_mc/builds_meta.json"))["meta"]
om = json.load(open("/home/linux/redstone_ai/render_mc/builds_meta.v2compressed.json"))["meta"]
for i in IDX:
    a,b = nm[i], om[i]
    def fmt(m):
        e = m.get("eye"); l = m.get("look")
        return f"n={m.get('n')} bb={m.get('bbox_min')}..{m.get('bbox_max')} spans={m.get('spans')} eye={[round(v,1) for v in e] if e else None} look={[round(v,1) for v in l] if l else None} dist={round(m.get('dist',0),1)}"
    print(f"idx{i}\n  NEW {fmt(a)}\n  OLD {fmt(b)}", flush=True)
