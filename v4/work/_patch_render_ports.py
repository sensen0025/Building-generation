p = "/home/linux/schem_render/render_new_batch.mjs"
src = open(p).read()
old = """const LIT_PORT = 8942;
const VITE_PORT = 8943;"""
new = """const LIT_PORT = parseInt(process.argv[5] || "8942", 10);
const VITE_PORT = parseInt(process.argv[6] || "8943", 10);"""
assert old in src, "anchor not found"
src = src.replace(old, new, 1)
open(p, "w").write(src)
print("patched ports")
