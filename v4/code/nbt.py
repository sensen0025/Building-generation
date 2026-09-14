"""Minimal fast NBT (Named Binary Tag) reader supporting gzipped and raw data.

Returns plain Python objects: dict for TAG_Compound, list for TAG_List,
int/float/str/bytes/np.ndarray.  Long/byte arrays become numpy arrays which
decode.py consumes without copying.
"""
import gzip, io, struct
import numpy as np

_TAG_END = 0


def skip_payload(f, tag):
    """Skip a payload of `tag` without materializing it (except small headers)."""
    if tag == 1:
        f.seek(1, 1)
    elif tag == 2:
        f.seek(2, 1)
    elif tag in (3, 5):
        f.seek(4, 1)
    elif tag in (4, 6):
        f.seek(8, 1)
    elif tag == 7:
        n = _read_i32(f)
        f.seek(n, 1)
    elif tag == 8:
        _read_string(f)
    elif tag == 9:
        et = f.read(1)[0]
        n = _read_i32(f)
        if et in (1, 2, 3, 4, 5, 6):
            f.seek(n * {1: 1, 2: 2, 3: 4, 4: 8, 5: 4, 6: 8}[et], 1)
        elif et == _TAG_END:
            pass
        else:
            for _ in range(n):
                skip_payload(f, et)
    elif tag == 10:
        while True:
            t = f.read(1)
            if not t:
                raise NBTParseError("truncated compound in skip")
            t = t[0]
            if t == _TAG_END:
                return
            _read_string(f)
            skip_payload(f, t)
    elif tag == 11:
        n = _read_i32(f)
        f.seek(4 * n, 1)
    elif tag == 12:
        n = _read_i32(f)
        f.seek(8 * n, 1)
    else:
        raise NBTParseError("skip: unknown tag %d" % tag)


def peek_litematic_dims(data):
    """Stream-scan a litematic root for each region's Size without loading arrays.

    Returns {region_name: (w,h,l)} with abs values. Raises on malformed input.
    """
    f = io.BytesIO(data)
    t = f.read(1)
    if not t or t[0] != 10:
        raise NBTParseError("root not compound")
    _read_string(f)  # root name
    out = {}

    def scan_region(rname, f):
        # scan region compound for Size, consuming to its END tag
        while True:
            t3 = f.read(1)
            if not t3 or t3[0] == _TAG_END:
                return
            nm3 = _read_string(f)
            if t3[0] == 10 and nm3 == "Size":
                d = {}
                while True:
                    t4 = f.read(1)
                    if not t4 or t4[0] == _TAG_END:
                        break
                    nm4 = _read_string(f)
                    if t4[0] == 3:
                        d[nm4] = _read_i32(f)
                    else:
                        skip_payload(f, t4[0])
                out[rname] = (abs(d.get("x", 0)), abs(d.get("y", 0)), abs(d.get("z", 0)))
            else:
                skip_payload(f, t3[0])

    while True:
        t = f.read(1)
        if not t or t[0] == _TAG_END:
            break
        name = _read_string(f)
        if t[0] != 10:
            skip_payload(f, t[0])
            continue
        if name == "Regions":
            while True:
                t2 = f.read(1)
                if not t2 or t2[0] == _TAG_END:
                    break
                rname = _read_string(f)
                if t2[0] != 10:
                    skip_payload(f, t2[0])
                    continue
                scan_region(rname, f)
        else:
            skip_payload(f, t[0])
    return out


class NBTParseError(Exception):
    pass


def _read_u16(f):
    d = f.read(2)
    if len(d) < 2:
        raise NBTParseError("truncated u16")
    return struct.unpack(">H", d)[0]


def _read_i32(f):
    d = f.read(4)
    if len(d) < 4:
        raise NBTParseError("truncated i32")
    return struct.unpack(_END + "i", d)[0]


def _read_i64(f):
    d = f.read(8)
    if len(d) < 8:
        raise NBTParseError("truncated i64")
    return struct.unpack(_END + "q", d)[0]


def _read_f32(f):
    return struct.unpack(_END + "f", f.read(4))[0]


def _read_f64(f):
    return struct.unpack(_END + "d", f.read(8))[0]


def _read_string(f):
    ln = struct.unpack(_END + "H", f.read(2))[0]
    if ln > 1 << 22:
        raise NBTParseError("string too long")
    b = f.read(ln)
    if len(b) < ln:
        raise NBTParseError("truncated string")
    return b.decode("utf-8", "replace")


def _read_string_le(f):
    ln = struct.unpack("<H", f.read(2))[0]
    if ln > 1 << 22:
        raise NBTParseError("string too long")
    b = f.read(ln)
    if len(b) < ln:
        raise NBTParseError("truncated string")
    return b.decode("utf-8", "replace")


def _read_payload(f, tag):
    if tag == 1:
        b = f.read(1)
        if not b:
            raise NBTParseError("truncated byte")
        return b[0] if b[0] < 128 else b[0] - 256
    if tag == 2:
        d = f.read(2)
        if len(d) < 2:
            raise NBTParseError("truncated short")
        return struct.unpack(_END + "h", d)[0]
    if tag == 3:
        return _read_i32(f)
    if tag == 4:
        return _read_i64(f)
    if tag == 5:
        return _read_f32(f)
    if tag == 6:
        return _read_f64(f)
    if tag == 7:
        n = _read_i32(f)
        if n < 0 or n > (1 << 30):
            raise NBTParseError("bad byte array len %d" % n)
        b = f.read(n)
        if len(b) < n:
            raise NBTParseError("truncated byte array")
        return np.frombuffer(b, dtype=np.uint8)
    if tag == 8:
        return _read_string(f)
    if tag == 9:
        et = f.read(1)
        if not et:
            raise NBTParseError("truncated list type")
        et = et[0]
        n = _read_i32(f)
        if n < 0 or n > (1 << 26):
            raise NBTParseError("bad list len %d" % n)
        if et == _TAG_END:
            return []
        if et == 7:  # list of byte arrays: read length once per element
            return [_read_payload(f, 7) for _ in range(n)]
        return [_read_payload(f, et) for _ in range(n)]
    if tag == 10:
        out = {}
        while True:
            t = f.read(1)
            if not t:
                raise NBTParseError("truncated compound")
            t = t[0]
            if t == _TAG_END:
                return out
            name = _read_string(f)
            out[name] = _read_payload(f, t)
    if tag == 11:
        n = _read_i32(f)
        if n < 0 or n > (1 << 29):
            raise NBTParseError("bad int array len %d" % n)
        b = f.read(4 * n)
        if len(b) < 4 * n:
            raise NBTParseError("truncated int array")
        return np.frombuffer(b, dtype=">i4").astype(np.int64)
    if tag == 12:
        n = _read_i32(f)
        if n < 0 or n > (1 << 29):
            raise NBTParseError("bad long array len %d" % n)
        b = f.read(8 * n)
        if len(b) < 8 * n:
            raise NBTParseError("truncated long array")
        return np.frombuffer(b, dtype=">i8")
    raise NBTParseError("unknown tag %d" % tag)


def parse_bytes(data, little_endian=False):
    """Parse raw (uncompressed) NBT bytes -> (root_name, root_payload)."""
    global _END
    f = io.BytesIO(data)
    t = f.read(1)
    if not t:
        raise NBTParseError("empty")
    if t[0] != 10:
        raise NBTParseError("root is not a compound (tag %d)" % t[0])
    name = _read_string_le(f) if little_endian else _read_string(f)
    _END = "<" if little_endian else ">"
    try:
        return name, _read_payload(f, 10)
    finally:
        _END = ">"


_END = ">"


def parse(data):
    """Accept gzip/zlib/raw NBT. Returns (root_name, payload)."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    elif data[:1] in (b"\x78", b"\x08", b"\x01", b"\x9c", b"\xda") :
        # raw zlib streams start with CMF byte; 0x78 common.  Try zlib.
        try:
            import zlib
            data = zlib.decompress(data)
        except Exception:
            pass
    return parse_bytes(data)


def probe_format(path):
    """Read a schematic-ish file and classify its container format by NBT shape.

    Returns (fmt, payload, extra) where fmt in
      'litematic' | 'sponge2' | 'sponge3' | 'legacy' | 'structure' | 'unknown'
    extra is a dict of useful metadata (dataversion, metadata...).
    """
    raw = open(path, "rb").read()
    data = raw
    if data[:2] == b"\x1f\x8b":
        import gzip as _gz
        data = _gz.decompress(data)
    name, root = parse_bytes(data)
    _known = ("Regions", "Schematic", "Version", "blocks", "Width", "size",
              "blockentities", "MinecraftDataVersion", "format_version")
    if not any(k in root for k in _known) and data[:3] == b"\x0a\x00\x00":
        # ambiguous: retry little-endian (bedrock .mcstructure)
        try:
            name2, root2 = parse_bytes(data, little_endian=True)
            if "size" in root2 and "structure" in root2:
                return "mcstructure", root2, {"data_version": root2.get("format_version")}
        except NBTParseError:
            pass
    if not isinstance(root, dict):
        return "unknown", root, {}
    if "Regions" in root and "MinecraftDataVersion" in root:
        return "litematic", root, {"data_version": root.get("MinecraftDataVersion"),
                                   "meta": root.get("Metadata", {}),
                                   "version": root.get("Version")}
    if "Schematic" in root and isinstance(root["Schematic"], dict):
        s = root["Schematic"]
        if "Version" in s and "BlockData" in s:
            return ("sponge3" if s.get("Version", 0) >= 3 else "sponge2"), s, {
                "data_version": s.get("DataVersion")}
        if "Blocks" in s and isinstance(s["Blocks"], dict) and "Palette" in s["Blocks"]:
            return "sponge3", s, {"data_version": s.get("DataVersion")}
        if "Blocks" in s and ("Materials" in s or "AddBlocks" in s or "BlockIDs" in s):
            return "legacy", s, {"data_version": s.get("DataVersion")}
        if "Width" in s and "Blocks" in s:
            return "legacy", s, {"data_version": s.get("DataVersion")}
        if "Version" in s and "Blocks" in s and "BlockData" not in s and "Palette" in s:
            # sponge v3 keeps Blocks compound; check
            return "sponge3", s, {"data_version": s.get("DataVersion")}
        return "unknown", root, {}
    if "Version" in root and "BlockData" in root:
        return ("sponge3" if root.get("Version", 0) >= 3 else "sponge2"), root, {}
    if "blocks" in root and "palette" in root and "size" in root:
        return "structure", root, {"data_version": root.get("DataVersion")}
    if "Width" in root and "Length" in root and "Blocks" in root:
        return "legacy", root, {}
    if "blockentities" in root or ("blocks" in root and "palette" in root):
        return "structure", root, {}
    return "unknown", root, {"keys": list(root.keys())[:24]}
