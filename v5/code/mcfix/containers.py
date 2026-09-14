"""Strict schematic readers. No palette truncation, no implicit air recovery.

Coordinates: returned vol is [y,z,x], origin_xyz is its global minimum.
Unsupported containers are quarantined by the builder with original bytes kept.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import nbtlib
from .codec import parse_state, state_string, ASSETS

MAX_VOLUME = 128 * 1024 * 1024
AIR = {'minecraft:air', 'minecraft:cave_air', 'minecraft:void_air'}


class DecodeError(ValueError):
    pass


def shape(w, h, l):
    w, h, l = int(w), int(h), int(l)
    if min(w, h, l) <= 0 or w * h * l > MAX_VOLUME:
        raise DecodeError(f'Invalid or resource-limited dimensions {(w,h,l)}')
    return w, h, l


def unpack_longs(longs, count, bits):
    if not 1 <= bits <= 32:
        raise DecodeError(f'Unsupported bit width {bits}')
    # Convert endian numerically, not by a platform-dependent byte view.
    words = np.asarray(longs).astype(np.int64).view(np.uint64)
    if len(words) < (count * bits + 63) // 64:
        raise DecodeError('Truncated packed state array')
    out = np.empty(count, dtype=np.uint32)
    mask = np.uint64((1 << bits) - 1)
    for start in range(0, count, 262144):
        end = min(start + 262144, count)
        bit = np.arange(start, end, dtype=np.uint64) * bits
        wi, offset = bit // 64, bit % 64
        value = words[wi] >> offset
        cross = offset + bits > 64
        value[cross] |= words[wi[cross] + 1] << (np.uint64(64) - offset[cross])
        out[start:end] = (value & mask).astype(np.uint32)
    return out


def pack_longs(ids, bits):
    values = np.asarray(ids, dtype=np.uint64).ravel()
    if np.any(values >= (1 << bits)):
        raise ValueError('Value exceeds bit width')
    out = np.zeros((len(values) * bits + 63) // 64, dtype=np.uint64)
    for start in range(0, len(values), 262144):
        end = min(start + 262144, len(values))
        bit = np.arange(start, end, dtype=np.uint64) * bits
        wi, offset = bit // 64, bit % 64
        val = values[start:end]
        np.bitwise_or.at(out, wi, val << offset)
        cross = offset + bits > 64
        np.bitwise_or.at(out, wi[cross] + 1, val[cross] >> (np.uint64(64) - offset[cross]))
    return out.view(np.int64)


def varint_decode(data):
    a = np.asarray(data, dtype=np.uint8).ravel() if not isinstance(data, bytes) else np.frombuffer(data, dtype=np.uint8)
    if not a.size:
        return np.zeros(0, np.int64)
    ends = np.flatnonzero(a < 128)
    if not len(ends) or ends[-1] != len(a) - 1:
        raise DecodeError('Truncated varint')
    starts = np.r_[0, ends[:-1] + 1]
    widths = ends - starts + 1
    if widths.max() > 5:
        raise DecodeError('Varint longer than 32 bits')
    values = np.zeros(len(ends), np.int64)
    for shift in range(int(widths.max())):
        valid = widths > shift
        values[valid] |= (a[starts[valid] + shift].astype(np.int64) & 127) << (7 * shift)
    if np.any(values > 0x7fffffff):
        raise DecodeError('Negative/overflow palette index')
    return values


def _state(entry):
    return state_string(str(entry['Name']), {str(k): str(v) for k, v in entry.get('Properties', {}).items()})


def _pack(ids, names, dims):
    w, h, l = shape(*dims)
    ids = np.asarray(ids, dtype=np.int64).ravel()
    if len(ids) != w * h * l or np.any(ids < 0) or np.any(ids >= len(names)):
        raise DecodeError('Palette index range or volume mismatch')
    names = [state_string(n) if n is not None else None for n in names]
    used = np.unique(ids)
    if any(names[int(i)] is None for i in used):
        raise DecodeError('Referenced missing palette entry')
    # Reserve literal air at zero. Preserve cave/void air separately.
    palette = ['minecraft:air']
    lookup = {'minecraft:air': 0}
    mapping = np.zeros(len(names), dtype=np.uint32)
    for old in used:
        name = names[int(old)]
        if name not in lookup:
            lookup[name] = len(palette)
            palette.append(name)
        mapping[old] = lookup[name]
    dtype = np.uint8 if len(palette) <= 256 else np.uint16 if len(palette) <= 65536 else np.uint32
    return palette, mapping[ids].astype(dtype).reshape(h, l, w)


def non_air(vol, names):
    lut = np.array([parse_state(n)[0] not in AIR for n in names], dtype=bool)
    return lut[vol]


def _vec(x):
    try:
        values=[x[k] for k in ('x','y','z')] if isinstance(x,dict) else list(x)
        if len(values)!=3:raise ValueError('expected three coordinates')
        result=tuple(int(v) for v in values)
        if any(float(v)!=n for v,n in zip(values,result)):raise ValueError('noninteger coordinate')
        return result
    except (TypeError,ValueError,KeyError,OverflowError) as e:
        raise DecodeError('Invalid 3D integer coordinate vector') from e


def _palette(pal):
    if isinstance(pal, dict):
        if not pal:
            raise DecodeError('Empty palette')
        ids = [int(v) for v in pal.values()]
        if min(ids) < 0 or max(ids) > MAX_VOLUME or len(set(ids)) != len(ids):
            raise DecodeError('Invalid/duplicate palette indices')
        names = [None] * (max(ids) + 1)
        for k, v in pal.items():
            names[int(v)] = str(k)
        return names
    return [_state(p) for p in pal]


def _litematic(root):
    pieces = []
    for key, r in root['Regions'].items():
        size = _vec(r['Size'])
        dims = shape(*(abs(v) for v in size))
        pos = _vec(r['Position'])
        origin = tuple(p + min(0, s + 1) for p, s in zip(pos, size))
        names = [_state(p) for p in r['BlockStatePalette']]
        bits = max(2, (len(names) - 1).bit_length())
        ids = unpack_longs(r['BlockStates'], int(np.prod(dims)), bits)
        pal, vol = _pack(ids, names, dims)
        pieces.append((str(key), origin, dims, pal, vol))
    if not pieces:
        raise DecodeError('No regions')
    lower = np.min([p[1] for p in pieces], axis=0)
    upper = np.max([np.array(p[1]) + p[2] for p in pieces], axis=0)
    dims = shape(*(upper - lower))
    merged = np.zeros((dims[1], dims[2], dims[0]), dtype=np.uint32)
    names, lookup = ['minecraft:air'], {'minecraft:air': 0}
    for _, origin, rdims, palette, vol in pieces:
        lut = []
        for name in palette:
            if name not in lookup:
                lookup[name] = len(names)
                names.append(name)
            lut.append(lookup[name])
        mapped = np.array(lut, np.uint32)[vol]
        x, y, z = np.array(origin) - lower
        w, h, l = rdims
        target = merged[y:y+h, z:z+l, x:x+w]
        active = non_air(vol, palette)
        target_solid = np.array([parse_state(n)[0] not in AIR for n in names], dtype=bool)[target]
        conflict = active & target_solid & (target != mapped)
        if conflict.any():
            raise DecodeError(f'Conflicting non-air overlapping regions: {int(conflict.sum())} cells; original retained')
        target[active] = mapped[active]
        # Preserve exceptional air if no occupied region covers it.
        exceptional = (~active) & (mapped != 0) & (target == 0)
        target[exceptional] = mapped[exceptional]
    names, merged = _pack(merged, names, dims)
    return dims, names, merged, lower.tolist(), [{'name': p[0], 'origin_xyz': p[1], 'dims_xyz': p[2]} for p in pieces]


def _sponge(root):
    dims = shape(*(int(root[k]) & 0xffff for k in ('Width', 'Height', 'Length')))
    version = int(root.get('Version', 2))
    if version not in (1,2,3):raise DecodeError(f'Unsupported Sponge version {version}')
    block = root['Blocks'] if version == 3 else root
    names = _palette(block['Palette'])
    key = 'Data' if version == 3 else 'BlockData'
    if key in block:
        ids = varint_decode(block[key])
    elif 'BlockIDs' in block:
        ids = np.asarray(block['BlockIDs']).ravel()
    else:
        raise DecodeError('Missing block data')
    names, vol = _pack(ids, names, dims)
    return dims, names, vol, list(_vec(root.get('Offset', [0, 0, 0]))), []


def _legacy(root):
    dims = shape(*(int(root[k]) for k in ('Width','Height','Length')))
    count = int(np.prod(dims))
    block = np.asarray(root['Blocks'], dtype=np.uint8).ravel().astype(np.uint16)
    data = np.asarray(root.get('Data', np.zeros(count)), dtype=np.uint8).ravel()
    if len(block) != count or len(data) != count:
        raise DecodeError('Legacy Blocks/Data length mismatch')
    if 'AddBlocks' in root:
        add = np.asarray(root['AddBlocks'], dtype=np.uint8).ravel()
        if len(add) < (count+1)//2:
            raise DecodeError('Legacy AddBlocks truncated')
        nibbles = np.empty(len(add)*2, np.uint16)
        nibbles[0::2], nibbles[1::2] = add & 15, add >> 4
        block |= nibbles[:count] << 8
    legacy = json.loads((ASSETS/'legacy.json').read_text())['blocks']
    packed = (block.astype(np.uint32) << 4) | (data & 15)
    used, inv = np.unique(packed, return_inverse=True)
    names = [legacy.get(f'{int(v)>>4}:{int(v)&15}', f'legacy:block_{int(v)>>4}[data={int(v)&15}]') for v in used]
    names, vol = _pack(inv, names, dims)
    return dims, names, vol, [0,0,0], []


def _structure(root):
    if root.get('sub_levels'):
        raise DecodeError('Structure sub_levels require explicit coordinate format; original retained')
    if 'palettes' in root and len(root['palettes']) != 1:
        raise DecodeError('Multiple structure alternatives require explicit selection')
    dims = shape(*_vec(root['size']))
    names = _palette(root.get('palette', root.get('palettes', [[]])[0]))
    palette = ['minecraft:air'] + names
    vol = np.zeros((dims[1], dims[2], dims[0]), np.uint32)
    occupied = set()
    for b in root.get('blocks', []):
        x,y,z = _vec(b['pos']); idx=int(b['state'])
        if not (0<=x<dims[0] and 0<=y<dims[1] and 0<=z<dims[2] and 0<=idx<len(names)):
            raise DecodeError('Structure block outside bounds/palette')
        if (x,y,z) in occupied:
            raise DecodeError('Duplicate structure block position')
        occupied.add((x,y,z)); vol[y,z,x]=idx+1
    names, vol = _pack(vol, palette, dims)
    return dims,names,vol,[0,0,0],[]


def decode_path(path):
    path = Path(path)
    if path.suffix.lower() == '.mcstructure':
        raise DecodeError('Bedrock layers/states are not Java; original retained')
    try:
        root = nbtlib.load(path)
    except Exception as e:
        raise DecodeError(f'NBT read failed: {e}') from e
    if 'Schematic' in root and isinstance(root['Schematic'], dict):
        root=root['Schematic']
    if 'Regions' in root:
        fmt='litematic'; values=_litematic(root)
    elif 'Palette' in root or isinstance(root.get('Blocks'), dict):
        fmt='sponge'+str(int(root.get('Version',2))); values=_sponge(root)
    elif 'Blocks' in root and 'Width' in root:
        fmt='legacy'; values=_legacy(root)
    elif 'size' in root and ('palette' in root or 'palettes' in root):
        fmt='structure'; values=_structure(root)
    else:
        raise DecodeError('Unsupported container; original retained')
    dims,names,vol,origin,regions=values
    return {'fmt':fmt,'dims':dims,'names':names,'vol':vol,
            'meta':{'data_version':int(root.get('MinecraftDataVersion',root.get('DataVersion',0))),
                    'origin_xyz':origin,'regions':regions,
                    'overlap_policy':'non_air_union_conflicting_solids_rejected',
                    'block_entities_preserved_in_source':True},
            'source':str(path)}


def export_litematic(path, xyz, palette, *, origin=(0,0,0), data_version=4189, name='mcfix'):
    """Fast palette-preserving exporter. Does not synthesize invalid properties."""
    xyz = np.asarray(xyz)
    w,h,l = shape(*xyz.shape)
    if np.any(xyz < 0) or np.any(xyz >= len(palette)):
        raise ValueError('Export palette index out of range')
    nbtpal=[]
    for entry in palette:
        bn,p=parse_state(entry)
        compound=nbtlib.Compound({'Name':nbtlib.String(bn)})
        if p: compound['Properties']=nbtlib.Compound({k:nbtlib.String(v) for k,v in p.items()})
        nbtpal.append(compound)
    def vec(v):return nbtlib.Compound({k:nbtlib.Int(int(x)) for k,x in zip('xyz',v)})
    region=nbtlib.Compound({'Position':vec(origin),'Size':vec((w,h,l)),
        'BlockStatePalette':nbtlib.List[nbtlib.Compound](nbtpal),
        'BlockStates':nbtlib.LongArray(pack_longs(xyz.transpose(1,2,0).ravel(),max(2,(len(palette)-1).bit_length())))})
    for key in ['Entities','TileEntities','PendingBlockTicks','PendingFluidTicks']:
        region[key]=nbtlib.List[nbtlib.Compound]()
    root=nbtlib.File({'Version':nbtlib.Int(6),'SubVersion':nbtlib.Int(1),'MinecraftDataVersion':nbtlib.Int(data_version),
        'Metadata':nbtlib.Compound({'Name':nbtlib.String(name),'Author':nbtlib.String('mcfix'),
            'Description':nbtlib.String('State-preserving block export; entity NBT retained in source archive'),
            'RegionCount':nbtlib.Int(1),'TimeCreated':nbtlib.Long(0),'TimeModified':nbtlib.Long(0),
            'TotalVolume':nbtlib.Int(w*h*l),'TotalBlocks':nbtlib.Int(int(non_air(xyz,palette).sum())),
            'EnclosingSize':vec((w,h,l))}),
        'Regions':nbtlib.Compound({'main':region})})
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');root.save(tmp,gzipped=True);tmp.replace(path)
