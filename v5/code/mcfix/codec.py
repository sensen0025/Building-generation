"""Lossless state strings; validated registry IDs; explicit legacy projection.

Global registry IDs are version-specific and must travel with registry_hash.
The old five-field integer is only an explicitly lossy compatibility feature.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import re
import numpy as np

ASSETS = Path(__file__).parent / 'assets'
DIRECTIONS = ('north', 'east', 'south', 'west')
RENAMES = {'minecraft:oak_plank': 'minecraft:oak_planks',
           'minecraft:spruce_plank': 'minecraft:spruce_planks',
           'minecraft:birch_plank': 'minecraft:birch_planks',
           'minecraft:white_stained_clay': 'minecraft:white_terracotta',
           'minecraft:grass': 'minecraft:short_grass'}


def parse_state(state):
    if not isinstance(state, str):
        raise TypeError('state must be a string')
    m = re.fullmatch(r'([\w.-]+:[\w./-]+|[\w./-]+)(?:\[([^\[\]]*)\])?', state.strip())
    if not m:
        raise ValueError(f'Malformed blockstate: {state!r}')
    name = m[1] if ':' in m[1] else 'minecraft:' + m[1]
    props = {}
    if m[2]:
        for pair in m[2].split(','):
            key, sep, value = pair.partition('=')
            if not sep or not key or not value or key in props:
                raise ValueError(f'Malformed/duplicate property: {pair!r}')
            props[key] = value
    return name, props


def state_string(name, props=None):
    name, base = parse_state(name)
    if props is not None:
        if base:
            raise ValueError('Properties supplied twice')
        base = {str(k): str(v).lower() if isinstance(v, bool) else str(v) for k, v in props.items()}
    return name + ('[' + ','.join(f'{k}={v}' for k, v in sorted(base.items())) + ']' if base else '')


def rotate_state(state, k=1):
    """Rotate with np.transpose(a[:,:,::-1], (2,1,0)): N->E->S->W.

    Does not infer or discard unknown properties. Unknown directional enum
    values are rejected if they cannot be transformed unambiguously.
    """
    name, p = parse_state(state)
    k %= 4
    if not k:
        return state_string(name, p)
    rotate = {d: DIRECTIONS[(i + k) % 4] for i, d in enumerate(DIRECTIONS)}
    q = {rotate.get(key, key): value for key, value in p.items()}
    for key in ('facing', 'horizontal_facing'):
        if key in q:
            q[key] = rotate.get(q[key], q[key])
    if q.get('axis') in ('x', 'z') and k % 2:
        q['axis'] = 'z' if q['axis'] == 'x' else 'x'
    if 'rotation' in q:
        value = int(q['rotation'])
        if not 0 <= value < 16:
            raise ValueError('Expected 16-position rotation')
        q['rotation'] = str((value + 4 * k) % 16)
    shape = q.get('shape')
    if shape in ('north_south', 'east_west'):
        if k % 2:
            q['shape'] = 'east_west' if shape == 'north_south' else 'north_south'
    elif shape and shape.startswith('ascending_'):
        q['shape'] = 'ascending_' + rotate[shape[len('ascending_'):]]
    elif shape in ('south_east', 'south_west', 'north_west', 'north_east'):
        corners = ['south_east', 'south_west', 'north_west', 'north_east']
        q['shape'] = corners[(corners.index(shape) + k) % 4]
    if 'orientation' in q:
        q['orientation'] = '_'.join(rotate.get(v, v) for v in q['orientation'].split('_'))
    return state_string(name, q)


def rotate_volume(a, k=1):
    for _ in range(k % 4):
        a = np.transpose(a[:, :, ::-1], (2, 1, 0))
    return np.ascontiguousarray(a)


class Registry:
    def __init__(self, path=None):
        path = Path(path) if path else ASSETS / 'blocks_1.21.4.json'
        raw = path.read_bytes()
        self.hash = hashlib.sha256(raw).hexdigest()
        self.version = '1.21.4'
        self.data_version = 4189
        self.blocks = {'minecraft:' + b['name']: b for b in json.loads(raw)}
        self.names = {}
        self.properties = {}
        self.state_ids = {}
        self.defaults = {}
        for name, block in self.blocks.items():
            for sid in range(block['minStateId'], block['maxStateId'] + 1):
                index = sid - block['minStateId']
                props = {}
                for prop in reversed(block['states']):
                    values = self.values(prop)
                    props[prop['name']] = values[index % len(values)]
                    index //= len(values)
                if index:
                    raise ValueError('Invalid registry mixed-radix cardinality')
                self.names[sid] = name
                self.properties[sid] = props
                self.state_ids[state_string(name, props)] = sid
            self.defaults[name] = block['defaultState']
        self.rotation = np.empty((4, max(self.names) + 1), dtype=np.int32)
        for sid in self.names:
            for k in range(4):
                s = rotate_state(self.string(sid), k)
                if s not in self.state_ids:
                    raise ValueError(f'Rotation outside registry: {s}')
                self.rotation[k, sid] = self.state_ids[s]

    @staticmethod
    def values(prop):
        if prop['type'] == 'bool':
            return ['true', 'false']  # Minecraft/Prismarine numeric ordering
        return [str(v) for v in prop.get('values', range(prop['num_values']))]

    def encode(self, state, *, fill_defaults=True):
        name, props = parse_state(state)
        if name not in self.blocks:
            raise ValueError(f'Unsupported block in {self.version}: {name}')
        legal = {p['name']: self.values(p) for p in self.blocks[name]['states']}
        for key, value in props.items():
            if key not in legal or value not in legal[key]:
                raise ValueError(f'Illegal {name} property {key}={value}')
        if not fill_defaults and set(props) != set(legal):
            raise ValueError(f'Incomplete state: {state}')
        full = dict(self.properties[self.defaults[name]]) if fill_defaults else {}
        full.update(props)
        return self.state_ids[state_string(name, full)]

    def string(self, sid):
        sid = int(sid)
        if sid not in self.names:
            raise ValueError(f'Invalid registry state ID: {sid}')
        return state_string(self.names[sid], self.properties[sid])

    def normalize_source(self, state, data_version=None):
        """Only explicit spelling/version renames. No shape/material aliases."""
        name, props = parse_state(state)
        normalized = RENAMES.get(name, name)
        if name == 'minecraft:grass':
            if data_version is not None and 0 < data_version <= 1343:
                normalized = 'minecraft:grass_block'
            elif data_version is not None and data_version >= 1519:
                normalized = 'minecraft:short_grass'
            else:
                raise ValueError('Ambiguous historical minecraft:grass without a supported DataVersion')
        sid = self.encode(state_string(normalized, props))
        missing = sorted(set(self.properties[sid]) - set(props))
        return sid, {'renamed': normalized != name, 'defaulted_properties': missing}

    def project_legacy(self, sid):
        """Lossy adapter for frozen v4 flow ONLY; never use for source storage."""
        p = self.properties[int(sid)]
        o = {'north': 1, 'south': 2, 'east': 3, 'west': 4, 'up': 5, 'down': 6}.get(p.get('facing'), 0)
        if not o:
            o = {'x': 3, 'z': 1, 'y': 5}.get(p.get('axis'), 0)
        v = {'top': 1, 'upper': 1, 'bottom': 2, 'lower': 2}.get(p.get('half', p.get('type')), 0)
        sh = {'straight': 1, 'inner_left': 2, 'inner_right': 3, 'outer_left': 4, 'outer_right': 5}.get(p.get('shape'), 0)
        hz = sum((1 << i) for i, d in enumerate(('north', 'south', 'east', 'west')) if p.get(d) == 'true')
        ac = next((2 if p[key] == 'true' else 1 for key in ('lit', 'powered', 'open', 'triggered', 'extended') if key in p), 0)
        return o + 7 * (v + 3 * (sh + 6 * (hz + 16 * ac)))

    def legacy_to_state(self, name, packed):
        """Best-effort migration, explicit lossy status required by callers.

        Uses block-specific allowed properties, cannot recover missing fields.
        """
        name = RENAMES.get(name, name)
        sid = self.encode(name)
        packed = int(packed)
        if not 0 <= packed < 6048:
            raise ValueError('Legacy packed state outside 0..6047')
        o, v, sh, hz, ac = [(packed // m) % d for m, d in zip((1, 7, 21, 126, 2016), (7, 3, 6, 16, 3))]
        props = dict(self.properties[sid])
        changes = {'facing': {1: 'north', 2: 'south', 3: 'east', 4: 'west', 5: 'up', 6: 'down'}.get(o),
                   'axis': {1: 'z', 3: 'x', 5: 'y'}.get(o),
                   'shape': {1: 'straight', 2: 'inner_left', 3: 'inner_right', 4: 'outer_left', 5: 'outer_right'}.get(sh)}
        if 'half' in props:
            allowed = next(self.values(p) for p in self.blocks[name]['states'] if p['name'] == 'half')
            changes['half'] = ({1: 'upper', 2: 'lower'} if 'upper' in allowed else {1: 'top', 2: 'bottom'}).get(v)
        if 'type' in props and name.endswith('_slab'):
            changes['type'] = {1: 'top', 2: 'bottom'}.get(v)
        for i, d in enumerate(('north', 'south', 'east', 'west')):
            if props.get(d) in ('true', 'false'):
                changes[d] = 'true' if hz & (1 << i) else 'false'
        # Only the first field was encoded by v4. Never fabricate the other booleans.
        if ac:
            key = next((k for k in ('lit', 'powered', 'open', 'triggered', 'extended') if k in props), None)
            if key:
                changes[key] = 'true' if ac == 2 else 'false'
        legal = {p['name']: self.values(p) for p in self.blocks[name]['states']}
        for key, value in changes.items():
            if key in legal and value in legal[key]:
                props[key] = value
        return self.encode(state_string(name, props))


@lru_cache(maxsize=1)
def registry():
    return Registry()
