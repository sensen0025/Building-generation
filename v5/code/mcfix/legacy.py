"""Explicit metadata for historical five-field checkpoints, never full states."""
from functools import lru_cache
from pathlib import Path
import json
from .codec import registry

@lru_cache(maxsize=1)
def legacy_state_classes(config='/home/linux/redstone_ai/TRELLIS/outputs/ss_vae_mc_v2/config.json'):
    c=json.loads(Path(config).read_text())
    dims=c['trainer']['args']['state_field_dims']
    if dims!=[7,3,6,16,3]:raise ValueError('Historical checkpoint state schema mismatch')
    return tuple(c['trainer']['args']['state_class_list'])

def legacy_properties(name,packed):
    r=registry();sid=r.legacy_to_state(name,packed);return dict(r.properties[sid])
