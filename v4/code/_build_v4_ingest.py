# -*- coding: utf-8 -*-
"""Stage B: ingest new slice dataset into v4 training arrays.

Reads /data/redstone_v4/qualified.json + sources/, decodes each sample,
extracts cut slices (or full fits64 volumes), refits bbox, centers in 64^3,
maps blocks via alias table to frozen 128-class map, packs states.

Outputs in /data/redstone_v4/work/:
  ingest_manifest.json   per-sample records (source, md5, coverage, category)
  ss64_new.npy           (N,64,64,64) uint8 class ids
  ss64_state_new.npy     (N,64,64,64) uint16 packed 5-field states
  alias_report.json      block name mapping stats
"""
import os, sys, json, hashlib, time
import numpy as np

sys.path.insert(0, '/data/redstone_v4/code')
from decode import decode_path

SRC = '/data/redstone_v4/sources'
WORK = '/data/redstone_v4/work'
C = 64
N_CLASSES = 128
OTHER = 127

# ---------- frozen 128-class map (from server v2) ----------
VOCAB_PATH = '/home/linux/redstone_ai/dataset/BLOCK_VOCAB_FULL.txt'
vocab = [l.strip() for l in open(VOCAB_PATH, encoding='utf-8') if l.strip()] + ['minecraft:other']
bm = np.load('/home/linux/redstone_ai/dataset/v2/block_map.npy')  # new128 -> old202 idx
old2new = {vocab[int(o)]: n for n, o in enumerate(bm)}

# ---------- alias mapping: shape family + color preserving ----------
SHAPE_SUFFIXES = ['stairs', 'slab', 'wall', 'fence', 'fence_gate', 'trapdoor', 'door',
                  'pressure_plate', 'button', 'candle', 'carpet', 'stained_glass',
                  'terracotta', 'concrete', 'shulker_box', 'banner', 'sign', 'coral',
                  'leaves', 'sapling', 'bed']
COLORS = ['white', 'orange', 'magenta', 'light_blue', 'yellow', 'lime', 'pink', 'gray',
          'light_gray', 'cyan', 'purple', 'blue', 'brown', 'green', 'red', 'black']
# same-family fallback (sorted by visual closeness preference)
FAMILY_FALLBACK = {
    'stairs': ['oak_stairs', 'stone_brick_stairs'],
    'slab': ['oak_slab', 'stone_brick_slab'],
    'wall': ['stone_brick_wall', 'cobblestone_wall'],
    'fence': ['oak_fence'],
    'fence_gate': ['oak_fence_gate'],
    'trapdoor': ['oak_trapdoor'],
    'door': ['oak_door'],
    'pressure_plate': ['oak_pressure_plate', 'stone_pressure_plate'],
    'button': ['oak_button', 'stone_button'],
    'candle': ['white_candle'],
    'carpet': ['white_carpet'],
    'stained_glass': ['white_stained_glass'],
    'terracotta': ['terracotta'],
    'concrete': ['white_concrete'],
    'shulker_box': ['shulker_box'],
    'banner': ['white_banner'],
    'sign': ['oak_sign'],
    'leaves': ['oak_leaves'],
    'sapling': ['oak_sapling'],
    'bed': ['red_bed'],
}
SPECIAL = {
    'sea_lantern': 'glowstone', 'lantern': 'glowstone', 'end_rod': 'glowstone',
    'shroomlight': 'glowstone', 'ochre_froglight': 'glowstone',
    'verdant_froglight': 'quartz_block', 'pearlescent_froglight': 'quartz_block',
    'coarse_dirt': 'dirt', 'podzol': 'dirt', 'rooted_dirt': 'dirt',
    'grass_path': 'dirt', 'farmland': 'dirt',
    'cobbled_deepslate': 'cobblestone', 'deepslate_bricks': 'stone_bricks',
    'deepslate_tiles': 'stone_bricks', 'polished_deepslate': 'stone_bricks',
    'chiseled_deepslate': 'stone_bricks', 'cracked_deepslate_bricks': 'stone_bricks',
    'chiseled_stone_bricks': 'stone_bricks', 'cracked_stone_bricks': 'stone_bricks',
    'mossy_stone_bricks': 'stone_bricks', 'mossy_cobblestone': 'cobblestone',
    'polished_andesite': 'andesite', 'polished_diorite': 'diorite',
    'polished_granite': 'granite', 'polished_basalt': 'basalt',
    'polished_blackstone': 'blackstone', 'polished_blackstone_bricks': 'blackstone',
    'chiseled_sandstone': 'sandstone', 'cut_sandstone': 'sandstone',
    'smooth_sandstone': 'sandstone', 'chiseled_red_sandstone': 'red_sandstone',
    'cut_red_sandstone': 'red_sandstone', 'smooth_red_sandstone': 'red_sandstone',
    'chiseled_quartz_block': 'quartz_block', 'quartz_bricks': 'quartz_block',
    'quartz_pillar': 'quartz_block', 'smooth_quartz': 'quartz_block',
    'white_concrete_powder': 'white_concrete', 'concrete_powder': 'gravel',
    'brick_wall': 'stone_brick_wall', 'stone_brick_wall': 'stone_brick_wall',
    'mud_bricks': 'stone_bricks', 'packed_mud': 'stone_bricks',
    'purpur_block': 'purpur_pillar', 'purpur_pillar': 'purpur_pillar',
    'nether_bricks': 'netherrack',
    'red_nether_bricks': 'nether_bricks', 'prismarine_bricks': 'prismarine',
    'dark_prismarine': 'prismarine', 'prismarine_wall': 'stone_brick_wall',
    'smooth_stone': 'stone', 'dripstone_block': 'stone', 'tuff': 'stone',
    'calcite': 'quartz_block', 'copper_block': 'iron_block',
    'stripped_oak_log': 'oak_log', 'stripped_spruce_log': 'spruce_log',
    'stripped_birch_log': 'birch_log', 'stripped_jungle_log': 'jungle_log',
    'stripped_acacia_log': 'acacia_log', 'stripped_dark_oak_log': 'dark_oak_log',
    'stripped_cherry_log': 'cherry_log', 'stripped_mangrove_log': 'mangrove_log',
    'stripped_oak_wood': 'oak_log', 'stripped_spruce_wood': 'spruce_log',
    'stripped_birch_wood': 'birch_log', 'stripped_jungle_wood': 'jungle_log',
    'stripped_acacia_wood': 'acacia_log', 'stripped_dark_oak_wood': 'dark_oak_log',
    'stripped_cherry_wood': 'cherry_log', 'stripped_mangrove_wood': 'mangrove_log',
    'oak_wood': 'oak_log', 'spruce_wood': 'spruce_log', 'birch_wood': 'birch_log',
    'jungle_wood': 'jungle_log', 'acacia_wood': 'acacia_log',
    'dark_oak_wood': 'dark_oak_log', 'cherry_wood': 'cherry_log',
    'mangrove_wood': 'mangrove_log', 'bamboo_block': 'bamboo',
    'infested_stone': 'stone', 'infested_stone_bricks': 'stone_bricks',
    'infested_cobblestone': 'cobblestone', 'infested_deepslate': 'stone_bricks',
    'warped_wart_block': 'nether_wart_block', 'bone_block': 'bone_block',
    'dried_kelp_block': 'hay_block', 'moss_block': 'green_wool',
    'sculk': 'blackstone', 'sculk_catalyst': 'blackstone', 'mud': 'dirt',
    'snow_block': 'snow', 'ice': 'packed_ice', 'packed_ice': 'packed_ice',
    'blue_ice': 'packed_ice', 'frosted_ice': 'packed_ice', 'obsidian': 'obsidian',
    'crying_obsidian': 'obsidian', 'ancient_debris': 'netherrack',
    'gilded_blackstone': 'blackstone', 'crimson_nylium': 'netherrack',
    'warped_nylium': 'netherrack', 'mycelium': 'dirt',
}
# color carry within same material family: X_terracotta -> Y_terracotta etc.


def bare(name):
    return name.split('[')[0]


def resolve(name, family_cache):
    """blockstate full string -> (class128 id, canonical key) or (OTHER, None). Transitive, <=4 hops."""
    base = bare(name)
    seen = 0
    cur = base
    while seen < 4:
        seen += 1
        if cur in old2new and old2new[cur] != OTHER:
            return old2new[cur], cur
        nxt = None
        m = cur[len('minecraft:'):]
        # waxed_X -> X
        if m.startswith('waxed_'):
            nxt = 'minecraft:' + m[len('waxed_'):]
        elif m in SPECIAL and 'minecraft:' + SPECIAL[m] != cur:
            nxt = 'minecraft:' + SPECIAL[m]
        elif m.startswith('stripped_'):
            other = 'minecraft:' + m[len('stripped_'):]
            nxt = other  # stripped_X_log -> X_log (direct map has spruce_log etc)
        elif m.endswith('_wall'):
            mat = m[:-len('_wall')]
            # wall color/material -> plain block
            for cand in ['minecraft:' + mat, 'minecraft:' + mat.replace('stone_brick', 'stone_bricks'),
                         'minecraft:' + mat.replace('mud_brick', 'stone_bricks'),
                         'minecraft:cobblestone', 'minecraft:stone_bricks']:
                if cand in old2new and old2new[cand] != OTHER:
                    return old2new[cand], cand
            nxt = 'minecraft:cobblestone'
        elif m.endswith('_bricks'):
            for cand in ['minecraft:bricks', 'minecraft:stone_bricks', 'minecraft:netherrack']:
                if cand in old2new and old2new[cand] != OTHER:
                    return old2new[cand], cand
            nxt = 'minecraft:bricks'
        elif m.endswith('_planks'):
            for v in ['minecraft:oak_planks', 'minecraft:spruce_planks', 'minecraft:birch_planks']:
                if v in old2new and old2new[v] != OTHER:
                    return old2new[v], v
            nxt = 'minecraft:oak_planks'
        elif m.endswith('_log'):
            nxt = 'minecraft:oak_log'
        elif m.endswith('_wood'):
            nxt = 'minecraft:' + m[:-len('_wood')] + '_log'
        else:
            # color prefix strip within family
            done = False
            for col in COLORS:
                pre = col + '_'
                if m.startswith(pre):
                    rest = m[len(pre):]
                    for c2 in ['white', 'light_gray', 'gray', 'black']:
                        cand = 'minecraft:' + c2 + '_' + rest
                        if cand in old2new and old2new[cand] != OTHER:
                            return old2new[cand], cand
                    if rest in old2new and old2new[rest] != OTHER:
                        return old2new[rest], rest
                    # family fallback: any *_<rest> in map
                    for v in sorted(old2new):
                        if v.endswith('_' + rest) and old2new[v] != OTHER:
                            return old2new[v], v
                    if rest in [k[len('minecraft:'):] for k in old2new]:
                        nxt = 'minecraft:' + rest
                    else:
                        for suf in SHAPE_SUFFIXES:
                            s2 = '_' + suf
                            if rest.endswith(s2):
                                for v in sorted(old2new):
                                    if v.endswith(s2) and old2new[v] != OTHER:
                                        return old2new[v], v
                        nxt = None
                    done = True
                    break
            if not done:
                # shape family fallback on full name
                for suf in SHAPE_SUFFIXES:
                    s2 = '_' + suf
                    if m.endswith(s2):
                        for v in sorted(old2new):
                            if v.endswith(s2) and old2new[v] != OTHER:
                                return old2new[v], v
                break
        if nxt is None or nxt == cur:
            break
        cur = nxt
    return OTHER, None

SPECIAL.update({
    'torch': 'end_rod', 'wall_torch': 'wall_torch', 'lantern': 'glowstone',
    'redstone_torch': 'end_rod', 'soul_torch': 'end_rod', 'redstone_lamp': 'glowstone',
    'chain': 'iron_bars', 'iron_bars': 'iron_bars', 'scaffolding': 'ladder',
    'bookshelf': 'oak_planks', 'chiseled_bookshelf': 'oak_planks',
    'crafting_table': 'spruce_planks', 'cartography_table': 'spruce_planks',
    'fletching_table': 'birch_planks', 'smithing_table': 'dark_oak_planks',
    'barrel': 'chest', 'chest': 'chest', 'trapped_chest': 'chest', 'ender_chest': 'chest',
    'gravel': 'cobblestone', 'soul_sand': 'dirt', 'soul_soil': 'dirt',
    'redstone_block': 'red_concrete', 'redstone_wire': 'red_concrete',
    'redstone_torch_off': 'end_rod', 'observer': 'quartz_block',
    'repeater': 'stone', 'comparator': 'stone', 'piston': 'stone_bricks',
    'sticky_piston': 'stone_bricks', 'dropper': 'stone', 'dispenser': 'stone',
    'hopper': 'iron_block', 'lever': 'stone', 'tripwire': 'iron_bars',
    'target': 'white_wool', 'daylight_detector': 'stone_slab',
    'rail': 'iron_bars', 'powered_rail': 'iron_bars', 'detector_rail': 'iron_bars',
    'activator_rail': 'iron_bars',
    'oak_door': 'oak_trapdoor', 'spruce_door': 'spruce_trapdoor',
    'birch_door': 'oak_trapdoor', 'jungle_door': 'oak_trapdoor',
    'acacia_door': 'oak_trapdoor', 'dark_oak_door': 'dark_oak_trapdoor',
    'mangrove_door': 'dark_oak_trapdoor', 'cherry_door': 'oak_trapdoor',
    'iron_door': 'iron_trapdoor', 'door': 'oak_trapdoor',
    'tall_grass': 'vine', 'fern': 'vine', 'large_fern': 'vine', 'dead_bush': 'vine',
    'sugar_cane': 'vine', 'bamboo': 'vine', 'cactus': 'lime_concrete',
    'brown_mushroom_block': 'spruce_planks', 'red_mushroom_block': 'pink_concrete',
    'mushroom_stem': 'mushroom_stem',
    'amethyst_block': 'purple_concrete', 'budding_amethyst': 'purple_concrete',
    'copper_ore': 'andesite', 'iron_ore': 'andesite', 'gold_ore': 'andesite',
    'coal_ore': 'stone', 'diamond_ore': 'stone', 'emerald_ore': 'stone',
    'lapis_ore': 'stone', 'redstone_ore': 'stone',
    'deepslate': 'polished_deepslate', 'cobbled_deepslate': 'cobblestone',
    'create:\\window_in_a_block': 'glass', 'window_in_a_block': 'glass',
    'beehive': 'spruce_planks', 'bee_nest': 'spruce_planks',
    'composter': 'oak_fence', 'note_block': 'spruce_planks', 'jukebox': 'dark_oak_planks',
    'loom': 'spruce_planks', 'smoker': 'stone', 'blast_furnace': 'stone',
    'furnace': 'stone', 'anvil': 'iron_block', 'chipped_anvil': 'iron_block',
    'damaged_anvil': 'iron_block', 'cauldron': 'iron_block', 'water_cauldron': 'iron_block',
    'bell': 'gold_block' if False else 'iron_block', 'lightning_rod': 'iron_block',
    'flower_pot': 'terracotta' if False else 'cyan_terracotta',
    'cake': 'white_wool', 'cobweb': 'white_stained_glass', 'sponge': 'yellow_concrete',
    'wet_sponge': 'yellow_concrete', 'slime_block': 'lime_concrete',
    'honey_block': 'orange_concrete', 'honeycomb_block': 'orange_concrete',
    'mushroom': 'white_wool', 'wheat': 'yellow_concrete', 'carrots': 'orange_concrete',
    'potatoes': 'yellow_concrete', 'beetroot': 'red_concrete', 'pumpkin': 'orange_concrete',
    'carved_pumpkin': 'orange_concrete', 'jack_o_lantern': 'orange_concrete',
    'melon': 'lime_concrete', 'hay_block': 'yellow_concrete',
    'terracotta': 'cyan_terracotta',
    'red_nether_bricks': 'nether_brick_stairs', 'nether_bricks': 'nether_brick_stairs',
    'coal_block': 'black_concrete', 'diorite': 'polished_diorite',
    'prismarine_bricks': 'deepslate_tiles', 'dead_tube_coral_block': 'purple_concrete',
    'dead_brain_coral_block': 'gray_concrete', 'dead_bubble_coral_block': 'gray_concrete',
    'dead_fire_coral_block': 'red_concrete', 'dead_horn_coral_block': 'yellow_concrete',
    'tube_coral_block': 'purple_concrete', 'brain_coral_block': 'pink_concrete',
    'bubble_coral_block': 'magenta_concrete', 'fire_coral_block': 'red_concrete',
    'horn_coral_block': 'yellow_concrete',
    'purpur_block': 'quartz_block', 'purpur_pillar': 'quartz_pillar',
    'waxed_exposed_copper': 'light_gray_concrete', 'waxed_oxidized_copper': 'cyan_concrete',
    'waxed_weathered_copper': 'lime_concrete', 'waxed_copper_block': 'orange_concrete',
    'exposed_copper': 'light_gray_concrete', 'oxidized_copper': 'cyan_concrete',
    'weathered_copper': 'lime_concrete', 'copper_block': 'orange_concrete',
    'waxed_exposed_cut_copper': 'light_gray_concrete', 'waxed_oxidized_cut_copper': 'cyan_concrete',
    'waxed_weathered_cut_copper': 'lime_concrete', 'waxed_cut_copper': 'orange_concrete',
    'exposed_cut_copper': 'light_gray_concrete', 'oxidized_cut_copper': 'cyan_concrete',
    'weathered_cut_copper': 'lime_concrete', 'cut_copper': 'orange_concrete',
    'emerald_block': 'lime_concrete', 'nether_wart_block': 'red_concrete',
    'window_in_a_block': 'glass', 'create:window_in_a_block': 'glass',
    'rose_bush': 'red_concrete', 'peony': 'pink_concrete', 'lilac': 'magenta_concrete',
    'sunflower': 'yellow_concrete', 'rose': 'red_concrete', 'dandelion': 'yellow_concrete',
    'poppy': 'red_concrete', 'cornflower': 'blue_concrete', 'lily_of_the_valley': 'white_wool',
    'oxeye_daisy': 'white_wool', 'allium': 'magenta_concrete', 'azure_bluet': 'white_wool',
    'blue_orchid': 'blue_concrete', 'tulip': 'red_concrete', 'orchid': 'blue_concrete',
    'light_gray_concrete_powder': 'light_gray_concrete', 'polished_blackstone': 'polished_blackstone_bricks',
    'gilded_blackstone': 'polished_blackstone_bricks',
    'lapis_block': 'blue_concrete', 'gold_block': 'yellow_concrete', 'diamond_block': 'light_blue_concrete',
    'coal_ore': 'stone', 'nether_quartz_ore': 'netherrack' if False else 'polished_blackstone_bricks',
    'quartz_ore': 'polished_blackstone_bricks',
 'white_terracotta': 'white_concrete',
    'yellow_terracotta': 'yellow_concrete', 'orange_terracotta': 'orange_concrete',
    'red_terracotta': 'red_concrete', 'brown_terracotta': 'brown_concrete',
    'black_terracotta': 'black_concrete', 'gray_terracotta': 'gray_concrete',
    'light_gray_terracotta': 'light_gray_concrete', 'magenta_terracotta': 'magenta_concrete',
    'pink_terracotta': 'pink_concrete', 'lime_terracotta': 'lime_concrete',
    'green_terracotta': 'green_concrete', 'blue_terracotta': 'blue_concrete',
    'light_blue_terracotta': 'light_blue_concrete', 'purple_terracotta': 'purple_concrete',
    'cyan_terracotta': 'cyan_terracotta',
})

# ---------- state packing (must match server _voxelize.pack_state) ----------
def parse_props(name):
    if '[' not in name:
        return {}
    inner = name[name.index('[') + 1:name.rindex(']')]
    props = {}
    for kv in inner.split(','):
        if '=' in kv:
            k, v = kv.split('=', 1)
            props[k] = v
    return props


def pack_state_from_props(p):
    om = {"north": 1, "south": 2, "east": 3, "west": 4, "up": 5, "down": 6}
    o = om.get(p.get("facing", ""), 0)
    if o == 0 and p.get("axis", "") in ("x", "z", "y"):
        o = {"x": 3, "z": 1, "y": 5}[p["axis"]]
    hf = p.get("half", "")
    v = 1 if hf in ("top", "upper") else (2 if hf in ("bottom", "lower") else 0)
    tp = p.get("type", "")
    if v == 0 and tp == "top":
        v = 1
    elif v == 0 and tp == "bottom":
        v = 2
    sm = {"straight": 1, "inner_left": 2, "inner_right": 3, "outer_left": 4, "outer_right": 5}
    sh = sm.get(p.get("shape", ""), 0)
    hz = 0
    for i, d in enumerate(["north", "south", "east", "west"]):
        if str(p.get(d, "")).lower() == "true":
            hz |= (1 << i)
    ac = 0
    for k in ["lit", "powered", "open", "triggered", "extended"]:
        if str(p.get(k, "")).lower() == "true":
            ac = 2
            break
        if str(p.get(k, "")).lower() == "false":
            ac = 1
            break
    return o + 7 * (v + 3 * (sh + 6 * (hz + 16 * ac)))


def main():
    raise RuntimeError('Legacy lossy ingest retired. Run python -m mcfix.build --out /data/redstone_v5/dataset')
    t_start = time.time()
    d = json.load(open('/data/redstone_v4/qualified.json'))
    old_md5 = {r['md5'] for r in json.load(open(f'{WORK}/old_valid_md5.json'))}
    files = d['files']
    f2m = {f['file']: f['md5'] for f in files}

    # ---- build sample list ----
    samples = []  # (src_file, md5, origin_yzx or None, dims_whl or None, kind, type_label)
    for f in files:
        if f['category'] == 'fits64' and f['md5'] not in old_md5:
            samples.append({'file': f['file'], 'md5': f['md5'], 'kind': 'fits64',
                            'type': f.get('type', ''), 'render': f.get('render', '')})
    for c in d['qualified_cuts']:
        samples.append({'file': c['file'], 'md5': f2m.get(c['file'], ''), 'kind': 'cut',
                        'origin_yzx': c['origin_yzx'], 'dims_whl': c['dims_whl'],
                        'type': c.get('type', ''), 'render': c.get('render', '')})
    print(f'samples to ingest: {len(samples)} '
          f'(fits64 {sum(1 for s in samples if s["kind"]=="fits64")}, '
          f'cuts {sum(1 for s in samples if s["kind"]=="cut")})', flush=True)

    N = len(samples)
    ss = np.lib.format.open_memmap(f'{WORK}/ss64_new.npy', mode='w+', dtype=np.uint8, shape=(N, C, C, C))
    st = np.lib.format.open_memmap(f'{WORK}/ss64_state_new.npy', mode='w+', dtype=np.uint16, shape=(N, C, C, C))
    manifest = []
    alias_stats = {}
    family_cache = {}
    ndone = nskip = 0
    t0 = time.time()

    for i, s in enumerate(samples):
        rec = dict(s)
        try:
            dec = decode_path(os.path.join(SRC, s['file']))
            vol = dec['vol']  # [y, z, x] palette idx
            names = dec['names']
            if s['kind'] == 'cut':
                oy, oz, ox = s['origin_yzx']
                w, h, l = s['dims_whl']
                sl = vol[oy:oy + h, oz:oz + l, ox:ox + w]  # [y,z,x]
            else:
                sl = vol
            # transpose [y,z,x] -> [x,y,z]
            vox = np.ascontiguousarray(sl.transpose(2, 0, 1))
            # tight bbox refit
            nzc = np.argwhere(vox > 0)
            if len(nzc) == 0:
                raise ValueError('empty slice')
            mn, mx = nzc.min(0), nzc.max(0)
            vox = vox[mn[0]:mx[0] + 1, mn[1]:mx[1] + 1, mn[2]:mx[2] + 1]
            # center in 64^3
            D, Hh, W = vox.shape
            if D > C or Hh > C or W > C:
                raise ValueError(f'slice too big {vox.shape}')
            out = np.zeros((C, C, C), dtype=np.uint8)
            outs = np.zeros((C, C, C), dtype=np.uint16)
            ox0, oy0, oz0 = (C - D) // 2, (C - Hh) // 2, (C - W) // 2
            # map classes
            lut_cls = np.zeros(len(names), dtype=np.uint8)
            lut_st = np.zeros(len(names), dtype=np.uint16)
            n_alias = n_direct = 0
            for pi, nm in enumerate(names):
                if pi == 0:
                    continue
                cls, canon = resolve(nm, family_cache)
                lut_cls[pi] = cls
                lut_st[pi] = pack_state_from_props(parse_props(nm)) if cls != 0 else 0
                if canon is not None and bare(nm) == canon:
                    n_direct += 1
                elif canon is not None:
                    n_alias += 1
                    alias_stats[bare(nm)] = alias_stats.get(bare(nm), 0) + 1
            mapped = vox.astype(np.int64)
            cls_vol = lut_cls[mapped]
            st_vol = lut_st[mapped]
            # coverage stats (solid voxels only)
            solid = mapped > 0
            solid_n = int(solid.sum())
            if solid_n < 300:
                raise ValueError(f'too small {solid_n}')
            other_n = int(((cls_vol == OTHER) & solid).sum())
            coverage = 1.0 - other_n / solid_n
            if coverage < 0.85:
                raise ValueError(f'alias coverage {coverage:.2f} < 0.85')
            out[ox0:ox0 + D, oy0:oy0 + Hh, oz0:oz0 + W] = cls_vol
            outs[ox0:ox0 + D, oy0:oy0 + Hh, oz0:oz0 + W] = st_vol
            ss[i] = out
            st[i] = outs
            rec.update({'ok': True, 'idx': i, 'coverage': round(coverage, 4),
                        'solid': solid_n, 'content_dims': [int(D), int(Hh), int(W)],
                        'offset': [int(ox0), int(oy0), int(oz0)]})
            ndone += 1
        except Exception as e:
            rec.update({'ok': False, 'error': str(e)[:120]})
            nskip += 1
        manifest.append(rec)
        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print(f'{i+1}/{N} ok={ndone} skip={nskip} ({el:.0f}s, {el/(i+1)*1000:.0f}ms/it)', flush=True)
            ss.flush(); st.flush()
            json.dump(manifest, open(f'{WORK}/ingest_manifest.json', 'w'), ensure_ascii=False)

    ss.flush(); st.flush()
    json.dump(manifest, open(f'{WORK}/ingest_manifest.json', 'w'), ensure_ascii=False)
    json.dump(alias_stats, open(f'{WORK}/alias_report.json', 'w'), ensure_ascii=False)
    print(f'DONE: ok={ndone} skip={nskip} elapsed={time.time()-t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
