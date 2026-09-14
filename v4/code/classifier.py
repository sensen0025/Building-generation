"""Block classification for building-quality analysis.

Categories:
  redstone   - redstone components & machines (rule 1: disqualify)
  natural    - terrain blocks (rule 2: disqualify when dominant)
  building   - player-built structural blocks (cut-locking signal)
  detail     - interior detail (furniture, decoration)
  special    - map-art canvases / tech blocks (rule 3 heuristics)
"""
import re

NS = re.compile(r"^([a-z0-9_\-\.]+):([a-z0-9_/\-\.]+)(\[.*\])?$")


def bare(state):
    """'minecraft:oak_stairs[facing=north]' -> 'minecraft:oak_stairs'"""
    m = NS.match(state)
    if m:
        return m.group(1) + ":" + m.group(2)
    return state


def short(state):
    """'minecraft:oak_stairs[...]' -> 'oak_stairs'"""
    b = bare(state)
    return b.split(":", 1)[-1] if ":" in b else b


# ---------------------------------------------------------------- redstone ---
REDSTONE_EXACT = {
    "redstone_wire", "redstone_torch", "redstone_block", "redstone_lamp",
    "repeater", "comparator", "lever", "tripwire_hook", "tripwire",
    "observer", "dropper", "dispenser", "hopper", "piston", "sticky_piston",
    "piston_head", "slime_block", "honey_block", "target", "daylight_detector",
    "sculk_sensor", "calibrated_sculk_sensor", "lightning_rod", "note_block",
    "jukebox", "command_block", "chain_command_block", "repeating_command_block",
    "structure_block", "structure_void", "jigsaw", "barrier", "debug_stick",
    "tnt", "rail", "powered_rail", "detector_rail", "activator_rail",
    "hopper_minecart_placeholder", "lectern", "respawn_anchor", "lodestone",
    "beacon", "conduit", "end_rod", "light", "copper_bulb", "copper_grate",
    "heavy_core", "trial_spawner", "vault", "creaking_heart", "crafter",
    "copper_golem_statue_placeholder", "projector", "schematicannon",
    "and_gate", "or_gate", "not_gate", "xor_gate", "redstone_encoder",
    # create / tech mod blocks are strong machine indicators
    "andesite_casing", "brass_casing", "copper_casing", "railway_casing",
    "andesite_funnel", "brass_funnel", "andesite_tunnel", "brass_tunnel",
    "mechanical_press", "mechanical_mixer", "mechanical_drill", "mechanical_saw",
    "mechanical_arm", "mechanical_bearing", "mechanical_piston", "rope_pulley",
    "windmill_bearing", "flywheel", "cogwheel", "large_cogwheel", "shaft",
    "belt", "deployer", "depot", "chute", "portable_fluid_interface",
    "rotation_speed_controller", "crushing_wheel", "millstone", "mixer",
    "blaze_burner", "steam_engine", "steam_whistle", "fluid_tank", "item_vault",
    "spout", "drain", "smart_observer", "stock_link", "redstone_link",
    "display_link", "nixie_tube", "analog_lever", "brass_door", "copper_door",
}
REDSTONE_SUBSTR = (
    "_rail", "redstone", "repeater", "comparator", "piston", "observer",
    "hopper", "dispenser", "dropper", "command_block", "jigsaw", "cogwheel",
    "crushing", "mechanical_", "contraption", "create:", "windmill", "flywheel",
    "conjured", "clockwork",
)


def is_redstone(state):
    s = short(state)
    if s in REDSTONE_EXACT:
        return True
    if any(t in s for t in REDSTONE_SUBSTR):
        return True
    return False


# ---------------------------------------------------------------- natural ----
NATURAL_EXACT = {
    "stone", "deepslate", "dirt", "grass_block", "podzol", "mycelium",
    "coarse_dirt", "rooted_dirt", "mud", "muddy_mangrove_roots", "sand",
    "red_sand", "gravel", "clay", "sandstone", "red_sandstone", "bedrock",
    "water", "lava", "ice", "packed_ice", "blue_ice", "frosted_ice",
    "snow_block", "obsidian", "crying_obsidian", "netherrack", "basalt",
    "blackstone", "end_stone", "calcite", "tuff", "dripstone_block",
    "moss_block", "mossy_cobblestone", "sculk", "sculk_catalyst",
    "nether_quartz_ore", "nether_gold_ore", "ancient_debris", "glowstone",
    "magma_block", "soul_sand", "soul_soil", "terracotta", "white_terracotta",
    "orange_terracotta", "magenta_terracotta", "light_blue_terracotta",
    "yellow_terracotta", "lime_terracotta", "pink_terracotta",
    "gray_terracotta", "light_gray_terracotta", "cyan_terracotta",
    "purple_terracotta", "blue_terracotta", "brown_terracotta",
    "green_terracotta", "red_terracotta", "black_terracotta",
    "infested_stone", "infested_deepslate", "dripstone", "mossy_stone_bricks",
    "cracked_stone_bricks", "chiseled_stone_bricks", "cracked_deepslate_bricks",
    "deepslate_bricks", "deepslate_tiles", "polished_deepslate",
    "polished_basalt", "smooth_basalt", "tuff_bricks", "chiseled_tuff",
    "short_grass", "tall_grass", "fern", "large_fern", "dead_bush",
    "sugar_cane", "bamboo", "cactus", "vine", "glow_lichen", "seagrass",
    "tall_seagrass", "kelp", "kelp_plant", "lily_pad", "big_dripleaf",
    "small_dripleaf", "hanging_roots", "spore_blossom", "azalea",
    "flowering_azalea", "pumpkin", "carved_pumpkin", "melon", "wheat",
    "carrots", "potatoes", "beetroots", "farmland", "dirt_path",
    "nether_wart", "cave_vine", "cave_vines", "twisting_vines",
    "weeping_vines", "crimson_roots", "warped_roots", "nether_sprouts",
    "brown_mushroom", "red_mushroom", "cobweb", "snow", " pointed_dripstone",
}
# ores / stone variants / common cave + terrain fillers
NATURAL_SUBSTR = (
    "_ore", "dripstone", "azalea_leaves", "sculk_", "mushroom_block",
)


def is_natural(state):
    s = short(state)
    if s in NATURAL_EXACT:
        return True
    if s.endswith("_leaves") or s.endswith("_log") or s.endswith("_sapling"):
        return True
    if s.endswith("_coral") or s.endswith("_coral_block") or s.endswith("_coral_fan"):
        return True
    if any(t in s for t in NATURAL_SUBSTR):
        return True
    return False


# --------------------------------------------------------------- building ----
# Strong player-construction signal: shaped / manufactured blocks.
BUILDING_EXACT = {
    "bricks", "stone_bricks", "bookshelf", "crafting_table", "furnace",
    "chest", "glass", "glass_pane", "iron_bars", "iron_block", "gold_block",
    "diamond_block", "emerald_block", "lapis_block", "coal_block",
    "nether_bricks", "red_nether_bricks", "end_stone_bricks", "purpur_block",
    "prismarine", "prismarine_bricks", "dark_prismarine", "quartz_block",
    "smooth_quartz", "quartz_pillar", "quartz_bricks", "smooth_stone",
    "polished_andesite", "polished_diorite", "polished_granite", "andesite",
    "diorite", "granite", "sponge", "wet_sponge", "slime_block", "bone_block",
    "white_concrete", "orange_concrete", "magenta_concrete",
    "light_blue_concrete", "yellow_concrete", "lime_concrete", "pink_concrete",
    "gray_concrete", "light_gray_concrete", "cyan_concrete", "purple_concrete",
    "blue_concrete", "brown_concrete", "green_concrete", "red_concrete",
    "black_concrete", "white_concrete_powder", "white_glazed_terracotta",
    "red_stained_glass", "blue_stained_glass", "white_stained_glass",
    "black_stained_glass", "gray_stained_glass", "light_gray_stained_glass",
    "sea_lantern", "shroomlight", "ochre_froglight", "verdant_froglight",
    "pearlescent_froglight", "copper_block", "exposed_copper",
    "weathered_copper", "oxidized_copper", "waxed_copper_block", "amethyst_block",
    "budding_amethyst", "candle", "white_candle", "lantern", "soul_lantern",
    "chain", "iron_chain", "flower_pot", "torch", "wall_torch", "soul_torch",
    "soul_wall_torch", "lantern_wall", "chest_minecart_placeholder",
    "barrel", "smoker", "blast_furnace", "loom", "cartography_table",
    "fletching_table", "smithing_table", "stonecutter", "grindstone",
    "anvil", "chipped_anvil", "damaged_anvil", "cauldron", "water_cauldron",
    "composter", "bell", "campfire", "soul_campfire", "book", "engraved",
}
BUILDING_SUBSTR = (
    "_planks", "_stairs", "_slab", "_fence", "_fence_gate", "_door",
    "_trapdoor", "_pressure_plate", "_button", "_sign", "_wall_sign",
    "_wool", "_carpet", "_terracotta", "_stained_glass", "_stained_glass_pane",
    "_concrete", "_glazed_terracotta", "_candle", "_candle_cake",
    "polished_", "chiseled_", "cut_", "_bricks", "_copper", "brick_",
    "smooth_", "shulker_box", "_bed", "_banner", "wall_banner", "flowering",
    "stripped_", "blackstone_bricks", "purpur_", "quartz_", "prismarine_",
    "_shingles", "copper_", "iron_", "_pillar",
)


def is_building(state):
    s = short(state)
    if s in BUILDING_EXACT:
        return True
    if s.endswith("_wall") or s.endswith("_wall_torch"):
        return True
    if is_redstone(state) or is_natural(state):
        return False
    if any(t in s for t in BUILDING_SUBSTR):
        return True
    return False


# ----------------------------------------------------------------- detail ----
DETAIL_SUBSTR = (
    "_stairs", "_slab", "_fence", "_door", "_trapdoor", "_carpet", "_candle",
    "_pressure_plate", "flower_pot", "potted_", "_sign", "_banner", "painting",
    "item_frame", "armor_stand", "_head", "skull",
)


def is_detail(state):
    s = short(state)
    if any(t in s for t in DETAIL_SUBSTR):
        return True
    return False


# ---------------------------------------------------------------- special ----
# Blocks that suggest map-art / pixel canvases: uniform color blocks used at
# massive counts (wool/concrete alone at very high ratio).
MAPART_CANDIDATES = (
    "_wool", "_concrete", "_terracotta",
)


def is_mapart_candidate(state):
    s = short(state)
    return any(s.endswith(t) for t in MAPART_CANDIDATES)


def classify_volume(names, vol, counts=None):
    """Vectorized per-palette classification.

    names: list[str]; vol: ndarray uint8/uint16.
    Returns dict with counts and derived metrics.
    """
    import numpy as np
    idx = np.unique(vol)
    id_list = idx.tolist()
    cls = np.zeros(len(names), dtype=np.int8)  # 0 air,1 redstone,2 natural,3 building,4 other
    air_id = None
    for i in id_list:
        nm = names[i] if i < len(names) else "minecraft:unknown"
        b = bare(nm)
        sh = b.split(":", 1)[-1]
        if sh == "air" or b == "minecraft:cave_air" or b == "minecraft:void_air":
            cls[i] = 0
            air_id = i
        elif is_redstone(nm):
            cls[i] = 1
        elif is_natural(nm):
            cls[i] = 2
        elif is_building(nm):
            cls[i] = 3
        else:
            cls[i] = 4
    clsarr = cls[vol]
    total = vol.size
    nair = int((clsarr == 0).sum())
    nred = int((clsarr == 1).sum())
    nnat = int((clsarr == 2).sum())
    nbld = int((clsarr == 3).sum())
    noth = int((clsarr == 4).sum())
    nonair = total - nair
    # fluid machine signature: lava+water+redstone-ish component => factory machine
    has_lava = any(names[i].startswith("minecraft:lava") for i in id_list)
    has_water = any(names[i].startswith("minecraft:water") for i in id_list)
    machine_signature = bool(has_lava and has_water and nred > 0)
    return {
        "dims": [int(vol.shape[2]), int(vol.shape[0]), int(vol.shape[1])],  # w,h,l
        "total": total,
        "air": nair,
        "nonair": nonair,
        "redstone": nred,
        "natural": nnat,
        "building": nbld,
        "other": noth,
        "redstone_ratio": nred / max(1, nonair),
        "natural_ratio": nnat / max(1, nonair),
        "building_ratio": nbld / max(1, nonair),
        "machine_signature": machine_signature,
        "mapart_like": _mapart_metric(names, cls, vol),
    }


def _mapart_metric(names, cls, vol):
    """Fraction of nonair that is wool/concrete/terracotta color blocks."""
    import numpy as np
    idx = np.unique(vol)
    cnt = np.bincount(vol.ravel(), minlength=int(vol.max(initial=0)) + 1)
    tot = 0
    m = 0
    for i in idx.tolist():
        c = int(cnt[i])
        if i == 0:
            continue
        nm = names[i] if i < len(names) else ""
        tot += c
        if is_mapart_candidate(nm) and not is_redstone(nm):
            m += c
    return m / max(1, tot)
