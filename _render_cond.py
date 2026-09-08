"""生成 render_output (bld_XXXX.png 等距渲染) + metadata.csv.

用 TRELLIS OctreeRenderer 渲染 v2 GT 体素 (64^3), 固定等距相机.
方块颜色: 简化 palette (覆盖常见方块).
"""
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/home/linux/redstone_ai/TRELLIS')
import numpy as np
import torch
from PIL import Image
from trellis.representations.octree import DfsOctree as Octree
from trellis.renderers import OctreeRenderer


def extrinsics_look_at(eye, center, up):
    """world->camera 4x4 (替代 utils3d.torch.extrinsics_look_at)."""
    f = center - eye
    f = f / f.norm()
    r = torch.cross(f, up)
    r = r / r.norm()
    u = torch.cross(r, f)
    R = torch.stack([r, u, -f], dim=1)  # columns
    t = -R @ eye
    T = torch.eye(4, device=eye.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def intrinsics_from_fov_xy(fov_x, fov_y):
    """3x3 intrinsics (替代 utils3d.torch.intrinsics_from_fov_xy)."""
    fx = 1.0 / torch.tan(fov_x / 2)
    fy = 1.0 / torch.tan(fov_y / 2)
    return torch.tensor([[fx, 0, 0], [0, fy, 0], [0, 0, 1]], device=fov_x.device)

BASE = '/home/linux/redstone_ai'
DATA = f'{BASE}/dataset/v2/ss64.npy'
BLOCK_MAP = f'{BASE}/dataset/v2/block_map.npy'
OUT = f'{BASE}/render_output'
os.makedirs(OUT, exist_ok=True)
dev = 'cuda'

block_map = np.load(BLOCK_MAP)
vocab = [l.strip() for l in open(f'{BASE}/dataset/BLOCK_VOCAB_FULL.txt', encoding='utf-8') if l.strip()]
def block_name(nid):
    oi = int(block_map[nid])
    return vocab[oi] if 0 <= oi < len(vocab) else 'minecraft:stone'

# ---- 简化方块颜色 palette (常见方块) ----
PALETTE = {
    'minecraft:stone': (125, 125, 125), 'minecraft:deepslate': (80, 80, 85),
    'minecraft:cobblestone': (115, 115, 115), 'minecraft:stone_bricks': (130, 130, 130),
    'minecraft:mossy_stone_bricks': (110, 125, 110), 'minecraft:cracked_stone_bricks': (120, 120, 120),
    'minecraft:chiseled_stone_bricks': (135, 135, 135), 'minecraft:polished_deepslate': (75, 75, 80),
    'minecraft:deepslate_bricks': (85, 85, 90), 'minecraft:deepslate_tiles': (80, 80, 85),
    'minecraft:smooth_stone': (140, 140, 140), 'minecraft:granite': (150, 95, 90),
    'minecraft:polished_granite': (155, 100, 95), 'minecraft:diorite': (170, 170, 170),
    'minecraft:polished_diorite': (175, 175, 175), 'minecraft:andesite': (135, 135, 140),
    'minecraft:polished_andesite': (140, 140, 145), 'minecraft:obsidian': (20, 15, 25),
    'minecraft:crying_obsidian': (40, 20, 60), 'minecraft:bedrock': (30, 30, 30),
    'minecraft:basalt': (55, 55, 60), 'minecraft:polished_basalt': (60, 60, 65),
    'minecraft:tuff': (100, 100, 90), 'minecraft:calcite': (220, 220, 215),
    'minecraft:dripstone_block': (140, 110, 90), 'minecraft:sandstone': (215, 200, 160),
    'minecraft:chiseled_sandstone': (220, 205, 165), 'minecraft:cut_sandstone': (215, 200, 160),
    'minecraft:smooth_sandstone': (220, 205, 165), 'minecraft:red_sandstone': (180, 95, 60),
    'minecraft:chiseled_red_sandstone': (185, 100, 65), 'minecraft:cut_red_sandstone': (180, 95, 60),
    'minecraft:smooth_red_sandstone': (185, 100, 65), 'minecraft:bricks': (155, 90, 80),
    'minecraft:packed_mud': (130, 100, 80), 'minecraft:mud_bricks': (135, 105, 85),
    'minecraft:prismarine': (100, 170, 155), 'minecraft:prismarine_bricks': (105, 175, 160),
    'minecraft:dark_prismarine': (60, 110, 100), 'minecraft:sea_lantern': (200, 230, 220),
    'minecraft:quartz_block': (235, 230, 225), 'minecraft:quartz_bricks': (235, 230, 225),
    'minecraft:quartz_pillar': (235, 230, 225), 'minecraft:chiseled_quartz_block': (235, 230, 225),
    'minecraft:smooth_quartz': (235, 230, 225), 'minecraft:purpur_block': (170, 140, 170),
    'minecraft:purpur_pillar': (170, 140, 170), 'minecraft:end_stone': (215, 215, 165),
    'minecraft:end_stone_bricks': (220, 220, 170), 'minecraft:netherrack': (105, 55, 55),
    'minecraft:nether_bricks': (60, 30, 30), 'minecraft:cracked_nether_bricks': (65, 35, 35),
    'minecraft:chiseled_nether_bricks': (65, 35, 35), 'minecraft:red_nether_bricks': (105, 40, 40),
    'minecraft:polished_blackstone': (40, 38, 42), 'minecraft:polished_blackstone_bricks': (45, 43, 47),
    'minecraft:cracked_polished_blackstone_bricks': (50, 48, 52),
    'minecraft:gilded_blackstone': (45, 42, 45), 'minecraft:white_concrete': (207, 213, 214),
    'minecraft:orange_concrete': (224, 97, 1), 'minecraft:magenta_concrete': (169, 48, 159),
    'minecraft:light_blue_concrete': (36, 137, 199), 'minecraft:yellow_concrete': (241, 175, 21),
    'minecraft:lime_concrete': (94, 169, 24), 'minecraft:pink_concrete': (214, 101, 143),
    'minecraft:gray_concrete': (54, 57, 62), 'minecraft:light_gray_concrete': (125, 125, 115),
    'minecraft:cyan_concrete': (21, 119, 136), 'minecraft:purple_concrete': (100, 31, 156),
    'minecraft:blue_concrete': (45, 47, 143), 'minecraft:brown_concrete': (96, 60, 32),
    'minecraft:green_concrete': (73, 91, 36), 'minecraft:red_concrete': (142, 32, 32),
    'minecraft:black_concrete': (8, 10, 15),
    'minecraft:oak_planks': (162, 130, 80), 'minecraft:spruce_planks': (110, 83, 52),
    'minecraft:birch_planks': (195, 180, 140), 'minecraft:jungle_planks': (160, 120, 90),
    'minecraft:acacia_planks': (170, 95, 60), 'minecraft:dark_oak_planks': (70, 50, 35),
    'minecraft:mangrove_planks': (100, 55, 50), 'minecraft:cherry_planks': (215, 170, 170),
    'minecraft:bamboo_planks': (215, 195, 130), 'minecraft:crimson_planks': (100, 55, 70),
    'minecraft:warped_planks': (45, 90, 95),
    'minecraft:oak_log': (110, 85, 50), 'minecraft:spruce_log': (85, 62, 40),
    'minecraft:birch_log': (190, 180, 160), 'minecraft:dark_oak_log': (60, 42, 30),
    'minecraft:jungle_log': (130, 110, 70), 'minecraft:acacia_log': (120, 100, 80),
    'minecraft:cherry_log': (110, 80, 80), 'minecraft:mangrove_log': (95, 55, 50),
    'minecraft:oak_wood': (110, 85, 50), 'minecraft:spruce_wood': (85, 62, 40),
    'minecraft:birch_wood': (190, 180, 160), 'minecraft:dark_oak_wood': (60, 42, 30),
    'minecraft:stripped_oak_log': (190, 165, 110), 'minecraft:stripped_spruce_log': (150, 130, 95),
    'minecraft:stripped_birch_log': (205, 195, 170), 'minecraft:stripped_dark_oak_log': (115, 90, 60),
    'minecraft:stripped_cherry_log': (215, 160, 150), 'minecraft:stripped_cherry_wood': (215, 160, 150),
    'minecraft:stripped_acacia_wood': (175, 140, 95), 'minecraft:stripped_oak_wood': (190, 165, 110),
    'minecraft:stripped_spruce_wood': (150, 130, 95),
    'minecraft:glass': (220, 235, 240), 'minecraft:white_stained_glass': (235, 235, 235),
    'minecraft:black_stained_glass': (30, 30, 30), 'minecraft:light_blue_stained_glass': (150, 200, 230),
    'minecraft:cyan_stained_glass': (100, 190, 200), 'minecraft:glass_pane': (220, 235, 240),
    'minecraft:white_wool': (233, 236, 236), 'minecraft:light_gray_wool': (141, 144, 141),
    'minecraft:gray_wool': (62, 67, 67), 'minecraft:black_wool': (28, 28, 33),
    'minecraft:brown_wool': (130, 84, 50), 'minecraft:red_wool': (160, 39, 34),
    'minecraft:orange_wool': (235, 136, 68), 'minecraft:yellow_wool': (248, 197, 39),
    'minecraft:lime_wool': (112, 185, 25), 'minecraft:green_wool': (84, 109, 27),
    'minecraft:cyan_wool': (21, 137, 145), 'minecraft:light_blue_wool': (58, 175, 217),
    'minecraft:blue_wool': (51, 68, 170), 'minecraft:purple_wool': (122, 42, 173),
    'minecraft:magenta_wool': (188, 68, 190), 'minecraft:pink_wool': (237, 141, 172),
    'minecraft:dirt': (134, 96, 67), 'minecraft:grass_block': (91, 135, 58),
    'minecraft:grass': (85, 160, 60), 'minecraft:sand': (219, 207, 163),
    'minecraft:gravel': (130, 125, 120), 'minecraft:water': (50, 100, 200),
    'minecraft:oak_leaves': (70, 110, 55), 'minecraft:spruce_leaves': (55, 95, 65),
    'minecraft:birch_leaves': (90, 120, 65), 'minecraft:cherry_leaves': (235, 180, 190),
    'minecraft:dark_oak_leaves': (55, 75, 45), 'minecraft:acacia_leaves': (90, 130, 55),
    'minecraft:jungle_leaves': (65, 110, 60), 'minecraft:iron_block': (218, 218, 218),
    'minecraft:gold_block': (246, 205, 60), 'minecraft:diamond_block': (100, 220, 220),
    'minecraft:emerald_block': (40, 200, 100), 'minecraft:lapis_block': (30, 65, 160),
    'minecraft:redstone_block': (170, 20, 20), 'minecraft:coal_block': (25, 25, 25),
    'minecraft:copper_block': (180, 110, 80), 'minecraft:exposed_copper': (140, 120, 90),
    'minecraft:weathered_copper': (100, 140, 120), 'minecraft:oxidized_copper': (80, 140, 130),
    'minecraft:glowstone': (255, 220, 140), 'minecraft:sea_lantern': (210, 230, 220),
    'minecraft:shroomlight': (255, 170, 90), 'minecraft:ochre_froglight': (240, 190, 100),
    'minecraft:verdant_froglight': (190, 240, 150), 'minecraft:pearlescent_froglight': (240, 170, 220),
    'minecraft:bookshelf': (160, 120, 70), 'minecraft:lantern': (255, 200, 100),
    'minecraft:soul_lantern': (100, 160, 180), 'minecraft:torch': (255, 200, 100),
    'minecraft:wall_torch': (255, 200, 100), 'minecraft:chest': (170, 125, 70),
    'minecraft:white_terracotta': (210, 180, 160), 'minecraft:orange_terracotta': (160, 84, 45),
    'minecraft:magenta_terracotta': (150, 88, 110), 'minecraft:light_blue_terracotta': (113, 109, 138),
    'minecraft:yellow_terracotta': (186, 133, 35), 'minecraft:lime_terracotta': (104, 118, 53),
    'minecraft:pink_terracotta': (162, 78, 78), 'minecraft:gray_terracotta': (58, 42, 36),
    'minecraft:light_gray_terracotta': (135, 107, 98), 'minecraft:cyan_terracotta': (87, 92, 92),
    'minecraft:purple_terracotta': (118, 70, 86), 'minecraft:blue_terracotta': (74, 60, 91),
    'minecraft:brown_terracotta': (77, 51, 35), 'minecraft:green_terracotta': (76, 83, 42),
    'minecraft:red_terracotta': (143, 61, 46), 'minecraft:black_terracotta': (37, 23, 15),
    'minecraft:snow_block': (245, 250, 250), 'minecraft:ice': (170, 210, 240),
    'minecraft:packed_ice': (140, 180, 220), 'minecraft:blue_ice': (110, 160, 220),
    'minecraft:hay_block': (170, 140, 60), 'minecraft:bone_block': (220, 215, 190),
    'minecraft:terracotta': (152, 94, 67), 'minecraft:clay': (160, 170, 180),
    'minecraft:magma_block': (140, 80, 40), 'minecraft:mushroom_stem': (210, 205, 195),
    'minecraft:brown_mushroom_block': (150, 110, 80), 'minecraft:red_mushroom_block': (190, 70, 60),
    'minecraft:nether_wart_block': (90, 20, 30), 'minecraft:warped_wart_block': (30, 90, 80),
    'minecraft:netherite_block': (60, 55, 55), 'minecraft:ancient_debris': (80, 55, 50),
    'minecraft:target': (220, 200, 190), 'minecraft:barrel': (150, 115, 70),
    'minecraft:smithing_table': (60, 50, 45), 'minecraft:fletching_table': (170, 135, 90),
    'minecraft:cartography_table': (130, 95, 60), 'minecraft:composter': (130, 95, 60),
    'minecraft:loom': (140, 115, 90), 'minecraft:lectern': (145, 110, 75),
    'minecraft:grindstone': (100, 100, 100), 'minecraft:anvil': (70, 70, 70),
    'minecraft:iron_bars': (200, 200, 200), 'minecraft:chain': (140, 140, 140),
    'minecraft:ladder': (190, 155, 95), 'minecraft:end_rod': (230, 240, 245),
    'minecraft:lightning_rod': (200, 200, 200), 'minecraft:spawner': (40, 40, 50),
    'minecraft:nether_portal': (120, 30, 160), 'minecraft:melon': (110, 150, 45),
    'minecraft:pumpkin': (200, 120, 30), 'minecraft:cake': (240, 220, 200),
    'minecraft:stonecutter': (130, 130, 130), 'minecraft:bell': (220, 190, 60),
    'minecraft:conduit': (90, 130, 180), 'minecraft:beacon': (120, 220, 220),
    'minecraft:structure_block': (120, 120, 150), 'minecraft:command_block': (180, 140, 60),
    'minecraft:repeating_command_block': (120, 60, 180), 'minecraft:chain_command_block': (80, 150, 80),
    'minecraft:jukebox': (120, 85, 55), 'minecraft:enchanting_table': (120, 40, 40),
    'minecraft:end_portal_frame': (90, 120, 80), 'minecraft:ochre_froglight': (240, 190, 100),
    'minecraft:stripped_bamboo_block': (215, 195, 130), 'minecraft:bamboo_block': (160, 140, 90),
    'minecraft:sculk': (25, 40, 45), 'minecraft:sculk_sensor': (30, 60, 65),
    'minecraft:amethyst_block': (140, 100, 200), 'minecraft:budding_amethyst': (130, 90, 190),
    'minecraft:tinted_glass': (50, 50, 55), 'minecraft:obsidian': (20, 15, 25),
    'minecraft:crying_obsidian': (40, 20, 60), 'minecraft:respawn_anchor': (90, 40, 100),
    'minecraft:lodestone': (140, 140, 145), 'minecraft:white_bed': (235, 235, 235),
    'minecraft:red_bed': (190, 50, 40), 'minecraft:oak_fence': (162, 130, 80),
    'minecraft:spruce_fence': (110, 83, 52), 'minecraft:birch_fence': (195, 180, 140),
    'minecraft:dark_oak_fence': (70, 50, 35), 'minecraft:acacia_fence': (170, 95, 60),
    'minecraft:jungle_fence': (160, 120, 90), 'minecraft:cherry_fence': (215, 170, 170),
    'minecraft:mangrove_fence': (100, 55, 50), 'minecraft:nether_brick_fence': (60, 30, 30),
    'minecraft:polished_blackstone_wall': (40, 38, 42), 'minecraft:cobblestone_wall': (115, 115, 115),
    'minecraft:mossy_cobblestone_wall': (105, 120, 105), 'minecraft:stone_brick_wall': (130, 130, 130),
    'minecraft:mossy_stone_brick_wall': (110, 125, 110), 'minecraft:brick_wall': (155, 90, 80),
    'minecraft:sandstone_wall': (215, 200, 160), 'minecraft:red_sandstone_wall': (180, 95, 60),
    'minecraft:prismarine_wall': (100, 170, 155), 'minecraft:blackstone_wall': (50, 48, 52),
    'minecraft:end_stone_brick_wall': (220, 220, 170), 'minecraft:deepslate_brick_wall': (85, 85, 90),
    'minecraft:deepslate_tile_wall': (80, 80, 85), 'minecraft:granite_wall': (150, 95, 90),
    'minecraft:diorite_wall': (170, 170, 170), 'minecraft:andesite_wall': (135, 135, 140),
    'minecraft:nether_brick_stairs': (60, 30, 30), 'minecraft:red_nether_brick_stairs': (105, 40, 40),
    'minecraft:purpur_stairs': (170, 140, 170), 'minecraft:end_stone_brick_stairs': (220, 220, 170),
    'minecraft:blackstone_stairs': (50, 48, 52), 'minecraft:polished_blackstone_stairs': (40, 38, 42),
    'minecraft:polished_blackstone_brick_stairs': (45, 43, 47), 'minecraft:cobbled_deepslate': (85, 85, 90),
    'minecraft:polished_tuff': (105, 105, 95), 'minecraft:polished_tuff_stairs': (105, 105, 95),
    'minecraft:chiseled_tuff': (100, 100, 90), 'minecraft:chiseled_tuff_bricks': (100, 100, 90),
    'minecraft:tuff_bricks': (100, 100, 90), 'minecraft:oak_stairs': (162, 130, 80),
    'minecraft:spruce_stairs': (110, 83, 52), 'minecraft:birch_stairs': (195, 180, 140),
    'minecraft:dark_oak_stairs': (70, 50, 35), 'minecraft:acacia_stairs': (170, 95, 60),
    'minecraft:jungle_stairs': (160, 120, 90), 'minecraft:cherry_stairs': (215, 170, 170),
    'minecraft:mangrove_stairs': (100, 55, 50), 'minecraft:crimson_stairs': (100, 55, 70),
    'minecraft:warped_stairs': (45, 90, 95),
    'minecraft:oak_slab': (162, 130, 80), 'minecraft:spruce_slab': (110, 83, 52),
    'minecraft:birch_slab': (195, 180, 140), 'minecraft:dark_oak_slab': (70, 50, 35),
    'minecraft:acacia_slab': (170, 95, 60), 'minecraft:jungle_slab': (160, 120, 90),
    'minecraft:cherry_slab': (215, 170, 170), 'minecraft:mangrove_slab': (100, 55, 50),
    'minecraft:stone_slab': (125, 125, 125), 'minecraft:smooth_stone_slab': (140, 140, 140),
    'minecraft:stone_brick_slab': (130, 130, 130), 'minecraft:quartz_slab': (235, 230, 225),
    'minecraft:smooth_quartz_slab': (235, 230, 225), 'minecraft:purpur_slab': (170, 140, 170),
    'minecraft:end_stone_brick_slab': (220, 220, 170), 'minecraft:polished_blackstone_slab': (40, 38, 42),
    'minecraft:polished_blackstone_brick_slab': (45, 43, 47), 'minecraft:blackstone_slab': (50, 48, 52),
    'minecraft:sandstone_slab': (215, 200, 160), 'minecraft:red_sandstone_slab': (180, 95, 60),
    'minecraft:cut_sandstone_slab': (215, 200, 160), 'minecraft:cut_red_sandstone_slab': (180, 95, 60),
    'minecraft:polished_diorite_slab': (175, 175, 175), 'minecraft:polished_granite_slab': (155, 100, 95),
    'minecraft:polished_andesite_slab': (140, 140, 145), 'minecraft:prismarine_slab': (100, 170, 155),
    'minecraft:prismarine_brick_slab': (105, 175, 160), 'minecraft:dark_prismarine_slab': (60, 110, 100),
    'minecraft:polished_deepslate_slab': (75, 75, 80), 'minecraft:deepslate_brick_slab': (85, 85, 90),
    'minecraft:deepslate_tile_slab': (80, 80, 85), 'minecraft:cobbled_deepslate_slab': (85, 85, 90),
    'minecraft:crimson_slab': (100, 55, 70), 'minecraft:warped_slab': (45, 90, 95),
    'minecraft:bamboo_slab': (215, 195, 130), 'minecraft:bamboo_mosaic_slab': (215, 195, 130),
    'minecraft:polished_tuff_slab': (105, 105, 95), 'minecraft:tuff_slab': (100, 100, 90),
    'minecraft:tuff_brick_slab': (100, 100, 90), 'minecraft:stone_brick_stairs': (130, 130, 130),
    'minecraft:smooth_stone_stairs': (140, 140, 140), 'minecraft:granite_stairs': (150, 95, 90),
    'minecraft:diorite_stairs': (170, 170, 170), 'minecraft:andesite_stairs': (135, 135, 140),
    'minecraft:polished_granite_stairs': (155, 100, 95), 'minecraft:polished_diorite_stairs': (175, 175, 175),
    'minecraft:polished_andesite_stairs': (140, 140, 145), 'minecraft:quartz_stairs': (235, 230, 225),
    'minecraft:smooth_quartz_stairs': (235, 230, 225), 'minecraft:sandstone_stairs': (215, 200, 160),
    'minecraft:red_sandstone_stairs': (180, 95, 60), 'minecraft:cut_sandstone_stairs': (215, 200, 160),
    'minecraft:cut_red_sandstone_stairs': (180, 95, 60), 'minecraft:prismarine_stairs': (100, 170, 155),
    'minecraft:prismarine_brick_stairs': (105, 175, 160), 'minecraft:dark_prismarine_stairs': (60, 110, 100),
    'minecraft:polished_deepslate_stairs': (75, 75, 80), 'minecraft:deepslate_brick_stairs': (85, 85, 90),
    'minecraft:deepslate_tile_stairs': (80, 80, 85), 'minecraft:cobbled_deepslate_stairs': (85, 85, 90),
    'minecraft:mossy_cobblestone_stairs': (105, 120, 105), 'minecraft:mossy_stone_brick_stairs': (110, 125, 110),
    'minecraft:cobblestone_stairs': (115, 115, 115), 'minecraft:end_stone_brick_stairs': (220, 220, 170),
    'minecraft:purpur_stairs': (170, 140, 170), 'minecraft:spruce_trapdoor': (110, 83, 52),
    'minecraft:oak_trapdoor': (162, 130, 80), 'minecraft:dark_oak_trapdoor': (70, 50, 35),
    'minecraft:birch_trapdoor': (195, 180, 140), 'minecraft:acacia_trapdoor': (170, 95, 60),
    'minecraft:jungle_trapdoor': (160, 120, 90), 'minecraft:cherry_trapdoor': (215, 170, 170),
    'minecraft:mangrove_trapdoor': (100, 55, 50), 'minecraft:iron_trapdoor': (140, 140, 140),
    'minecraft:crimson_trapdoor': (100, 55, 70), 'minecraft:warped_trapdoor': (45, 90, 95),
    'minecraft:oak_door': (162, 130, 80), 'minecraft:spruce_door': (110, 83, 52),
    'minecraft:birch_door': (195, 180, 140), 'minecraft:dark_oak_door': (70, 50, 35),
    'minecraft:iron_door': (140, 140, 140),
    'minecraft:sea_pickle': (100, 180, 130), 'minecraft:turtle_egg': (220, 215, 205),
    'minecraft:flower_pot': (150, 80, 55), 'minecraft:scaffolding': (200, 170, 100),
    'minecraft:oak_sign': (162, 130, 80), 'minecraft:spruce_sign': (110, 83, 52),
    'minecraft:birch_sign': (195, 180, 140), 'minecraft:dark_oak_sign': (70, 50, 35),
    'minecraft:acacia_sign': (170, 95, 60), 'minecraft:jungle_sign': (160, 120, 90),
    'minecraft:cherry_sign': (215, 170, 170), 'minecraft:mangrove_sign': (100, 55, 50),
    'minecraft:oak_wall_sign': (162, 130, 80), 'minecraft:spruce_wall_sign': (110, 83, 52),
    'minecraft:birch_wall_sign': (195, 180, 140), 'minecraft:dark_oak_wall_sign': (70, 50, 35),
    'minecraft:acacia_wall_sign': (170, 95, 60), 'minecraft:jungle_wall_sign': (160, 120, 90),
    'minecraft:cherry_wall_sign': (215, 170, 170), 'minecraft:mangrove_wall_sign': (100, 55, 50),
    'minecraft:white_banner': (235, 235, 235), 'minecraft:orange_banner': (235, 136, 68),
    'minecraft:magenta_banner': (188, 68, 190), 'minecraft:light_blue_banner': (58, 175, 217),
    'minecraft:yellow_banner': (248, 197, 39), 'minecraft:lime_banner': (112, 185, 25),
    'minecraft:pink_banner': (237, 141, 172), 'minecraft:gray_banner': (62, 67, 67),
    'minecraft:light_gray_banner': (141, 144, 141), 'minecraft:cyan_banner': (21, 137, 145),
    'minecraft:purple_banner': (122, 42, 173), 'minecraft:blue_banner': (51, 68, 170),
    'minecraft:brown_banner': (130, 84, 50), 'minecraft:green_banner': (84, 109, 27),
    'minecraft:red_banner': (160, 39, 34), 'minecraft:black_banner': (28, 28, 33),
}
DEFAULT_COLOR = (128, 128, 128)

def color_for_name(nm):
    if nm in PALETTE:
        return PALETTE[nm]
    # 尝试移除 _stairs/_slab/_trapdoor/_fence 等后缀找基础色
    for suf in ['_stairs', '_slab', '_trapdoor', '_fence', '_door', '_wall', '_sign', '_banner', '_fence_gate']:
        base = nm.replace(suf, '')
        if base in PALETTE:
            return PALETTE[base]
    return DEFAULT_COLOR

def main():
    dev = 'cuda'
    ss = np.load(DATA, mmap_mode='r')
    N = ss.shape[0]
    print(f'rendering {N} samples...', flush=True)

    renderer = OctreeRenderer()
    renderer.rendering_options.resolution = 512
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0.92, 0.92, 0.9)
    renderer.rendering_options.ssaa = 2
    renderer.pipe.primitive = 'voxel'

    yaw = np.deg2rad(-60)
    pitch = np.deg2rad(25)
    dist = 1.8
    orig = torch.tensor([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)]).float().cuda() * dist
    ext = extrinsics_look_at(orig, torch.zeros(3).cuda(), torch.tensor([0, 0, 1]).float().cuda())
    intr = intrinsics_from_fov_xy(torch.deg2rad(torch.tensor(30)).cuda(), torch.deg2rad(torch.tensor(30)).cuda())

    import glob
    done = set(int(os.path.basename(p)[4:-4]) for p in glob.glob(os.path.join(OUT, 'bld_*.png')))
    for i in range(N):
        if i in done:
            continue
        v = np.asarray(ss[i])
        coords = torch.nonzero(torch.from_numpy(v > 0))
        if len(coords) == 0:
            continue
        oct = Octree(depth=10, aabb=[-0.5, -0.5, -0.5, 1, 1, 1], device=dev, primitive='voxel',
                     sh_degree=0, primitive_config={'solid': True})
        oct.position = coords.float().cuda() / 64.0
        oct.depth = torch.full((coords.shape[0], 1), 6, dtype=torch.uint8, device=dev)
        labels = v[v > 0]
        colors = np.zeros((len(labels), 3), dtype=np.float32)
        for c in np.unique(labels):
            rgb = color_for_name(block_name(int(c)))
            colors[labels == c] = np.array(rgb, dtype=np.float32) / 255.0
        with torch.no_grad():
            out = renderer.render(oct, ext, intr, colors_overwrite=torch.from_numpy(colors).float().cuda())
        img = out['color'].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(OUT, f'bld_{i:04d}.png'))
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{N}', flush=True)
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
