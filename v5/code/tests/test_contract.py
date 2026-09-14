import tempfile
import unittest
from pathlib import Path
import numpy as np
import nbtlib as n
from mcfix.codec import registry,rotate_state,rotate_volume,parse_state
from mcfix.containers import unpack_longs,pack_longs,varint_decode,export_litematic,decode_path,DecodeError


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.r=registry()
    def test_all_registered_states_roundtrip_and_four_rotations(self):
        for sid in self.r.names:
            self.assertEqual(self.r.encode(self.r.string(sid),fill_defaults=False),sid)
        lut=self.r.rotation[1]
        ids=np.arange(len(lut));np.testing.assert_array_equal(lut[lut[lut[lut[ids]]]],ids)
    def test_independent_boolean_states_do_not_collide(self):
        a=self.r.encode('oak_trapdoor[powered=false,open=true]')
        b=self.r.encode('oak_trapdoor[powered=false,open=false]')
        self.assertNotEqual(a,b)
        self.assertEqual(self.r.properties[a]['open'],'true')
    def test_water_wall_and_double_slab_preserved(self):
        for a,b in [('oak_slab[type=double]','oak_slab[type=bottom]'),('oak_stairs[waterlogged=true]','oak_stairs[waterlogged=false]'),('cobblestone_wall[north=low]','cobblestone_wall[north=tall]')]:
            self.assertNotEqual(self.r.encode(a),self.r.encode(b))
    def test_split_identity_ignores_palette_noise_and_state_variants(self):
        from mcfix.data import semantic_hash
        a=np.zeros((3,3,3),int);a[1,1,1]=1
        self.assertEqual(semantic_hash(a,['minecraft:air','minecraft:oak_log[axis=x]','minecraft:stone']),semantic_hash(a,['minecraft:air','minecraft:oak_log[axis=y]']))
    def test_version_aware_grass(self):
        old,_=self.r.normalize_source('minecraft:grass',data_version=1343)
        new,_=self.r.normalize_source('minecraft:grass',data_version=2975)
        self.assertEqual(self.r.names[old],'minecraft:grass_block')
        self.assertEqual(self.r.names[new],'minecraft:short_grass')
        with self.assertRaises(ValueError):self.r.normalize_source('minecraft:grass')
    def test_rotation_probe(self):
        a=np.zeros((7,3,7),int);a[3,1,2]=1
        b=rotate_volume(a)
        self.assertEqual(b[4,1,3],1)
        self.assertEqual(parse_state(rotate_state('oak_stairs[facing=north]'))[1]['facing'],'east')
        self.assertEqual(parse_state(rotate_state('oak_log[axis=x]'))[1]['axis'],'z')
        self.assertEqual(parse_state(rotate_state('rail[shape=north_east]'))[1]['shape'],'south_east')
    def test_invalid_properties_rejected(self):
        for state in ['stone[north=false]','oak_log[facing=east]','oak_slab[half=top]','oak_stairs[facing=up]','stone[x=1,x=2]']:
            with self.assertRaises(ValueError):self.r.encode(state)
    def test_legacy_adapter_cannot_create_illegal_properties(self):
        for name in ['stone','oak_log','oak_slab','oak_trapdoor','oak_stairs','mushroom_stem']:
            for v in range(6048):
                sid=self.r.legacy_to_state('minecraft:'+name,v)
                self.assertEqual(self.r.encode(self.r.string(sid)),sid)
    def test_bitstream_cross_words_and_nonbyte_rows(self):
        for bits in [2,3,5,9,12,17]:
            a=np.random.default_rng(bits).integers(0,1<<bits,262147,dtype=np.uint32)
            np.testing.assert_array_equal(unpack_longs(pack_longs(a,bits),len(a),bits),a)
        with self.assertRaises(DecodeError):unpack_longs([],7,3)
    def test_varint_values_and_errors(self):
        raw=bytes([0,127,128,1,255,1,128,128,1])
        np.testing.assert_array_equal(varint_decode(raw),[0,127,128,255,16384])
        for raw in [bytes([128]),bytes([255]*6+[0]),bytes([255]*4+[127])]:
            with self.assertRaises(DecodeError):varint_decode(raw)
    def test_more_than_256_states_and_air_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            palette=[self.r.string(s) for s in range(1,602)] # air deliberately absent
            xyz=np.arange(601).reshape(601,1,1)
            p=Path(d)/'x.litematic';export_litematic(p,xyz,palette)
            decoded=decode_path(p)
            actual=np.array(decoded['names'])[decoded['vol'].transpose(2,0,1)]
            np.testing.assert_array_equal(actual,np.array(palette)[xyz])
    def test_all_registry_states_survive_file_export(self):
        with tempfile.TemporaryDirectory() as d:
            palette=[self.r.string(i) for i in range(len(self.r.names))]
            a=np.arange(len(palette)).reshape(-1,1,1);p=Path(d)/'all.litematic'
            export_litematic(p,a,palette);out=decode_path(p)
            self.assertEqual(out['meta']['data_version'],4189)
            actual=np.array(out['names'],object)[out['vol'].transpose(2,0,1)]
            np.testing.assert_array_equal(actual,np.array(palette,object)[a])
    def test_multiple_regions_negative_size(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.litematic';export_litematic(p,np.ones((3,2,4),int),['minecraft:air','minecraft:stone'])
            root=n.load(p);r=root['Regions']['main'];r['Size']['x']=n.Int(-3);r['Position']['x']=n.Int(7)
            import copy
            r2=copy.deepcopy(r);r2['Position']['x']=n.Int(-5);root['Regions']['second']=r2;root.save(p,gzipped=True)
            out=decode_path(p)
            self.assertEqual(out['dims'],(15,2,4));self.assertEqual(out['meta']['origin_xyz'][0],-7)
            self.assertEqual(np.count_nonzero(out['vol']),48)
    def test_sponge3_varints(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.schem'
            root=n.File({'Schematic':n.Compound({'Version':n.Int(3),'DataVersion':n.Int(4189),'Width':n.Short(2),'Height':n.Short(1),'Length':n.Short(1),
              'Blocks':n.Compound({'Palette':n.Compound({'minecraft:air':n.Int(0),'minecraft:oak_log[axis=x]':n.Int(128)}),
                                 'Data':n.ByteArray(np.array([0,128,1],np.uint8).view(np.int8))})})})
            root.save(p,gzipped=True);out=decode_path(p);self.assertEqual(out['names'][out['vol'][0,0,1]],'minecraft:oak_log[axis=x]')
    def test_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.schematic';root=n.File({'Width':n.Short(2),'Height':n.Short(1),'Length':n.Short(1),'Blocks':n.ByteArray([17,17]),'Data':n.ByteArray([4,8])})
            root.save(p,gzipped=True);out=decode_path(p);self.assertIn('axis=x',out['names'][out['vol'][0,0,0]]);self.assertIn('axis=z',out['names'][out['vol'][0,0,1]])
    def test_structure_air_not_palette_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.nbt';root=n.File({'size':n.List[n.Int]([2,1,1]),'palette':n.List[n.Compound]([n.Compound({'Name':n.String('minecraft:stone')})]),
                'blocks':n.List[n.Compound]([n.Compound({'pos':n.List[n.Int]([1,0,0]),'state':n.Int(0)})])})
            root.save(p,gzipped=True);out=decode_path(p);self.assertEqual(out['names'][out['vol'][0,0,0]],'minecraft:air');self.assertEqual(out['names'][out['vol'][0,0,1]],'minecraft:stone')

if __name__=='__main__':unittest.main()
