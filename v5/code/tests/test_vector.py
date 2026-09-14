import unittest
from mcfix.containers import _vec,DecodeError
class CoordinateTests(unittest.TestCase):
    def test_invalid_offset_is_rejected_at_decode_boundary(self):
        for bad in [[],[1,2],[1,2,3,4],{'x':1,'y':2},[0,.5,0]]:
            with self.assertRaisesRegex(DecodeError,'3D integer'):_vec(bad)
        self.assertEqual(_vec([1,-2,3]),(1,-2,3))
        self.assertEqual(_vec({'x':1,'y':-2,'z':3}),(1,-2,3))
