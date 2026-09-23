import unittest
from prepare_v7 import COUNTS,ORDERS,order,make
class Sweep(unittest.TestCase):
 def test_counts(self):self.assertEqual(COUNTS[0],0);self.assertEqual(COUNTS[-1],256);self.assertEqual(sorted(set(COUNTS)),list(COUNTS))
 def test_order(self):
  for name in ORDERS:
   o=order('abc',name);self.assertEqual(len(o),256);self.assertEqual(set(o),set(range(256)))
   self.assertEqual(o,order('abc',name))
 def test_nested(self):
  for name in ORDERS:
   o=order('abc',name)
   for a,b in zip(COUNTS,COUNTS[1:]):self.assertTrue(set(o[:a])<set(o[:b]))
 def test_queue(self):
  obj=make();self.assertEqual(obj['expected_units'],len(obj['queue']));self.assertEqual(len({j['unit_key'] for j in obj['queue']}),228)
  for j in obj['queue']:
   if j['method']=='sweep':self.assertEqual(len(j['chunk_indices']),j['count'])
if __name__=='__main__':unittest.main(verbosity=2)
