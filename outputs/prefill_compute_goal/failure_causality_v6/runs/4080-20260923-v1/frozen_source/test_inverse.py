import unittest,torch
from test_engine import model,segments
from engine import Engine
from legacy_engine import Engine as Legacy
class Inverse(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=model()
 def run_e(self,spec,block=64,klass=Engine):
  e=klass(self.m,spec,record=True);s=e.prefill(e.new_state(),segments(),block);s.validate();return e,s
 def test_default64_is_identity(self):
  a,x=self.run_e({'kind':'fixed','depth':2,'keep':64});b,y=self.run_e({'kind':'identity'},klass=Legacy)
  torch.testing.assert_close(a.forward(x,[3,5],'Q'),b.forward(y,[3,5],'Q'),atol=0,rtol=0)
  for i,j in zip(x.positions,y.positions):torch.testing.assert_close(i,j)
 def test_only_chunk16_matches_legacy(self):
  a,x=self.run_e({'kind':'fixed','depth':2,'keep':64,'chunk_keeps':{'1':16}});b,y=self.run_e({'kind':'fixed','depth':2,'keep':16,'only_chunk':1},klass=Legacy)
  self.assertEqual([r['keep'] for r in a.routes],[64,16]);torch.testing.assert_close(a.forward(x,[3,5],'Q'),b.forward(y,[3,5],'Q'),atol=0,rtol=0)
 def test_partition(self):
  spec={'kind':'fixed','depth':2,'keep':64,'chunk_keeps':{'1':16}};a,x=self.run_e(spec);b,y=self.run_e(spec,128)
  for i,j in zip(x.positions,y.positions):torch.testing.assert_close(i,j)
  torch.testing.assert_close(a.forward(x,[3,5],'Q'),b.forward(y,[3,5],'Q'),atol=1e-6,rtol=1e-5)
 def test_equal_budget(self):
  for idx in ('0','1'):
   a,x=self.run_e({'kind':'fixed','depth':2,'keep':64,'chunk_keeps':{idx:16}})
   self.assertEqual(sum((4-r['merge_depth'])*(64-r['keep']) for r in a.routes),96)
if __name__=='__main__':unittest.main(verbosity=2)
