import unittest,torch
from test_engine import model,segments
from engine import Engine
from legacy_engine import Engine as Legacy
class Diagnostic(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=model()
 def run_e(self,spec,block=64,klass=Engine):
  e=klass(self.m,spec,record=True);s=e.prefill(e.new_state(),segments(),block);s.validate();return e,s
 def test_no_override_matches_legacy(self):
  spec={'kind':'fixed','depth':2,'keep':16};e,s=self.run_e(spec);f,t=self.run_e(spec,klass=Legacy)
  torch.testing.assert_close(e.forward(s,[4,5],'Q'),f.forward(t,[4,5],'Q'),atol=0,rtol=0)
 def test_keep64_equals_restore(self):
  spec={'kind':'fixed','depth':2,'keep':16};e,s=self.run_e({**spec,'chunk_keeps':{'1':64}});f,t=self.run_e({**spec,'restore_chunk':1},klass=Legacy)
  self.assertEqual([r['keep'] for r in e.routes],[16,64]);torch.testing.assert_close(e.forward(s,[4,5],'Q'),f.forward(t,[4,5],'Q'),atol=0,rtol=0)
 def test_keep32_mixed_lengths(self):
  e,s=self.run_e({'kind':'fixed','depth':2,'keep':16,'chunk_keeps':{'1':32}})
  self.assertEqual([r['keep'] for r in e.routes],[16,32]);self.assertEqual([len(p) for p in s.positions],[135,135,55,55])
 def test_partition_and_positions(self):
  spec={'kind':'fixed','depth':2,'keep':16,'chunk_keeps':{'0':64,'1':32}};e,s=self.run_e(spec);f,t=self.run_e(spec,128)
  for a,b in zip(s.positions,t.positions):torch.testing.assert_close(a,b)
  torch.testing.assert_close(e.forward(s,[4,5],'Q'),f.forward(t,[4,5],'Q'),atol=1e-6,rtol=1e-5)
 def test_all_restore_equals_identity(self):
  e,s=self.run_e({'kind':'fixed','depth':2,'keep':16,'chunk_keeps':{'0':64,'1':64}});f,t=self.run_e({'kind':'identity'})
  torch.testing.assert_close(e.forward(s,[4,5],'Q'),f.forward(t,[4,5],'Q'),atol=0,rtol=0)
 def test_future_override_does_not_touch_previous_chunk(self):
  spec={'kind':'fixed','depth':2,'keep':16};e,s=self.run_e({**spec,'chunk_keeps':{'1':32}});f,t=self.run_e(spec)
  for i,(a,b) in enumerate(zip(s.cache.layers,t.cache.layers)):
   n=68 if i<2 else 20
   torch.testing.assert_close(a.keys[:,:,:n],b.keys[:,:,:n],atol=0,rtol=0)
 def test_invalid(self):
  with self.assertRaises(ValueError):Engine(self.m,{'kind':'fixed','depth':2,'keep':16,'chunk_keeps':{'1':0}})
if __name__=='__main__':unittest.main(verbosity=2)
