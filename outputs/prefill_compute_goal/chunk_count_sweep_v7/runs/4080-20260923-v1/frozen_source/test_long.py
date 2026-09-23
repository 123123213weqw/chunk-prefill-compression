import unittest,torch
from test_engine import model,segments
from engine import Engine
class TestLong(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=model()
 def run_engine(self,spec,block=64):
  e=Engine(self.m,spec,record=True);s=e.prefill(e.new_state(),segments(),block);s.validate();return e,s
 def test_one_chunk_shape(self):
  e,s=self.run_engine({'kind':'fixed','depth':2,'keep':32,'only_chunk':0});self.assertEqual([len(x) for x in s.positions],[135,135,103,103]);self.assertEqual([r['keep'] for r in e.routes],[32,64])
 def test_restore_complement(self):
  a,sa=self.run_engine({'kind':'fixed','depth':2,'keep':48,'only_chunk':1})
  b,sb=self.run_engine({'kind':'fixed','depth':2,'keep':48,'restore_chunk':0})
  torch.testing.assert_close(a.forward(sa,[4,5],'Q'),b.forward(sb,[4,5],'Q'));self.assertEqual([r['keep'] for r in b.routes],[64,48])
 def test_partition(self):
  for key in ('only_chunk','restore_chunk'):
   spec={'kind':'fixed','depth':2,'keep':32,key:1};a,sa=self.run_engine(spec,64);b,sb=self.run_engine(spec,128)
   torch.testing.assert_close(a.forward(sa,[4,5],'Q'),b.forward(sb,[4,5],'Q'),atol=1e-6,rtol=1e-5)
 def test_endpoint(self):
  e,s=self.run_engine({'kind':'fixed','depth':2,'keep':48,'merge':'endpoint'});self.assertEqual([len(x) for x in s.positions],[135,135,103,103]);self.assertTrue(torch.isfinite(e.forward(s,[5,7],'Q')).all())
 def test_sibling_chunk_untouched(self):
  a,sa=self.run_engine({'kind':'fixed','depth':2,'keep':32,'only_chunk':1});b,sb=self.run_engine({'kind':'identity'})
  for x,y in zip(sa.cache.layers,sb.cache.layers):torch.testing.assert_close(x.keys[:,:,:68],y.keys[:,:,:68],atol=1e-6,rtol=1e-5)
if __name__=='__main__':unittest.main(verbosity=2)
