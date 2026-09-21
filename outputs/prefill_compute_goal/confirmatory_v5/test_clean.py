import unittest,torch
from test_engine import model,segments
from clean import native_prefix
from engine import Engine
from core import shape
from base_engine import ShapeAudit
class Clean(unittest.TestCase):
 def test_native_prefix(self):
  m=model();e=Engine(m,{'kind':'native'});s=e.prefill(e.new_state(),segments(),64);cache=native_prefix(m,segments(),64)
  for a,b in zip(cache.layers,s.cache.layers):torch.testing.assert_close(a.keys,b.keys);torch.testing.assert_close(a.values,b.values)
 def test_record_off_equivalence_and_shape_audit(self):
  m=model()
  for spec in ({'kind':'fixed','depth':3,'keep':16},{'kind':'fixed','depth':2,'keep':32,'merge':'endpoint'}):
   a=Engine(m,spec,record=True);h=ShapeAudit(m)
   try:sa=a.prefill(a.new_state(),segments(),64)
   finally:h.close()
   shape(a,sa,h,segments());b=Engine(m,spec,record=False);sb=b.prefill(b.new_state(),segments(),64)
   self.assertFalse(b.routes or b.features or b.attn_rows)
   torch.testing.assert_close(a.forward(sa,[4,7],'Q'),b.forward(sb,[4,7],'Q'))
 def test_block_and_four_token_pool(self):
  m=model();e=Engine(m,{'kind':'fixed','depth':3,'keep':16})
  a=e.prefill(e.new_state(),segments(),64);b=e.prefill(e.new_state(),segments(),128)
  torch.testing.assert_close(e.forward(a,[5,7],'Q'),e.forward(b,[5,7],'Q'),atol=1e-6,rtol=1e-5)
if __name__=='__main__':unittest.main(verbosity=2)
