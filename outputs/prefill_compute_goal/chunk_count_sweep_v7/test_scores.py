import unittest,torch
from test_engine import model,segments
from engine import Engine
from core import likelihood
class Scores(unittest.TestCase):
 def test_teacher_likelihood_and_restore(self):
  m=model();e=Engine(m,{'kind':'native'});seg=segments();s=e.prefill(e.new_state(),seg,64);cp=s.checkpoint();q=[21,23,25];target=[11,13,17,19]
  got=likelihood(e,s,q,target);self.assertEqual(s.checkpoint(),cp)
  allids=seg['S']+seg['T']+seg['R']+q+target[:-1];start=len(allids)-len(target)
  with torch.inference_mode():
   logits=m(torch.tensor([allids]),use_cache=False).logits[0];expected=sum(float(logits[start+j].float().log_softmax(-1)[t]) for j,t in enumerate(target))
  self.assertAlmostEqual(got['sum_logp'],expected,places=5)
if __name__=='__main__':unittest.main(verbosity=2)
