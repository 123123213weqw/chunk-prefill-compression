"""Small random Qwen CPU correctness tests, NOT task accuracy or GPU performance."""
import hashlib,json,unittest
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from engine import Engine,ShapeAudit,shrink

torch.set_num_threads(2)

def model():
    torch.manual_seed(20260908)
    config=Qwen3Config(vocab_size=127,hidden_size=64,intermediate_size=96,num_hidden_layers=4,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=4096,
                      attention_dropout=0.,tie_word_embeddings=True)
    config._attn_implementation='sdpa'
    return Qwen3ForCausalLM(config).eval()

def segments():return {'S':[1,2,3,4],'T':[5+i%110 for i in range(128)],'R':[9,4,2]}

def cache_tensors(s):return [(l.keys.clone(),l.values.clone()) for l in s.cache.layers]

class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=model()
    def test_01_identity_native_prefill_decode(self):
        a=Engine(self.m,split_depth=2,native=True);b=Engine(self.m,split_depth=2)
        sa=a.prefill(a.new_state(),segments(),64);sb=b.prefill(b.new_state(),segments(),64)
        sa.validate();sb.validate()
        for la,lb in zip(sa.cache.layers,sb.cache.layers):
            torch.testing.assert_close(la.keys,lb.keys,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(la.values,lb.values,atol=1e-6,rtol=1e-5)
        for q,stage in [([20,22,24],'Q'),([4],'decode'),([5],'decode')]:
            torch.testing.assert_close(a.forward(sa,q,stage),b.forward(sb,q,stage),atol=1e-6,rtol=1e-5)
    def test_02_shrink_means_and_positions(self):
        h=torch.arange(128.).reshape(1,64,2);p=torch.arange(50,114)
        x,pos=shrink(h,p,'pair_mean');torch.testing.assert_close(x,(h[:,0::2]+h[:,1::2])/2)
        assert torch.equal(pos,p[1::2]);assert x.shape==(1,32,2)
        end,_=shrink(h,p,'endpoint');assert torch.equal(end,h[:,1::2])
    def test_03_variable_cache_and_shape_reduction(self):
        for method in ['pair_mean','endpoint']:
            e=Engine(self.m,method,2);audit=ShapeAudit(self.m)
            try:s=e.prefill(e.new_state(),segments(),64)
            finally:audit.close()
            s.validate();self.assertEqual(s.logical_next,135)
            self.assertEqual([len(p) for p in s.positions],[135,135,71,71])
            for i in range(4):
                for op in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
                    self.assertEqual(sum(r['tokens'] for r in audit.rows if r['layer']==i and r['op']==op),135 if i<2 else 71)
            logits=e.forward(s,[44,55],'Q');self.assertTrue(torch.isfinite(logits).all())
            self.assertEqual(s.logical_next,137);self.assertEqual([len(p) for p in s.positions],[137,137,73,73])
            self.assertEqual(s.positions[2][-1].item(),136)
    def test_04_causal_future_perturbation(self):
        for method in ['identity','pair_mean','endpoint']:
            e=Engine(self.m,method,2);a=segments();b={k:list(v) for k,v in a.items()};b['T'][64:]=[6]*64
            sa=e.new_state();sb=e.new_state()
            for state,part in [(sa,a),(sb,b)]:
                e.forward(state,part['S'],'S',want_logits=False);e.forward(state,part['T'],'T',want_logits=False)
            for i,(ka,kb) in enumerate(zip(sa.cache.layers,sb.cache.layers)):
                n=68 if i<2 or method=='identity' else 36
                torch.testing.assert_close(ka.keys[:,:,:n],kb.keys[:,:,:n],atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(ka.values[:,:,:n],kb.values[:,:,:n],atol=1e-6,rtol=1e-5)
    def test_05_block_partition_equivalence(self):
        for method in ['identity','pair_mean','endpoint']:
            e=Engine(self.m,method,2);a=e.prefill(e.new_state(),segments(),64);b=e.prefill(e.new_state(),segments(),128)
            torch.testing.assert_close(e.forward(a,[21,34],'Q'),e.forward(b,[21,34],'Q'),atol=1e-6,rtol=1e-5)
    def test_06_probe_order_and_restore(self):
        for method in ['identity','pair_mean','endpoint']:
            e=Engine(self.m,method,2);s=e.prefill(e.new_state(),segments(),64);cp=s.checkpoint();before=cache_tensors(s)
            qa=e.answer(s,[21,34],5,set());qb=e.answer(s,[31,44],5,set())
            self.assertEqual(qb,e.answer(s,[31,44],5,set()));self.assertEqual(qa,e.answer(s,[21,34],5,set()))
            self.assertEqual(cp,s.checkpoint());s.validate()
            for old,l in zip(before,s.cache.layers):self.assertTrue(torch.equal(old[0],l.keys) and torch.equal(old[1],l.values))
    def test_07_wrong_position_rejected(self):
        e=Engine(self.m,'pair_mean',2);s=e.prefill(e.new_state(),segments(),64)
        with self.assertRaises(ValueError):e.forward(s,[1],'Q',logical_start=71)
        with self.assertRaises(ValueError):e.forward(s,[1]*63,'T')
        with self.assertRaises(ValueError):e.prefill(e.new_state(),segments(),65)
    def test_08_answer_exception_restore(self):
        e=Engine(self.m,'pair_mean',2);s=e.prefill(e.new_state(),segments(),64);cp=s.checkpoint()
        original=e.forward
        def faulty(state,ids,stage,**kwargs):
            x=original(state,ids,stage,**kwargs)
            if stage=='decode':raise RuntimeError('injected after append')
            return x
        e.forward=faulty
        with self.assertRaises(RuntimeError):e.answer(s,[31],3,set())
        self.assertEqual(cp,s.checkpoint());s.validate()

    def test_09_partial_layer_failure_restore(self):
        e=Engine(self.m,'pair_mean',2);s=e.prefill(e.new_state(),segments(),64);cp=s.checkpoint();before=cache_tensors(s)
        def fail_after_kv_update(module,inputs):raise RuntimeError('injected within decoder layer')
        handle=self.m.model.layers[2].self_attn.o_proj.register_forward_pre_hook(fail_after_kv_update)
        try:
            with self.assertRaises(RuntimeError):e.answer(s,[31],3,set())
        finally:handle.remove()
        self.assertEqual(cp,s.checkpoint());s.validate()
        for old,l in zip(before,s.cache.layers):self.assertTrue(torch.equal(old[0],l.keys) and torch.equal(old[1],l.values))

    def test_10_native_mask_policy_equivalence(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original=ALL_ATTENTION_FUNCTIONS['sdpa'];captured=[]
        def observed(module,q,k,v,mask,**kwargs):
            captured.append((module.layer_idx,None if mask is None else mask.clone()))
            return original(module,q,k,v,mask,**kwargs)
        ALL_ATTENTION_FUNCTIONS.register('sdpa',observed)
        try:
            runs=[]
            for native in (True,False):
                captured.clear();e=Engine(self.m,split_depth=2,native=native);s=e.new_state()
                for ids,stage in [([1,2,3],'S'),([5]*64,'T'),([6,7,8],'Q'),([9],'decode')]:e.forward(s,ids,stage)
                runs.append(captured.copy())
            self.assertEqual(len(runs[0]),len(runs[1]))
            for (ia,a),(ib,b) in zip(*runs):
                self.assertEqual(ia,ib)
                if a is None:self.assertIsNone(b)
                else:self.assertEqual(a.dtype,b.dtype);self.assertTrue(torch.equal(a,b))
        finally:ALL_ATTENTION_FUNCTIONS.register('sdpa',original)

if __name__=='__main__':unittest.main(verbosity=2)
