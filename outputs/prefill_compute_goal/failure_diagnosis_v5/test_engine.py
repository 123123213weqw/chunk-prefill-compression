import unittest,torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from engine import Engine,ShapeAudit,pool,score,plans
from base_engine import shrink

torch.set_num_threads(2)
def model():
    torch.manual_seed(20260920)
    c=Qwen3Config(vocab_size=127,hidden_size=64,intermediate_size=96,num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=4096)
    c._attn_implementation='sdpa';return Qwen3ForCausalLM(c).eval()
def segments():return {'S':[1,2,3,4],'T':[5+i%110 for i in range(128)],'R':[9,4,2]}
class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=model()
    def test_identity(self):
        a=Engine(self.m,{'kind':'native'});b=Engine(self.m)
        sa=a.prefill(a.new_state(),segments(),64);sb=b.prefill(b.new_state(),segments(),64)
        for x,y in zip(sa.cache.layers,sb.cache.layers):
            torch.testing.assert_close(x.keys,y.keys,atol=1e-6,rtol=1e-5);torch.testing.assert_close(x.values,y.values,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(a.forward(sa,[3,5,7],'Q'),b.forward(sb,[3,5,7],'Q'),atol=1e-6,rtol=1e-5)
    def test_pool(self):
        h=torch.randn(2,64,12);p=torch.arange(64)
        for k,old in [(32,'pair_mean'),(48,'pair_mean_48')]:
            a,_=shrink(h,p,old);torch.testing.assert_close(a,pool(h,k))
        self.assertEqual(float(score(torch.ones(1,64,8),32)),0.)
    def test_all_shapes(self):
        specs=list(plans(4)[0].values())+[{'kind':k} for k in ('adaptive_equal','adaptive_safe','random_equal')]
        for spec in specs:
            e=Engine(self.m,spec,{'early_median':.5,'early_q25':.1,'late_q25':.5},record=True);audit=ShapeAudit(self.m)
            try:s=e.prefill(e.new_state(),segments(),64)
            finally:audit.close()
            s.validate()
            for i,p in enumerate(s.positions):
                n=7+sum(64 if i<r['merge_depth'] else r['keep'] for r in e.routes)
                self.assertEqual(len(p),n)
                for op in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'):
                    self.assertEqual(sum(r['tokens'] for r in audit.rows if r['layer']==i and r['op']==op),n)
            cp=s.checkpoint();a=e.answer(s,[3,5],4,set());self.assertEqual(cp,s.checkpoint());self.assertEqual(a,e.answer(s,[3,5],4,set()))
    def test_equal_budget(self):
        for l in (4,28,36):
            _,a,b=plans(l);self.assertEqual((l-a)*16,(l-b)*32)
        for kind in ('adaptive_equal','random_equal'):
            e=Engine(self.m,{'kind':kind},{'early_median':.5},record=True);e.prefill(e.new_state(),segments(),128)
            self.assertEqual(sum((4-r['merge_depth'])*(64-r['keep']) for r in e.routes),64)
    def test_adaptive_extremes(self):
        for threshold,depth,keep in [(-1,3,32),(2,2,48)]:
            e=Engine(self.m,{'kind':'adaptive_equal'},{'early_median':threshold});f=Engine(self.m,{'kind':'fixed','depth':depth,'keep':keep})
            a=e.prefill(e.new_state(),segments(),128);b=f.prefill(f.new_state(),segments(),128)
            torch.testing.assert_close(e.forward(a,[4,6],'Q'),f.forward(b,[4,6],'Q'))
        e=Engine(self.m,{'kind':'adaptive_safe'},{'early_q25':-1,'late_q25':-1});f=Engine(self.m)
        a=e.prefill(e.new_state(),segments(),128);b=f.prefill(f.new_state(),segments(),128)
        torch.testing.assert_close(e.forward(a,[4,6],'Q'),f.forward(b,[4,6],'Q'))
    def test_future_chunk_isolation(self):
        for spec in ({'kind':'fixed','depth':2,'keep':48},{'kind':'adaptive_equal'},{'kind':'adaptive_safe'},{'kind':'random_equal'}):
            e=Engine(self.m,spec,{'early_median':.5,'early_q25':.3,'late_q25':.4},record=True)
            a=segments();b=segments();b['T'][64:]=[6]*64;states=[];routes=[]
            for seg in (a,b):
                s=e.new_state();e.routes=[];e.forward(s,seg['S'],'S');e.forward(s,seg['T'],'T');states.append(s);routes.append(e.routes)
            self.assertEqual(routes[0][0],routes[1][0])
            r=routes[0][0]
            for i,(x,y) in enumerate(zip(states[0].cache.layers,states[1].cache.layers)):
                n=4+(64 if i<r['merge_depth'] else r['keep'])
                torch.testing.assert_close(x.keys[:,:,:n],y.keys[:,:,:n],atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(x.values[:,:,:n],y.values[:,:,:n],atol=1e-6,rtol=1e-5)
    def test_block_partition(self):
        for kind in ('adaptive_equal','adaptive_safe','random_equal'):
            e=Engine(self.m,{'kind':kind},{'early_median':.5,'early_q25':.3,'late_q25':.4})
            a=e.prefill(e.new_state(),segments(),64);b=e.prefill(e.new_state(),segments(),128)
            torch.testing.assert_close(e.forward(a,[4,8],'Q'),e.forward(b,[4,8],'Q'),atol=1e-6,rtol=1e-5)
    def test_teacher(self):
        e=Engine(self.m,record=True,measure_depths=(1,2,3));e.prefill(e.new_state(),segments(),128)
        self.assertEqual(len(e.features),12);self.assertTrue(all(r['keep']==64 for r in e.routes))
if __name__=='__main__':unittest.main(verbosity=2)
