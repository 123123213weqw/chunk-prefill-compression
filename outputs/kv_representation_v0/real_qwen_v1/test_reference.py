import json, math
from pathlib import Path
import torch
from run import cluster,groups

torch.set_num_threads(2);torch.manual_seed(20260907)
k=torch.randn(16,8).repeat_interleave(4,0);v=torch.randn(64,8);q=torch.randn(10,8)
ok=torch.randn(32,8);ov=torch.randn(32,8)
pk,pv,b=groups(k,v,cluster(k,16),16)
raw=(q@torch.cat([k,ok]).T/math.sqrt(8)).softmax(-1)@torch.cat([v,ov])
pred=(q@torch.cat([pk,ok]).T/math.sqrt(8)+torch.cat([b,torch.zeros(32)])).softmax(-1)@torch.cat([pv,ov])
exact=float((raw-pred).abs().max());assert exact<1e-6
bad=(q@torch.cat([pk,ok]).T/math.sqrt(8)).softmax(-1)@torch.cat([pv,ov])
assert float((raw-bad).abs().max())>.01
# The partition-function recombination used by the evaluator equals direct mixed attention.
a=q@k.T/math.sqrt(8);z=a.logsumexp(-1);y=a.softmax(-1)@v
r=q@ok.T/math.sqrt(8);rz=r.logsumexp(-1);ry=r.softmax(-1)@ov
mass=torch.sigmoid(z-rz)
merged=mass[:,None]*y+(1-mass[:,None])*ry
err=float((raw-merged).abs().max());assert err<1e-6
# Repeated identical keys must still produce exactly m nonempty groups.
labels=cluster(torch.ones(64,8),32);assert (torch.bincount(labels,minlength=32)>0).all()
# Future entries must not affect reference masked attention.
K=torch.randn(100,8);V=torch.randn(100,8);mask=torch.arange(100)>70
l=(q@K.T/math.sqrt(8)).masked_fill(mask,float('-inf'));a=l.softmax(-1)@V
K[71:]+=999;V[71:]-=999
l=(q@K.T/math.sqrt(8)).masked_fill(mask,float('-inf'));c=l.softmax(-1)@V
assert torch.equal(a,c)
print(json.dumps({'status':'PASS','checks':5,'identical_key_max_abs_error':exact,'partition_recombination_max_abs_error':err}))
