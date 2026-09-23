"""Closed-chunk adaptive downsampling. Not a token-level likelihood-preserving LM.
Routing sees current completed chunk only, never Q/gold or later chunks.
"""
import math,torch
from base_engine import Engine as BaseEngine,ShapeAudit

def bins(keep,phase='ceil'):
    if not 1<=keep<=64:raise ValueError('invalid keep')
    f=math.ceil if phase=='ceil' else math.floor
    boundaries=[f(i*64/keep) for i in range(keep+1)]
    return [(boundaries[i],boundaries[i+1]) for i in range(keep)]

_POOL={}

def pool(h,keep,phase='ceil'):
    if h.shape[-2]!=64:raise ValueError('requires 64-token chunk')
    if keep==64:return h
    key=(keep,phase,str(h.device))
    if key not in _POOL:
        w=torch.zeros(keep,64,device=h.device,dtype=torch.float32)
        for j,(a,b) in enumerate(bins(keep,phase)):w[j,a:b]=1/(b-a)
        _POOL[key]=w
    return torch.matmul(_POOL[key],h.float()).to(h.dtype)

def score(h,keep,phase='ceil'):
    z=pool(h,keep,phase).float()
    owner=torch.tensor([j for j,(a,b) in enumerate(bins(keep,phase)) for _ in range(a,b)],device=h.device)
    err=(h.float()-z.index_select(-2,owner)).square().sum(dim=(-2,-1))
    scale=h.float().square().sum(dim=(-2,-1)).clamp_min(1e-20)
    return (err/scale).sqrt()

def plans(layers):
    depths=sorted({round(layers*f) for f in (1/3,1/2,2/3,3/4,5/6)})
    p={f'd{d}_k{k}':{'kind':'fixed','depth':d,'keep':k,'phase':'ceil'} for d in depths for k in (32,48)}
    early=layers//2;late=3*layers//4
    assert layers%4==0 and (layers-early)*16==(layers-late)*32
    p[f'd{early}_k48_floor']={'kind':'fixed','depth':early,'keep':48,'phase':'floor'}
    # Additional exact-budget points exist for 36 layers; no rounding disguised as equality.
    budget=(layers-early)*16
    for keep in (16,40):
        if budget%(64-keep)==0:
            d=layers-budget//(64-keep)
            if 0<d<layers:p[f'd{d}_k{keep}']={'kind':'fixed','depth':d,'keep':keep,'phase':'ceil'}
    return p,early,late

class Engine(BaseEngine):
    def __init__(self,model,spec=None,thresholds=None,record=False,measure_depths=()):
        self.spec=spec or {'kind':'identity'};self.kind=self.spec['kind']
        _,self.early,self.late=plans(model.config.num_hidden_layers)
        super().__init__(model,method='identity',split_depth=self.early,native=self.kind=='native')
        self.thresholds=thresholds or {};self.record=record;self.routes=[];self.features=[];self.attn_rows=[]
        self.t_start=None
        self.measure_depths=set(measure_depths)
        self.boundaries=({self.spec['depth']} if self.kind=='fixed' else {self.early,self.late})
    @torch.inference_mode()
    def forward(self,state,ids,stage,logical_start=None,want_logits=True):
        if self.native:return super().forward(state,ids,stage,logical_start,want_logits)
        if stage not in ('S','T','R','Q','decode') or not ids:raise ValueError('stage/input')
        start=state.logical_next if logical_start is None else logical_start
        if start!=state.logical_next:raise ValueError('logical position')
        if stage=='T' and len(ids)%64:raise ValueError('whole chunks required')
        pos=torch.arange(start,start+len(ids),device=self.device)
        h=self.model.model.embed_tokens(torch.tensor([ids],device=self.device))
        if stage=='T' and self.t_start is None:self.t_start=start
        chunks=len(ids)//64 if stage=='T' else 0
        lengths=[64]*chunks;routes=[{'chunk_start':start+64*j,'merge_depth':self.model.config.num_hidden_layers,'keep':64} for j in range(chunks)]
        mask=rope=None
        for i,layer in enumerate(self.model.model.layers):
            changed=False
            if stage=='T':
                if self.measure_depths and i in self.measure_depths:
                    assert all(n==64 for n in lengths),'teacher collection must be uncompressed'
                    batch=h.reshape(chunks,64,-1)
                    for k in (32,48):
                        values=score(batch,k).cpu().tolist()
                        self.features.extend({'chunk_start':start+64*j,'depth':i,'keep':k,'score':v} for j,v in enumerate(values))
                target=None;values=None;selected=[]
                if self.kind=='fixed' and i==self.spec['depth']:
                    target=self.spec['keep'];selected=list(range(chunks))
                    if 'only_chunk' in self.spec:selected=[j for j in selected if (start-self.t_start)//64+j==self.spec['only_chunk']]
                    if 'restore_chunk' in self.spec:selected=[j for j in selected if (start-self.t_start)//64+j!=self.spec['restore_chunk']]
                elif self.kind in ('adaptive_equal','adaptive_safe','random_equal') and i in (self.early,self.late):
                    eligible=[j for j,n in enumerate(lengths) if n==64]
                    if eligible:
                        parts=list(h.split(lengths,dim=1));batch=torch.cat([parts[j] for j in eligible],dim=0)
                        target=48 if i==self.early else 32
                        if self.kind=='random_equal':
                            selected=([j for j in eligible if ((start+64*j)//64*1103515245+20260920)%2147483647<1073741824] if i==self.early else eligible)
                        elif self.kind=='adaptive_equal' and i==self.late:selected=eligible
                        else:
                            values=score(batch,target).cpu().tolist()
                            threshold=self.thresholds['early_median' if self.kind=='adaptive_equal' else 'early_q25' if i==self.early else 'late_q25']
                            selected=[j for j,v in zip(eligible,values) if v<=threshold]
                            if self.record:
                                for j,v in zip(eligible,values):routes[j][f'score_d{i}']=v
                if selected:
                    selected=set(selected);parts=h.split(lengths,dim=1);pp=pos.split(lengths);out=[];op=[]
                    for j,(part,p) in enumerate(zip(parts,pp)):
                        if j in selected:
                            if self.record and self.kind=='fixed':routes[j]['local_score']=float(score(part,target,self.spec.get('phase','ceil'))[0])
                            ends=torch.tensor([b-1 for a,b in bins(target,self.spec.get('phase','ceil'))],device=pos.device)
                            part=part[:,ends] if self.spec.get('merge')=='endpoint' else pool(part,target,self.spec.get('phase','ceil'))
                            p=p[ends];lengths[j]=target;routes[j].update(merge_depth=i,keep=target)
                        out.append(part);op.append(p)
                    h=torch.cat(out,dim=1);pos=torch.cat(op);changed=True
            if i==0 or i in self.boundaries or changed:
                if not len(state.positions[i]) or len(pos)==1:mask=None
                else:mask=(torch.cat([state.positions[i],pos])[None,:]<=pos[:,None])[None,None]
                rope=self.model.model.rotary_emb(h,pos[None])
            if self.record:self.attn_rows.append({'stage':stage,'layer':i,'queries':len(pos),'keys':len(state.positions[i])+len(pos)})
            h=layer(h,attention_mask=mask,position_ids=pos[None],position_embeddings=rope,past_key_values=state.cache,use_cache=True)
            state.positions[i]=torch.cat([state.positions[i],pos])
        state.logical_next+=len(ids)
        if self.record:self.routes.extend(routes)
        h=self.model.model.norm(h)
        return self.model.lm_head(h[:,-1:])[:,0] if want_logits else None
