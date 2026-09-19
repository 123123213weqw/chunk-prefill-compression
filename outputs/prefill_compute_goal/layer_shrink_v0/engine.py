"""Causal Qwen layer-wise hidden-state downsampling, not post-hoc KV eviction.

No weights trained. Only completed T blocks are shortened, after split_depth
full decoder layers. S/R/Q/decode remain full resolution. Each cache layer has
its own physical length and original logical RoPE positions.
"""
from dataclasses import dataclass
import torch
from transformers import DynamicCache

@dataclass(frozen=True)
class Checkpoint:
    lengths: tuple
    logical_next: int

class State:
    def __init__(self, config, device):
        self.cache=DynamicCache(config=config)
        self.positions=[torch.empty(0,dtype=torch.long,device=device) for _ in range(config.num_hidden_layers)]
        self.logical_next=0

    def checkpoint(self):
        return Checkpoint(tuple(len(p) for p in self.positions),self.logical_next)

    def restore(self, checkpoint):
        if checkpoint.logical_next>self.logical_next:
            raise ValueError('cannot restore future cache')
        for layer,pos,n in zip(self.cache.layers,self.positions,checkpoint.lengths):
            if n>len(pos):raise ValueError('cannot resurrect discarded entries')
            remove=layer.get_seq_length()-n
            if remove:layer.crop(-remove)
        self.positions=[p[:n] for p,n in zip(self.positions,checkpoint.lengths)]
        self.logical_next=checkpoint.logical_next

    def validate(self):
        assert len(self.cache.layers)==len(self.positions)
        for layer,pos in zip(self.cache.layers,self.positions):
            assert layer.get_seq_length()==len(pos)
            if len(pos):
                assert int(pos[-1])<self.logical_next
                assert bool((pos[1:]>pos[:-1]).all())

    def tensor_bytes(self):
        return sum((l.keys.numel()*l.keys.element_size()+l.values.numel()*l.values.element_size())
                   for l in self.cache.layers if l.is_initialized)

    def position_bytes(self):
        return sum(p.numel()*p.element_size() for p in self.positions)


def shrink(hidden,positions,method):
    """A complete 64-token chunk becomes 32 ending-position representations."""
    if method=='identity':return hidden,positions
    if hidden.shape[1]%64:raise ValueError('T calls must contain whole 64-token chunks')
    if method=='pair_mean':
        h=hidden.reshape(hidden.shape[0],-1,2,hidden.shape[-1]).float().mean(2).to(hidden.dtype)
    elif method=='endpoint':h=hidden[:,1::2,:]
    else:raise ValueError('unknown method')
    return h,positions[1::2]


class Engine:
    def __init__(self,model,method='identity',split_depth=12,native=False):
        if method not in ('identity','pair_mean','endpoint'):raise ValueError(method)
        if not 0<split_depth<model.config.num_hidden_layers:raise ValueError('split depth must be inside the model')
        if native and method!='identity':raise ValueError('native baseline cannot shrink')
        if model.config.model_type!='qwen3':raise ValueError('only inspected Qwen3 architecture supported')
        if model.config._attn_implementation!='sdpa':raise ValueError('v0 mask policy is validated for SDPA only')
        if getattr(model.config,'use_sliding_window',False):raise ValueError('sliding window not supported')
        self.model=model;self.method=method;self.split_depth=split_depth;self.native=native
        self.device=next(model.parameters()).device

    def new_state(self):return State(self.model.config,self.device)

    @torch.inference_mode()
    def forward(self,state,ids,stage,logical_start=None,want_logits=True):
        if stage not in ('S','T','R','Q','decode'):raise ValueError(stage)
        if not ids:raise ValueError('empty input')
        start=state.logical_next if logical_start is None else logical_start
        if start!=state.logical_next:raise ValueError('logical position must advance by ORIGINAL input count')
        if stage=='T' and self.method!='identity' and len(ids)%64:
            raise ValueError('compressed T prefill requires completed 64-token chunks')
        x=torch.tensor([ids],device=self.device,dtype=torch.long)
        pos=torch.arange(start,start+len(ids),device=self.device)
        if self.native:
            if any(len(p)!=start for p in state.positions):raise ValueError('native model requires unshortened cache')
            result=self.model.model(input_ids=x,position_ids=pos[None],past_key_values=state.cache,use_cache=True)
            logits=self.model.lm_head(result.last_hidden_state[:,-1:])[:,0] if want_logits else None
            for i in range(len(state.positions)):state.positions[i]=torch.cat([state.positions[i],pos])
            state.logical_next+=len(ids)
            return logits
        hidden=self.model.model.embed_tokens(x)
        # Lower and upper layer families have shared mask shape, but can have
        # DIFFERENT cache lengths. Recompute per family, never infer from layer 0.
        mask=None;rope=None;family=-1
        for i,layer in enumerate(self.model.model.layers):
            if i==self.split_depth and stage=='T':hidden,pos=shrink(hidden,pos,self.method)
            current_family=0 if i<self.split_depth else 1
            if current_family!=family:
                # Match HF SDPA mask elision and boolean dtype. An unnecessary
                # explicit float mask changes GQA repetition/backend selection
                # and produced FP16 numerical drift in the real-model gate.
                if len(state.positions[i])==0 or len(pos)==1:
                    mask=None  # square causal prefill, or fully visible 1-token decode
                else:
                    allpos=torch.cat([state.positions[i],pos])
                    mask=(allpos[None,:]<=pos[:,None])[None,None]
                rope=self.model.model.rotary_emb(hidden,pos[None]);family=current_family
            hidden=layer(hidden,attention_mask=mask,position_ids=pos[None],position_embeddings=rope,
                         past_key_values=state.cache,use_cache=True)
            state.positions[i]=torch.cat([state.positions[i],pos])
        state.logical_next+=len(ids)
        # Match the native model's final norm, even when prefix logits are not needed.
        hidden=self.model.model.norm(hidden)
        return self.model.lm_head(hidden[:,-1:])[:,0] if want_logits else None

    def prefill(self,state,segments,block=1024):
        if block<64 or block%64:raise ValueError('block must be a multiple of 64')
        for stage in ('S','T','R'):
            ids=segments[stage]
            for off in range(0,len(ids),block):self.forward(state,ids[off:off+block],stage,want_logits=False)
        return state

    @torch.inference_mode()
    def answer(self,state,q,max_tokens,eos_ids):
        if max_tokens<1:raise ValueError('max_tokens must be positive')
        checkpoint=state.checkpoint();tokens=[]
        try:
            logits=self.forward(state,q,'Q')
            for _ in range(max_tokens):
                token=int(logits[0].argmax());tokens.append(token)
                if token in eos_ids or len(tokens)>=max_tokens:break
                logits=self.forward(state,[token],'decode')
            return tokens
        finally:state.restore(checkpoint)


class ShapeAudit:
    """Actual module input hooks, disabled for timed trials."""
    def __init__(self,model):
        self.rows=[];self.handles=[]
        for i,l in enumerate(model.model.layers):
            modules={**{name:getattr(l.self_attn,name) for name in ('q_proj','k_proj','v_proj','o_proj')},
                     **{name:getattr(l.mlp,name) for name in ('gate_proj','up_proj','down_proj')}}
            for name,module in modules.items():
                def hook(mod,inputs,layer=i,op=name):
                    x=inputs[0];self.rows.append({'layer':layer,'op':op,'tokens':x.numel()//x.shape[-1],
                                               'in_features':mod.in_features,'out_features':mod.out_features})
                self.handles.append(module.register_forward_pre_hook(hook))
    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]
    def linear_flops(self):return sum(2*r['tokens']*r['in_features']*r['out_features'] for r in self.rows)
