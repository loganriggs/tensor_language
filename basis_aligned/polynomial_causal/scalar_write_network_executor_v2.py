"""Terminal-state capture extension of the frozen v1 executor; extracted from the
frozen gerund network runner, with row count and e supplied explicitly.
Native g forward, learned mixing, attention first-value cache and nonlinear
readout semantics are unchanged. This is an intervention harness, not a model.
"""
import torch
import torch.nn.functional as F
import circuit_unit_greedy as g
from gerund_scalar_network_v1 import coefficients

GROUPS={'attention':(0,),'mlp':(1,),'all':(0,1)}
def bridge(a,b):
    a=a.double();b=b.double();err=a-b
    return {'max_abs':float(err.abs().max()),'relative_l2':float(err.norm()/b.norm().clamp_min(1e-30))}

class ScalarWriteNetwork:
    def __init__(self,backend,e,forward_limit,sequence_limit):
        self.backend=backend;self.model=backend.model;self.e=e
        self.beta,self.gamma=coefficients(torch.stack([b.lambdas.detach() for b in self.model.transformer.h]))
        self.counts=[0,0];self.limits=[forward_limit,sequence_limit];self.visits=0
        self.bridges={};self.scalar_checks=[];self.finite=True
        def count(_m,args):
            self.counts[0]+=1;self.counts[1]+=len(args[0])
            assert all(a<=b for a,b in zip(self.counts,self.limits))
        self.counter=self.model.transformer.h[0].attn.register_forward_pre_hook(count)
    def close(self):self.counter.remove()
    def read(self,h):
        return 30*torch.tanh(F.linear(F.rms_norm(h.float(),(1152,)),self.model.lm_head.weight.float())/30)
    def body(self,panel,side,group=None,mode=None,donor=None,label=''):
        batch=g.batch_of(panel,side);device=self.e.device;n=len(panel)
        ix=torch.arange(n,device=device);pos=torch.tensor(batch.semantic_positions,device=device)
        cap={};raw={};hooks=[];e=self.e
        def callback(l,k):
            def hook(_m,args,out):
                self.visits+=1;y=out[0] if k==0 else out;value=y[ix,pos]
                if group is not None and k in GROUPS[group]:
                    target=donor[l,k] if mode=='donor' else torch.zeros(n,device=device,dtype=torch.float64)
                    updated=value.double()+(target-value.double()@e)[:,None]*e
                    changed=y.clone();changed[ix,pos]=updated.to(y);y=changed;value=y[ix,pos]
                cap[l,k]=value.detach().clone()
                return (y,out[1]) if k==0 else y
            return hook
        for l,block in enumerate(self.model.transformer.h):
            hooks.extend([block.attn.register_forward_hook(callback(l,0)),block.mlp.register_forward_hook(callback(l,1))])
        try:af,z=g.forward_units(self.backend,batch,return_logits=True,capture_resid=raw)
        finally:
            for handle in hooks:handle.remove()
        writes=torch.stack([torch.stack([cap[l,0],cap[l,1]]) for l in range(18)])
        raw17=torch.stack([raw[(rid,17)] for rid in batch.row_ids])
        h=raw17+cap[17,1]
        ids=torch.tensor([t[p] for t,p in zip(batch.token_rows,batch.semantic_positions)],device=device)
        x0=F.rms_norm(self.model.transformer.wte(ids),(1152,))
        reconstructed=self.gamma*x0.double()+(self.beta[:,None,None]*writes.double().sum(1)).sum(0)
        self.bridges[label+'_state']=bridge(reconstructed,h);self.bridges[label+'_readout']=bridge(self.read(reconstructed),z)
        if group=='all':
            expected=self.gamma*(x0.double()@e)
            if mode=='donor':expected=expected+(self.beta[:,None]*donor.sum(1)).sum(0)
            error=(h.double()@e-expected).abs();tol=torch.maximum(1e-3+1e-5*expected.abs(),1e-6*h.double().norm(dim=1))
            self.scalar_checks.append({'arm':label,'max_abs':float(error.max()),'max_scaled':float((error/tol).max())})
        self.finite=self.finite and all(bool(torch.isfinite(v).all()) for v in [writes,h,z,af])
        return {'af':af,'z':z,'h':h,'x0':x0,'scalars':writes.double()@e,'raw17':raw17,'writes':writes}
    def valid(self):
        return self.counts==self.limits and self.visits==36*self.counts[0] and self.finite and abs(float(self.e.norm())-1)<=1e-5 and all(
            x['relative_l2']<=1e-5 if k.endswith('_state') else x['max_abs']<=1e-3 and x['relative_l2']<=1e-5 for k,x in self.bridges.items()) and all(x['max_scaled']<=1 for x in self.scalar_checks)
