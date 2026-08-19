"""UNIT USAGE -- is the MLP top-512 active set per-position DYNAMIC
or is there a STATIC ~512-unit subset that nearly suffices? Sharp
fork: dynamic sparsity (the model routes among many units) vs a
fixed circuit subset (most units dead weight). Count how often each
unit appears in the per-position top-512 (mid MLPs, census grid),
then substitute the STATIC top-512-most-used set.
REGISTERED PREDICTIONS:
  (a) usage is heavy-tailed: the 512 most-used units capture >=50%
      of all top-512 slots;
  (b) static-512 substitution costs <=2x dynamic top-512 (+0.055)
      -> a fixed subset nearly suffices (echoes the zero-parameter
      half-units result); if >2x, routing is real;
  (c) usage histogram written."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'unit_usage_results.json'
MIDS=[4,5,6,7,8,9]

@torch.no_grad()
def main():
    t0=time.time()
    def hooks(k,random_units=False):
        hs=[]
        for li in MIDS:
            mlp=m.transformer.h[li].mlp
            def fh(mo,i_,o_,mlp=mlp,k=k,random_units=random_units):
                x=i_[0]
                h=mlp.Left(x)*mlp.Right(x)
                if random_units:
                    g=torch.Generator().manual_seed(4)
                    idx=torch.randperm(h.shape[-1],generator=g)[:k] \
                        .to(DEV)
                    msk=torch.zeros(h.shape[-1],device=DEV)
                    msk[idx]=1.0
                    h=h*msk
                else:
                    _,idx=h.abs().topk(k,dim=-1)
                    msk=torch.zeros_like(h).scatter(-1,idx,1.0)
                    h=h*msk
                return mlp.Down(h)+mlp.Down_bias
            hs.append(mlp.register_forward_hook(fh))
        return hs
    # usage counting on 64 rows
    ROWS=cl.rows()
    counts={li:torch.zeros(4608) for li in MIDS}
    cap=[]
    hs=[]
    for li in MIDS:
        mlp=m.transformer.h[li].mlp
        def fh(mo,i_,o_,mlp=mlp,li=li):
            x=i_[0]
            h=mlp.Left(x)*mlp.Right(x)
            _,idx=h.abs().topk(512,dim=-1)
            u,c=idx.reshape(-1).unique(return_counts=True)
            counts[li][u.cpu()]+=c.cpu().float()
            return o_
        hs.append(mlp.register_forward_hook(fh))
    for i in range(0,64,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    shares=[]
    STATIC={}
    for li in MIDS:
        c=counts[li]
        top=c.topk(512).indices
        shares.append(float(c[top].sum()/c.sum()))
        STATIC[li]=top.to(DEV)
    sh=sum(shares)/len(shares)
    print(f'static-512 slot share: {sh:.2%}',flush=True)
    def static_hooks():
        hs=[]
        for li in MIDS:
            mlp=m.transformer.h[li].mlp
            msk=torch.zeros(4608,device=DEV)
            msk[STATIC[li]]=1.0
            def fh(mo,i_,o_,mlp=mlp,msk=msk):
                x=i_[0]
                h=mlp.Left(x)*mlp.Right(x)*msk
                return mlp.Down(h)+mlp.Down_bias
            hs.append(mlp.register_forward_hook(fh))
        return hs
    base=cl.ce_sweep([])
    ds=cl.ce_sweep(static_hooks())
    st=round(float((ds-base).mean()),4)
    dd=cl.ce_sweep(hooks(512))
    dy=round(float((dd-base).mean()),4)
    print(f'static-512: {st:+.4f} | dynamic-512: {dy:+.4f}',flush=True)
    pa=sh>=0.5
    pb=st<=2*max(dy,1e-3)
    out={'slot_share_static512':round(sh,3),'static_cost':st,
         'dynamic_cost':dy,
         'usage_q':{li:[round(float(counts[li].topk(k).values[-1]),0)
                        for k in (10,512,2048)] for li in MIDS},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True}
    print(f"(a) heavy tail >=50%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) static <=2x dynamic: {'HELD' if pb else 'FAILED'}"
          f" -> {'fixed subset' if pb else 'routing is real'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
