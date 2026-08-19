"""The attention-pattern census -- the one component family never profiled.
bilin18's attention scores are products s1*s2 with no softmax. Question: how much
of each head's pattern is POSITIONAL (a function of query-key offset alone) vs
CONTENT-based, and how does that change with depth?

Per head, per layer in (0,2,5,9,13,16): collect patterns on 12 sequences;
positional predictor = the head's mean pattern as a function of offset (fit on 6
sequences, evaluated on the other 6); R^2_pos = held-out variance explained by
offset alone. REGISTERED PREDICTIONS: (a) depth gradient -- median R^2_pos of
early layers (0,2) exceeds late layers (13,16) by >= 0.2 (early = positional
scaffolding, late = content routing); (b) null: patterns with shuffled key
positions give R^2_pos <= 0.05; (c) within every layer at least one head is
strongly positional (R^2 >= 0.5) -- position tracking never disappears."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
LAYERS=(0,2,5,9,13,16)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_pattern_census_results.json')

@torch.no_grad()
def get_patterns(idx):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    pats={}
    for li in range(17):
        blk=m.transformer.h[li]; a=blk.attn
        x=blk.lambdas[0]*x+blk.lambdas[1]*x0
        hcur=F.rms_norm(x,(D,))
        def qk(l):
            z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
            return apply_rot(z,cosb,sinb)
        v=a.c_v(hcur).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
        s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0)
        if li in LAYERS: pats[li]=pat.detach().float()
        x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
    return pats

@torch.no_grad()
def main():
    t0=time.time()
    T=257
    ptr=get_patterns(FW[300:306,:T].to(DEV))
    pte=get_patterns(FW(slice(306,312)) if False else FW[306:312,:T].to(DEV))
    g=torch.Generator(device=DEV).manual_seed(0)
    res={}
    for li in LAYERS:
        r2s=[];r2n=[]
        for h in range(NH):
            Ptr=ptr[li][:,h]; Pte=pte[li][:,h]
            Tq=Ptr.shape[-1]
            offmean=torch.zeros(Tq,Tq,device=DEV)
            cnt=torch.zeros(Tq,Tq,device=DEV)
            di=torch.arange(Tq,device=DEV)
            off=(di[:,None]-di[None,:])
            for o in range(Tq):
                sel=(off==o)
                if sel.any():
                    offmean[sel]=Ptr[:, sel].mean()
            lowmask=torch.tril(torch.ones(Tq,Tq,device=DEV,dtype=torch.bool))
            y=Pte[:,lowmask]; yp=offmean[lowmask][None,:].expand_as(y)
            r2=1-float(((y-yp)**2).mean()/y.var().clamp_min(1e-12))
            r2s.append(r2)
            perm=torch.randperm(Tq,generator=g,device=DEV)
            Pn=Pte[:,:,perm]
            yn=Pn[:,lowmask]
            r2sh=1-float(((yn-yp)**2).mean()/yn.var().clamp_min(1e-12))
            r2n.append(r2sh)
        res[li]={'r2':r2s,'null':r2n}
        med=sorted(r2s)[NH//2]
        print(f'L{li:2d}: per-head positional R^2 '
              +' '.join(f'{r:+.2f}' for r in r2s)+f' | median {med:+.2f} | '
              f'null med {sorted(r2n)[NH//2]:+.2f}',flush=True)
    early=[r for li in (0,2) for r in res[li]['r2']]
    late=[r for li in (13,16) for r in res[li]['r2']]
    me=sorted(early)[len(early)//2]; ml=sorted(late)[len(late)//2]
    nulls=[r for li in LAYERS for r in res[li]['null']]
    pa=(me-ml)>=0.2
    pb=max(nulls)<=0.05
    pc=all(max(res[li]['r2'])>=0.5 for li in LAYERS)
    out={'per_layer':{str(k):v for k,v in res.items()},
         'median_early':me,'median_late':ml,
         'pred_a':bool(pa),'null_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) early more positional by >=0.2: {'HELD' if pa else 'FAILED'} "
          f"({me:+.2f} vs {ml:+.2f})")
    print(f"(b) shuffled null <=0.05: {'HELD' if pb else 'VIOLATED'}")
    print(f"(c) every layer keeps a positional head: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
