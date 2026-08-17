"""Lexical complement of the pattern census. Section 125: attention is mostly
content-based. Is "content" here largely the KEY TOKEN'S IDENTITY (lexical
attention), matching the model's lexical bus and token-dominated stream? Same
protocol, predictor = per-head mean score per key-token id (fit on 6 sequences,
held out on 6; only token ids seen in training rows are scored, rest fall back
to the global mean). REGISTERED: (a) median lexical R^2 exceeds median positional
R^2 at >= 4 of 6 layers; (b) shuffled-key null <= 0.05; (c) combined
offset+lexical predictor reaches median R^2 >= 0.5 at >= 3 layers (the two
simple features cover most of the pattern).

Prior context -- the attention-pattern census.
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
     'bilin18_pattern_lexical_results.json')

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
        tr_idx=FW[300:306,:257].to(DEV)[:,:-1]
        te_idx=FW[306:312,:257].to(DEV)[:,:-1]
        r2s=[];r2n=[];r2c=[]
        for h in range(NH):
            Ptr=ptr[li][:,h]; Pte=pte[li][:,h]
            Tq=Ptr.shape[-1]
            di=torch.arange(Tq,device=DEV)
            off=(di[:,None]-di[None,:])
            offmean=torch.zeros(Tq,Tq,device=DEV)
            for o in range(Tq):
                sel=(off==o)
                if sel.any(): offmean[sel]=Ptr[:,sel].mean()
            V=int(max(int(tr_idx.max()),int(te_idx.max())))+1
            ssum=torch.zeros(V,device=DEV); scnt=torch.zeros(V,device=DEV)
            lowmask=torch.tril(torch.ones(Tq,Tq,device=DEV,dtype=torch.bool))
            for b in range(Ptr.shape[0]):
                keys=tr_idx[b]
                for qpos in range(Tq):
                    row=Ptr[b,qpos,:qpos+1]
                    ssum.index_add_(0,keys[:qpos+1],row)
                    scnt.index_add_(0,keys[:qpos+1],torch.ones(qpos+1,device=DEV))
            gmean=float(ssum.sum()/scnt.sum().clamp_min(1))
            lex=torch.where(scnt>0, ssum/scnt.clamp_min(1),
                            torch.full_like(ssum,gmean))
            y=Pte[:,lowmask]
            yp_pos=offmean[lowmask][None,:].expand_as(y)
            yl=[]
            for b in range(Pte.shape[0]):
                lk=lex[te_idx[b]]
                yl.append(lk[None,:].expand(Tq,Tq)[lowmask])
            yp_lex=torch.stack(yl)
            def r2of(pred):
                return 1-float(((y-pred)**2).mean()/y.var().clamp_min(1e-12))
            r2s.append(r2of(yp_lex))
            # combined: offset + lexical (residual stacking)
            comb=yp_pos+ (yp_lex-yp_lex.mean())
            r2c.append(r2of(comb))
            perm=torch.randperm(Tq,generator=g,device=DEV)
            yn=Pte[:,:,perm][:,lowmask]
            r2n.append(1-float(((yn-yp_lex)**2).mean()/yn.var().clamp_min(1e-12)))
        res[li]={'lex':r2s,'null':r2n,'combined':r2c}
        print(f'L{li:2d}: lexical R^2 '
              +' '.join(f'{r:+.2f}' for r in r2s)
              +f' | med {sorted(r2s)[NH//2]:+.2f} | comb med '
              f'{sorted(r2c)[NH//2]:+.2f} | null {sorted(r2n)[NH//2]:+.2f}',
              flush=True)
    POS={0:0.03,2:0.26,5:0.03,9:0.09,13:0.01,16:0.02}
    wins=sum(1 for li in LAYERS
             if sorted(res[li]['lex'])[NH//2]>POS[li])
    nulls=[r for li in LAYERS for r in res[li]['null']]
    nc=sum(1 for li in LAYERS if sorted(res[li]['combined'])[NH//2]>=0.5)
    pa=wins>=4
    pb=max(nulls)<=0.05
    pc=nc>=3
    out={'per_layer':{str(k):v for k,v in res.items()},
         'lexical_beats_positional_layers':wins,
         'pred_a':bool(pa),'null_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) lexical beats positional at >=4/6 layers: "
          f"{'HELD' if pa else 'FAILED'} ({wins}/6)")
    print(f"(b) shuffled null <=0.05: {'HELD' if pb else 'VIOLATED'}")
    print(f"(c) combined med >=0.5 at >=3 layers: {'HELD' if pc else 'FAILED'} ({nc})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
