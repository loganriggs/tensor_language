"""DIGIT HEADS -- which head carries the digit subspace?
The effect is now solid. Removing a 16-dimensional subspace from
the outputs of attention layers 6 and 8 makes DIGIT predictions
worse by +0.143, +0.164, +0.132 on three disjoint samples (and
+0.090, +0.146 on two earlier ones), where the population effect
for the same kind of ablation SPARES digits at -0.018. It is
specific to these directions: alternative spans of the same rank
from the same components give -0.027 to +0.033. It is additive
across its two halves. Punctuation on the same rows lands inside
its expected bracket in three of three samples, so the instrument
is behaving.
What it is not yet is a mechanism. Layers 6 and 8 have nine heads
each, and the subspace lives in the residual space those heads
write into, so each head's contribution to it can be removed
separately and exactly: the component output is the sum over heads
of c_proj applied to that head's slice, and the projector can be
applied to one summand at a time.
The head atlas supplies an advance bet again, from a profile built
with no knowledge of this subspace or of digits as a target class.
Of the eighteen heads in these two layers, 8.7 has by far the
highest digit read-enrichment at 2.45; 8.3 follows at 1.68 and
nothing else exceeds 1.29. If one head carries the effect, the
atlas says which, and the bet can fail seventeen ways.
Three disjoint samples (skip 1200, 1400, 1600) per the v4 rule
that no behavioural number is quoted from one sample.
REGISTERED PREDICTIONS:
  (0) DECOMPOSITION SANITY, checked before scoring: removing the
      subspace from ALL eighteen heads at once reproduces the
      whole-bundle digit dissociation to within 20%. If the
      per-head parts do not sum to the whole, the attribution is
      meaningless and the run is VOID;
  (a) CONCENTRATION: some single head carries >= 40% of the
      total digit dissociation, in the mean over samples;
  (b) THE ATLAS CALLED IT: that head is 8.7;
  (c) CLASS-SPECIFIC: the leading head's digit dissociation
      exceeds its own punctuation dissociation by >= 0.05, so it
      is a digit effect and not that head being generally
      important at rare tokens.
  NULL: a rank-matched RANDOM subspace removed from the leading
      head, three seeds, gives |digit dissociation| < 0.03. If
      random directions in the same head do the same thing, the
      subspace is not what matters.
Pairs reported alongside every ratio; bars scored through
cl.score_bar."""
import ast, json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'digit_heads_results.json'
TAG='r.2.0.1'; LAYERS=[6,8]; NFRESH=96; SKIPS=(1200,1400,1600)

def projector(key,probes):
    """Same construction as cl.proj_hooks, exposed as a matrix."""
    seen=set(); keep=[]
    for pr in probes:
        _,k,stag,blk=pr
        if k!=key: continue
        if (k,stag,tuple(blk)) in seen: continue
        seen.add((k,stag,tuple(blk))); keep.append(pr)
    drop=set()
    for i,(_,k1,s1,b1) in enumerate(keep):
        for j,(_,k2,s2,b2) in enumerate(keep):
            if i!=j and k1==k2 and s1==s2 and b2[0]<=b1[0] \
               and b1[1]<=b2[1] and (b1[1]-b1[0])<(b2[1]-b2[0]):
                drop.add(i)
    vs=[cl.pca_block(p[1],p[2],p[3])
        for i,p in enumerate(keep) if i not in drop]
    return orth(torch.cat(vs).T)

@torch.no_grad()
def ce_rows(rows,hook_spec=None):
    """hook_spec: None | ('head',layer,head) | ('all',) |
    ('rand',layer,head,seed)"""
    out=torch.zeros(rows.shape[0],T)
    for i in range(0,rows.shape[0],4):
        bb=rows[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=bb.shape[0]; hs=[]
        if hook_spec is not None:
            for li in LAYERS:
                at=m.transformer.h[li].attn
                P=PROJ[f'a{li}']
                def mk(li=li,at=at,P=P):
                    def fh(mo,args,o_):
                        y,v1r=o_; X=args[0]
                        v1b=args[1] if args[1] is not None else v1r
                        z,_=cl.head_parts(li,X,v1b)
                        W=at.c_proj.weight.float()
                        parts=[]
                        for h in range(NH):
                            zh=z[:,h].float()
                            parts.append(zh@W[:,h*128:(h+1)*128].T)
                        kind=hook_spec[0]
                        for h in range(NH):
                            if kind=='all': Q=P
                            elif kind=='head' and (li,h)==hook_spec[1:3]:
                                Q=P
                            elif kind=='rand' and (li,h)==hook_spec[1:3]:
                                Q=RAND[(li,h,hook_spec[3])]
                            else: continue
                            ph=parts[h]
                            parts[h]=ph-(ph@Q)@Q.T
                        tot=sum(parts)
                        if at.c_proj.bias is not None:
                            tot=tot+at.c_proj.bias.float()
                        return (tot.to(y.dtype),v1r)
                    return fh
                hs.append(at.register_forward_hook(mk()))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        out[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none').view(B,T).cpu()
        for h in hs: h.remove()
    return out

def classes(rows):
    nxt=rows[:,1:257]
    dig=torch.zeros_like(nxt,dtype=torch.bool)
    pun=torch.zeros_like(nxt,dtype=torch.bool)
    for r in range(nxt.shape[0]):
        for q in range(nxt.shape[1]):
            t=cl.d1(int(nxt[r,q])).strip()
            if t and t[0].isdigit(): dig[r,q]=True
            elif t and all(not c.isalnum() for c in t): pun[r,q]=True
    return dig,pun

def diss(base,abl,mask):
    d=abl-base
    return float(d[mask].mean())-float(d[~mask].mean())

PROJ={}; RAND={}

def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    lf=cl.leaf(TAG)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    for li in LAYERS:
        PROJ[f'a{li}']=projector(f'a{li}',probes)
        print(f'a{li} projector rank '
              f'{PROJ[f"a{li}"].shape[1]}',flush=True)
    per={}
    for SKIP in SKIPS:
        rows=cl.fineweb_rows(NFRESH,skip=SKIP)
        dig,pun=classes(rows)
        base=ce_rows(rows)
        whole=ce_rows(rows,('all',))
        wd=diss(base,whole,dig)
        row={'all':{'digit':round(wd,4),
                    'punct':round(diss(base,whole,pun),4)}}
        print(f'[skip={SKIP}] all heads: digit {wd:+.4f}',flush=True)
        for li in LAYERS:
            for h in range(NH):
                a=ce_rows(rows,('head',li,h))
                row[f'{li}.{h}']={
                  'digit':round(diss(base,a,dig),4),
                  'punct':round(diss(base,a,pun),4)}
                print(f'  [skip={SKIP}] {li}.{h}: digit '
                      f"{row[f'{li}.{h}']['digit']:+.4f} punct "
                      f"{row[f'{li}.{h}']['punct']:+.4f}",flush=True)
        per[SKIP]=row
        json.dump({str(k):v for k,v in per.items()},
                  open(OUT,'w'),indent=1)
    heads=[f'{li}.{h}' for li in LAYERS for h in range(NH)]
    mean={k:sum(per[s][k]['digit'] for s in SKIPS)/len(SKIPS)
          for k in ['all']+heads}
    meanp={k:sum(per[s][k]['punct'] for s in SKIPS)/len(SKIPS)
           for k in ['all']+heads}
    total=mean['all']; hsum=sum(mean[k] for k in heads)
    print(f'\nmean over samples -- whole {total:+.4f}, '
          f'sum of heads {hsum:+.4f}')
    p0=abs(hsum-total)<=0.20*abs(total) if abs(total)>1e-6 else False
    print(f'(0) per-head parts sum to the whole within 20%: '
          f"{'HELD' if p0 else 'FAILED -- RUN VOID'}")
    order=sorted(heads,key=lambda k:-mean[k])
    for k in order[:6]:
        print(f'  {k}: digit {mean[k]:+.4f} punct {meanp[k]:+.4f}')
    top=order[0]
    if not p0:
        json.dump({'pred_0':False,'mean':mean,'whole':total,
                   'head_sum':hsum},open(OUT,'w'),indent=1)
        return
    # NULL: random rank-matched subspace in the leading head
    li,h=int(top.split('.')[0]),int(top.split('.')[1])
    rnd=[]
    rows=cl.fineweb_rows(NFRESH,skip=SKIPS[0])
    dig,pun=classes(rows); base=ce_rows(rows)
    for s in (3,11,19):
        g=torch.Generator(device=DEV).manual_seed(s)
        V=torch.randn(PROJ[f'a{li}'].shape[1],D,generator=g,device=DEV)
        RAND[(li,h,s)]=orth(V.T)
        a=ce_rows(rows,('rand',li,h,s))
        rnd.append(round(diss(base,a,dig),4))
        print(f'  random seed {s} in {top}: digit {rnd[-1]:+.4f}',
              flush=True)
    va,_=cl.score_bar('a',mean[top]/total if abs(total)>1e-6 else 0,
                      0.40,denom=total,ref=mean[top])
    vb='HELD' if top=='8.7' else 'FAILED'
    vc,_=cl.score_bar('c',mean[top]-meanp[top],0.05)
    nul=max(abs(x) for x in rnd)<0.03
    print(f'(b) the atlas bet: leading head is {top}, predicted '
          f'8.7: {vb}')
    print(f"NULL (random subspace in {top} gives |digit| < 0.03, "
          f"max {max(abs(x) for x in rnd):.4f}): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'per_sample':{str(k):v for k,v in per.items()},
         'mean_digit':{k:round(v,4) for k,v in mean.items()},
         'mean_punct':{k:round(v,4) for k,v in meanp.items()},
         'whole':round(total,4),'head_sum':round(hsum,4),
         'ranking':order,'top':top,'random_in_top':rnd,
         'atlas_digit_enrichment':{'8.7':2.45,'8.3':1.68,'6.5':1.29},
         'pred_0':True,'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':vc=='HELD','null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
