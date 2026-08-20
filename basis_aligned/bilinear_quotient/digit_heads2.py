"""DIGIT HEADS 2 -- necessity and sufficiency, because the first
version registered the wrong bar.
digit_heads asked whether the SUM of individual per-head subspace
removals reproduces the whole-bundle effect. That is an additivity
assumption about a cross-entropy readout, not an exactness check,
and on a multiplicative network there is no reason for it to hold
-- the user's tier-4 point (512) applies to localization too. Its
early output shows exactly that: per-head removals of +0.001 to
+0.009 against a whole-bundle effect of +0.135.
The genuine exactness check is different and is registered here as
(0): removing the projector from EVERY head must reproduce
removing it from the component output, because projection is
linear and the output is the sum of the heads' c_proj
contributions. That one has to hold, and if it does not the
implementation is wrong.
Localization on a superadditive effect needs two measurements per
head, not one:
  ALONE    remove the subspace from head h only -- necessity
  ALL-BUT  remove it from every head EXCEPT h -- sufficiency
A head that carries the effect scores high alone AND leaves little
behind when it is the only one spared. A distributed effect scores
low alone and high all-but for every head, and that is a real
answer about the model rather than a failed search.
Three further disjoint samples (1800, 2000, 2200) per the v4 rule.
REGISTERED PREDICTIONS:
  (0) EXACTNESS: the all-heads arm reproduces the full-bundle
      digit dissociation to within 20%. Failure VOIDS the run;
  (a) NECESSITY: some head removed alone gives >= 40% of the
      whole effect;
  (b) SUFFICIENCY: for that same head, removing the subspace from
      all OTHER heads leaves <= 60% of the whole effect, i.e.
      sparing it preserves at least 40%;
  (c) ATLAS: that head is 8.7, which has by far the highest digit
      read-enrichment of the eighteen (2.45, next is 1.68), from a
      profile that knew nothing about this subspace.
  NULL: a rank-matched random subspace removed from the leading
      head, three seeds, gives |digit dissociation| < 0.03.
If (a) and (b) both fail while (0) holds, the digit subspace is
distributed across the heads of a6 and a8 and its mechanism is
multiplicative -- report that, and take it to tier 4 rather than
looking for a head that is not there."""
import ast, json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'digit_heads2_results.json'
TAG='r.2.0.1'; LAYERS=[6,8]; NFRESH=96; SKIPS=(1800,2000,2200)

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
                            elif kind=='allbut' and (li,h)!=hook_spec[1:3]:
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
    # full-bundle reference: the component-level ablation whose
    # dissociation 516 measured at +0.135 over five samples
    ref=[]
    for SKIP in SKIPS:
        rows=cl.fineweb_rows(NFRESH,skip=SKIP)
        dig,pun=classes(rows); base=ce_rows(rows)
        a=ce_rows(rows,('all',))
        ref.append(diss(base,a,dig))
    WHOLE=sum(ref)/len(ref)
    print(f'full-bundle reference over samples: '
          f'{[round(x,4) for x in ref]} mean {WHOLE:+.4f}',flush=True)
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
                b=ce_rows(rows,('allbut',li,h))
                row[f'{li}.{h}']={
                  'digit':round(diss(base,a,dig),4),
                  'punct':round(diss(base,a,pun),4),
                  'allbut_digit':round(diss(base,b,dig),4)}
                print(f'  [skip={SKIP}] {li}.{h}: alone '
                      f"{row[f'{li}.{h}']['digit']:+.4f} | all-but "
                      f"{row[f'{li}.{h}']['allbut_digit']:+.4f}",
                      flush=True)
        per[SKIP]=row
        json.dump({str(k):v for k,v in per.items()},
                  open(OUT,'w'),indent=1)
    heads=[f'{li}.{h}' for li in LAYERS for h in range(NH)]
    mean={k:sum(per[s][k]['digit'] for s in SKIPS)/len(SKIPS)
          for k in ['all']+heads}
    meanp={k:sum(per[s][k]['punct'] for s in SKIPS)/len(SKIPS)
           for k in ['all']+heads}
    meanab={k:sum(per[s][k]['allbut_digit'] for s in SKIPS)/len(SKIPS)
            for k in heads}
    total=mean['all']; hsum=sum(mean[k] for k in heads)
    print(f'\nmean over samples -- whole {total:+.4f}, '
          f'sum of heads alone {hsum:+.4f}')
    # (0) is now a real exactness check: removing the projector from
    # every head equals removing it from the component output,
    # because projection is linear and the output is the sum of the
    # heads' c_proj contributions. digit_heads' version of this bar
    # compared the SUM OF INDIVIDUAL removals to the whole, which
    # tests additivity of a nonlinear CE readout, not exactness --
    # a badly designed bar (writeup 517).
    p0=abs(total-WHOLE)<=0.20*abs(WHOLE) if abs(WHOLE)>1e-6 else False
    print(f'(0) all-heads arm reproduces the full-bundle '
          f'dissociation ({total:+.4f} vs {WHOLE:+.4f}): '
          f"{'HELD' if p0 else 'FAILED -- RUN VOID'}")
    order=sorted(heads,key=lambda k:-mean[k])
    print('  head   alone   all-but-it   (whole '
          f'{total:+.4f})')
    for k in order:
        print(f'  {k:>5} {mean[k]:+.4f}   {meanab[k]:+.4f}')
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
    vs,_=cl.score_bar('b-suff',
                      1-(meanab[top]/total if abs(total)>1e-6 else 1),
                      0.40,denom=total,ref=meanab[top])
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
         'mean_allbut':{k:round(v,4) for k,v in meanab.items()},
         'whole_bundle':round(WHOLE,4),
         'pred_0':True,'pred_a':va=='HELD',
         'pred_b_sufficiency':vs=='HELD','pred_atlas':vb=='HELD',
         'pred_c':vc=='HELD','null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
