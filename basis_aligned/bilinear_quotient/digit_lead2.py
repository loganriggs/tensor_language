"""DIGIT LEAD 2 -- the control the first pass could not run.
508 established the effect and failed to localize it. On 96 fresh
rows disjoint from the reviewer's sample, ablating r.2.0.1's
16-dimensional bundle in a6 and a8 damages digit targets by
+0.0900 (95% CI +0.044 to +0.142) while the population effect for
digits is -0.018 -- digits are normally SPARED. Ten rank-matched
random subspaces in the same components give -0.015 to +0.015, so
this is not what removing sixteen arbitrary directions does. The
instrument checks out on the same rows: punctuation lands at
-0.0371, bracketing its population -0.025.
What failed was the leaf-specificity control, and it failed for a
reason worth its own line in the ledger: the three "peer" leaves
returned +0.0900, +0.0900, +0.0900 -- identical to four decimals,
because r.2.0.0, r.2.0.1, r.2.0.2 and r.2.0.3 ARE THE SAME LEAF.
Same four probes, same 864 members, listed four times with
different tags. Thirty-six of the census tree's 311 tags are
duplicates of this kind in fourteen groups. The peer control
compared the leaf against itself.
The right control does not need peers at all. If the effect
belongs to these particular directions, then OTHER directions of
the same rank in the SAME components must not reproduce it. The
bundle is PCA blocks (0,4) and (4,16) of a6 and a8 conditioned on
slice r.2.0; the alternatives are the next blocks along the same
spectrum -- (16,32) and (32,48) -- which are the same components,
the same slice conditioning, the same construction, the same rank,
and different directions. That is a far stronger control than a
random subspace, because it holds everything fixed except which
part of the component's output spectrum is removed.
Also asked, since it costs nothing: which half of the bundle
carries it, the top-4 block or the 4-to-16 block.
REGISTERED PREDICTIONS:
  (a) REPLICATION on a third disjoint sample (skip=400): the digit
      dissociation is positive with a CI excluding zero;
  (b) DIRECTION-SPECIFIC: the real bundle's digit dissociation
      exceeds every alternative-span bundle's by at least 0.03.
      If the (16,32) and (32,48) blocks do the same thing, the
      effect belongs to components a6/a8 wholesale and the leaf
      has no private function -- report that plainly;
  (c) SPLIT REPORTED: the (0,4) and (4,16) halves are measured
      separately, and their sum is compared to the whole so that
      superadditivity is visible rather than assumed.
  SANITY NULL, unchanged: punctuation on the same rows must land
      in [-0.040, -0.010]. Outside that, the run is uninformative.
No peer-leaf arm, because the census tree cannot currently supply
one for this bundle."""
import ast, json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'digit_lead2_results.json'
TAG='r.2.0.1'; NFRESH=96; SKIP=400

@torch.no_grad()
def ce_rows(rows,hooks_fn=None):
    out=torch.zeros(rows.shape[0],T)
    for i in range(0,rows.shape[0],4):
        bb=rows[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=bb.shape[0]
        hs=hooks_fn() if hooks_fn else []
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

def boot(base,abl,mask,n=400,seed=11):
    g=torch.Generator().manual_seed(seed); R=base.shape[0]; v=[]
    for _ in range(n):
        i=torch.randint(0,R,(R,),generator=g)
        x=diss(base[i],abl[i],mask[i])
        if x==x: v.append(x)
    v.sort(); return v[int(0.025*len(v))],v[int(0.975*len(v))]

def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    lf=cl.leaf(TAG)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    rows=cl.fineweb_rows(NFRESH,skip=SKIP)
    dig,pun=classes(rows)
    base=ce_rows(rows)
    arms={'real':probes,
          'half_0_4':[p for p in probes if tuple(p[3])==(0,4)],
          'half_4_16':[p for p in probes if tuple(p[3])==(4,16)],
          'alt_16_32':[(p[0],p[1],p[2],(16,32)) for p in probes
                       if tuple(p[3])==(4,16)]
                      +[(p[0],p[1],p[2],(32,36)) for p in probes
                        if tuple(p[3])==(0,4)],
          'alt_32_48':[(p[0],p[1],p[2],(32,48)) for p in probes
                       if tuple(p[3])==(4,16)]
                      +[(p[0],p[1],p[2],(48,52)) for p in probes
                        if tuple(p[3])==(0,4)]}
    res={}
    for nm,pr in arms.items():
        a=ce_rows(rows,lambda pr=pr: cl.proj_hooks(pr))
        rank=dict(cl.LAST_PROJ_RANK)
        dv=diss(base,a,dig); pv=diss(base,a,pun)
        lo,hi=boot(base,a,dig)
        res[nm]={'digit':round(dv,4),'punct':round(pv,4),
                 'ci':[round(lo,4),round(hi,4)],'rank':rank}
        print(f'{nm}: digit {dv:+.4f} CI [{lo:+.4f},{hi:+.4f}] | '
              f'punct {pv:+.4f} | rank {rank}',flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    R=res['real']
    sane=-0.040<=R['punct']<=-0.010
    va,_=cl.score_bar('a',R['ci'][0],1e-9)
    worst_alt=max(res['alt_16_32']['digit'],res['alt_32_48']['digit'])
    vb,_=cl.score_bar('b',R['digit']-worst_alt,0.03)
    halfsum=res['half_0_4']['digit']+res['half_4_16']['digit']
    print(f"(c) halves {res['half_0_4']['digit']:+.4f} + "
          f"{res['half_4_16']['digit']:+.4f} = {halfsum:+.4f} vs "
          f"whole {R['digit']:+.4f}: reported")
    print(f"SANITY (punct {R['punct']:+.4f} in [-0.040,-0.010]): "
          f"{'ok' if sane else 'VIOLATED -- run uninformative'}")
    out={'arms':res,'half_sum':round(halfsum,4),
         'best_alternative':round(worst_alt,4),
         'population_digit':-0.018,'sanity_ok':bool(sane),
         'pred_a':va=='HELD','pred_b':vb=='HELD','pred_c':True,
         'n_rows':NFRESH,'skip':SKIP,'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
