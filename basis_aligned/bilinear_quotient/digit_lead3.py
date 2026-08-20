"""DIGIT LEAD 3 -- settle it across three disjoint samples.
The effect has now replicated twice: +0.0900 (CI +0.044/+0.142) on
rows 200-296, and +0.1459 (CI +0.089/+0.209) on rows 400-496,
against a population value of -0.018 where digits are SPARED. It
is also direction-specific: alternative PCA spans of the same rank
in the same components (a6, a8) give +0.0260 and +0.0086, so it is
these directions and not those components wholesale. And it is
additive -- the (0,4) and (4,16) halves give +0.0931 and +0.0524,
summing to +0.1455 against +0.1459 for the whole bundle.
But the second run VIOLATED its own sanity null: punctuation came
out at -0.0487, outside the registered bracket [-0.040, -0.010]
around its population value of -0.025. I registered that bracket
as a condition for the run to be informative, so the run is
reported as uninformative rather than scored, and the question it
raises is whether the population value travels across samples at
all -- the digit numbers also moved a long way between samples
(+0.090 to +0.146), which suggests sample-to-sample spread this
design has never measured.
So: the same arms on THREE disjoint samples (skip 600, 800, 1000),
reporting the spread rather than a single number.
REGISTERED PREDICTIONS:
  (a) CONSISTENT SIGN AND SIZE: the digit dissociation is positive
      in all three samples and its smallest value exceeds +0.020,
      the population sign-flip margin;
  (b) DIRECTION-SPECIFIC IN ALL THREE: in every sample the real
      bundle beats both alternative-span bundles by >= 0.03;
  (c) SANITY, now measured rather than assumed: report punctuation
      in all three samples. If it sits outside [-0.040,-0.010] in
      more than one, the registered bracket was wrong -- it was
      set from a single population estimate -- and the honest
      conclusion is that this instrument has sample spread the
      earlier runs hid, which weakens every single-sample class
      claim in this program, not just this one."""
import ast, json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'digit_lead3_results.json'
TAG='r.2.0.1'; NFRESH=96; SKIPS=(600,800,1000)

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
    ALL={}
    for SKIP in SKIPS:
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
         print(f'[skip={SKIP}] {nm}: digit {dv:+.4f} CI [{lo:+.4f},{hi:+.4f}] | '
               f'punct {pv:+.4f} | rank {rank}',flush=True)
     ALL[SKIP]=res
    def g(sk,arm,f='digit'): return ALL[sk][arm][f]
    digs=[g(k,'real') for k in SKIPS]
    puns=[g(k,'real','punct') for k in SKIPS]
    alts=[max(g(k,'alt_16_32'),g(k,'alt_32_48')) for k in SKIPS]
    print('\ndigit per sample:',[round(x,4) for x in digs])
    print('punct per sample:',[round(x,4) for x in puns])
    print('best alternative :',[round(x,4) for x in alts])
    va,_=cl.score_bar('a',min(digs),0.020)
    vb,_=cl.score_bar('b',min(d-a for d,a in zip(digs,alts)),0.03)
    outside=sum(1 for p in puns if not -0.040<=p<=-0.010)
    print(f'(c) punctuation outside the registered bracket in '
          f'{outside} of 3 samples: reported')
    out={'per_sample':{str(k):ALL[k] for k in SKIPS},
         'digit':[round(x,4) for x in digs],
         'punct':[round(x,4) for x in puns],
         'best_alt':[round(x,4) for x in alts],
         'n_punct_outside_bracket':outside,
         'pred_a':va=='HELD','pred_b':vb=='HELD','pred_c':True,
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
