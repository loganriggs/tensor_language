"""DIGIT LEAD -- the one behavioural lead that points the wrong way.
Writeup 464 measured what ablating a 16-dimensional probe bundle
does to each token class across the population: punctuation -0.025
and digits -0.018 (both SPARED, i.e. the model predicts them
better without the bundle), capitalized +0.006, space-words
+0.014, newline +0.027. Every behavioural claim the swarm has
produced since has pointed the same way as that profile, which is
why four consecutive reviews returned WEAKEN: a leaf restating the
general bias is not a leaf with a private function.
r.2.0.1 is the exception. On fresh text its bundle DAMAGES digit
targets by +0.107 where the population SPARES them by -0.018 --
opposite sign, six times the magnitude -- and damages capitalized
targets by +0.051 against a population +0.006. The wave-7 reviewer
found this and explicitly declined to certify it: the sweep
covered five classes with no correction and only three control
seeds. A lead that contradicts the population direction cannot be
a restatement of the population effect, so it earns one dedicated
pre-registered test, which is this.
Single hypothesis, declared before running: DIGITS. Capitalized is
measured but reported as descriptive only, and is not scored.
Fresh rows disjoint from the reviewer's sample (skip=200).
Controls, all three of which the first pass lacked:
  RANDOM   ten rank-matched random subspaces in the same
           components, not three;
  PEER     every other leaf in the diverse tree whose bundle uses
           the same components -- if a neighbour's bundle does the
           same thing, the effect belongs to the components and
           not to this leaf;
  SANITY   the punctuation dissociation on the SAME rows, which
           must land near its population value. If the instrument
           cannot reproduce the known effect here, nothing else it
           reports on these rows is trustworthy.
REGISTERED PREDICTIONS:
  (a) IT REPLICATES: the digit dissociation is positive and its
      95% row-bootstrap CI excludes zero;
  (b) IT CONTRADICTS THE POPULATION: the dissociation exceeds
      +0.020, which is a sign flip from -0.018 plus a margin;
  (c) IT IS THIS LEAF'S: the dissociation exceeds the largest of
      the ten random subspaces AND the largest peer bundle.
  SANITY NULL: punctuation on the same rows lands in
      [-0.040, -0.010], bracketing the population -0.025. Outside
      that range the run is uninformative and is reported as such
      rather than scored.
Every bar goes through cl.score_bar, and the projector rank is
reported from cl.LAST_PROJ_RANK (writeup 504)."""
import ast, json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'digit_lead_results.json'
TAG='r.2.0.1'; NFRESH=96; SKIP=200; NRAND=10

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
    cap=torch.zeros_like(nxt,dtype=torch.bool)
    for r in range(nxt.shape[0]):
        for q in range(nxt.shape[1]):
            s=cl.d1(int(nxt[r,q])); t=s.strip()
            if t and t[0].isdigit(): dig[r,q]=True
            elif t and all(not c.isalnum() for c in t): pun[r,q]=True
            elif t and t[0].isupper(): cap[r,q]=True
    return {'digit':dig,'punct':pun,'capitalized':cap}

def diss(base,abl,mask):
    d=abl-base
    inm=float(d[mask].mean()) if int(mask.sum()) else float('nan')
    out=float(d[~mask].mean())
    return inm-out,int(mask.sum())

def boot(base,abl,mask,n=400,seed=7):
    g=torch.Generator().manual_seed(seed)
    R=base.shape[0]; vals=[]
    for _ in range(n):
        idx=torch.randint(0,R,(R,),generator=g)
        v,_=diss(base[idx],abl[idx],mask[idx])
        if v==v: vals.append(v)
    vals.sort()
    return (vals[int(0.025*len(vals))],vals[int(0.975*len(vals))])

def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    lf=cl.leaf(TAG)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    keys=sorted({p[1] for p in probes if p[0]=='pca'})
    rows=cl.fineweb_rows(NFRESH,skip=SKIP)
    CL=classes(rows)
    base=ce_rows(rows)
    abl=ce_rows(rows,lambda: cl.proj_hooks(lf['top_probes']))
    rank=dict(cl.LAST_PROJ_RANK)
    res={}
    for nm,mk in CL.items():
        v,n=diss(base,abl,mk)
        lo,hi=boot(base,abl,mk)
        res[nm]={'dissociation':round(v,4),'n':n,
                 'ci':[round(lo,4),round(hi,4)]}
        print(f'{nm}: dissociation {v:+.4f} (n={n}) '
              f'CI [{lo:+.4f}, {hi:+.4f}]',flush=True)
    # random rank-matched controls in the same components
    rnd=[]
    for s in range(NRAND):
        def hf(s=s):
            hs=[]
            for k in keys:
                g=torch.Generator(device=DEV).manual_seed(1000+s)
                V=torch.randn(rank.get(k,16),D,generator=g,
                              device=DEV)
                P=cl.orth(V.T)
                mod=cl.MODS[k]
                if k[0]=='a':
                    def fh(mo,i_,o_,P=P):
                        y,v1=o_
                        yf=y.float().reshape(-1,D)
                        return ((yf-(yf@P)@P.T).view(y.shape)
                                .to(y.dtype),v1)
                else:
                    def fh(mo,i_,o_,P=P):
                        yf=o_.float().reshape(-1,D)
                        return (yf-(yf@P)@P.T).view(o_.shape) \
                            .to(o_.dtype)
                hs.append(mod.register_forward_hook(fh))
            return hs
        a=ce_rows(rows,hf)
        v,_=diss(base,a,CL['digit']); rnd.append(round(v,4))
        print(f'  random seed {s}: digit dissociation {v:+.4f}',
              flush=True)
    # peer bundles on the same components
    peers={}
    for tg in cl.all_tags():
        if tg==TAG: continue
        pr=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in cl.leaf(tg)['top_probes']]
        if sorted({p[1] for p in pr if p[0]=='pca'})!=keys: continue
        peers[tg]=None
        if len(peers)>=6: break
    for tg in list(peers):
        a=ce_rows(rows,lambda tg=tg: cl.proj_hooks(
            cl.leaf(tg)['top_probes']))
        v,_=diss(base,a,CL['digit']); peers[tg]=round(v,4)
        print(f'  peer {tg}: digit dissociation {v:+.4f}',flush=True)
    dg=res['digit']; pv=res['punct']['dissociation']
    sane=-0.040<=pv<=-0.010
    va,_=cl.score_bar('a',dg['ci'][0],1e-9)
    vb,_=cl.score_bar('b',dg['dissociation'],0.020)
    beat=max(rnd+[v for v in peers.values() if v is not None]
             or [0.0])
    vc,_=cl.score_bar('c',dg['dissociation']-beat,1e-9)
    print(f"SANITY (punct {pv:+.4f} in [-0.040,-0.010]): "
          f"{'ok' if sane else 'VIOLATED -- run uninformative'}")
    out={'tag':TAG,'projector_rank':rank,'classes':res,
         'random_controls':rnd,'peer_controls':peers,
         'best_control':round(beat,4),
         'population_digit':-0.018,'population_punct':-0.025,
         'sanity_ok':bool(sane),
         'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':vc=='HELD','n_rows':NFRESH,'skip':SKIP,
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
