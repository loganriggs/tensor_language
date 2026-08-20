"""LEAF OUTPUT DECOMP -- 432: the input-side mechanism tool has
now returned ENRICHED_STABLE=False on five of seven swarm leaves
(r.5.0.1, r.18.2.0, r.5.3.1, r.13.2.1, r.2.0.1), with the one
positive (r.3.0.2) partly an adjacent-layer property. Leaf
selectivity is evidently not explained by WHICH WRITERS FEED the
machinery. Ask the complementary question: WHO CONSUMES IT. For a
leaf's probe bundle, ablate it and measure how much each
downstream component's input changes, member positions vs
off-slice positions in the same rows. The 430 lesson is designed
in from the start: every leaf is scored against a RANK-MATCHED
RANDOM SUBSPACE ablation in the same components, so a "consumer"
only counts if the real bundle beats the random one.
Usage: python leaf_output_decomp.py <tag> [<tag> ...]
REGISTERED PREDICTIONS:
  (a) CONCENTRATED CONSUMER: for >= 50% of tested leaves the top
      downstream consumer's member-vs-offslice change ratio is
      >= 1.3;
  (b) SPECIFICITY: for >= 50% of leaves the real bundle's top
      consumer differs from the random subspace's top consumer,
      or exceeds its ratio by >= 0.2;
  (c) report the full consumer profile per leaf (this is the
      artifact the swarm will quote)."""
import json, sys, ast, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV, orth
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'leaf_output_decomp_results.json'
MAXROWS=16

@torch.no_grad()
def profile(tag,random_subspace=False):
    lf=cl.leaf(tag); mem=lf['member']; sl=lf['slice']
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    NF=cl.nflat()
    memm=torch.zeros(NF,dtype=torch.bool); memm[mem]=True
    slm=torch.zeros(NF,dtype=torch.bool); slm[sl]=True
    g=torch.Generator().manual_seed(5)
    rr=(mem//256).unique()
    if len(rr)>MAXROWS:
        rr=rr[torch.randperm(len(rr),generator=g)[:MAXROWS]] \
            .sort().values
    rows=cl.rows()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    if random_subspace:
        rk={}
        for p in probes:
            key=p[1] if p[0] in ('comp','pca') else f'a{p[1]}'
            n=(p[3][1]-p[3][0]) if p[0]=='pca' else 8
            rk[key]=rk.get(key,0)+n
        def hooks():
            hs=[]
            for key,n in rk.items():
                gg=torch.Generator(device=DEV).manual_seed(202)
                P=orth(torch.randn(D,n,generator=gg,device=DEV))
                mod=MODS[key]
                if key[0]=='a':
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
    else:
        def hooks(): return cl.proj_hooks(lf['top_probes'])
    acc={}
    for i in range(0,len(rr),4):
        rid=rr[i:i+4]
        bb=rows[rid,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=len(rid)
        caps={}
        def cap(li,tagk):
            def h(mo_,args): caps[tagk]=args[0].detach().float()
            return h
        def sweep(with_hooks):
            caps.clear(); hs=[]
            for li in range(18):
                hs.append(m.transformer.h[li].attn
                          .register_forward_pre_hook(
                              cap(li,f'in_a{li}')))
            if with_hooks: hs+=hooks()
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            for h in hs: h.remove()
            return {k:v.clone() for k,v in caps.items()}
        base=sweep(False); abl=sweep(True)
        gi=(rid[:,None].to(DEV)*256
            +torch.arange(T,device=DEV)[None,:])
        mmask=memm.to(DEV)[gi]; omask=(~slm.to(DEV)[gi])
        for k in base:
            dv=(abl[k]-base[k]).norm(dim=-1) \
                /base[k].norm(dim=-1).clamp_min(1e-6)
            a=acc.setdefault(k,{'m':0.0,'nm':0,'o':0.0,'no':0})
            a['m']+=float(dv[mmask].sum()); a['nm']+=int(mmask.sum())
            a['o']+=float(dv[omask].sum()); a['no']+=int(omask.sum())
    prof={}
    for k,a in acc.items():
        mv=a['m']/max(a['nm'],1); ov=a['o']/max(a['no'],1)
        prof[k]={'member':round(mv,4),'offslice':round(ov,4),
                 'ratio':round(mv/max(ov,1e-6),3)}
    return prof

def main(tags):
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    res={}
    for tag in tags:
        try:
            torch.cuda.empty_cache()
            own=profile(tag,False); rnd=profile(tag,True)
            topo=max(own,key=lambda k:own[k]['ratio'])
            topr=max(rnd,key=lambda k:rnd[k]['ratio'])
            res[tag]={'own_profile':own,'random_profile':rnd,
                      'top_consumer':topo,
                      'top_ratio':own[topo]['ratio'],
                      'random_top_consumer':topr,
                      'random_top_ratio':rnd[topr]['ratio']}
            print(f"{tag}: top consumer {topo} ratio "
                  f"{own[topo]['ratio']} | random {topr} "
                  f"{rnd[topr]['ratio']}",flush=True)
            json.dump(res,open(OUT,'w'),indent=1)
        except Exception as e:
            print(f'{tag}: SKIPPED ({type(e).__name__}: {e})',
                  flush=True)
            torch.cuda.empty_cache()
    n=len(res)
    pa=(sum(1 for t in res if res[t]['top_ratio']>=1.3)
        /max(n,1))>=0.5
    pb=(sum(1 for t in res
            if res[t]['top_consumer']!=res[t]['random_top_consumer']
            or res[t]['top_ratio']-res[t]['random_top_ratio']>=0.2)
        /max(n,1))>=0.5
    out={'leaves':res,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':True,'runtime_s':time.time()-t0}
    for nm,v in (('a','>=50% have a top consumer ratio >=1.3'),
                 ('b','>=50% beat the random-subspace consumer'),
                 ('c','profiles reported')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__':
    ts=[a for a in sys.argv[1:] if not a.startswith('-')]
    main(ts or ['r.13.2.1','r.5.0.1','r.18.2.0','r.5.3.1',
                'r.2.0.1','r.3.0.2'])
