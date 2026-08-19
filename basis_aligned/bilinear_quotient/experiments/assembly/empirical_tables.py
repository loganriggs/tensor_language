"""EMPIRICAL TOKEN TABLES (discriminator for section 250): instead of
context-free fold states, use the EMPIRICAL token-conditional mean output
(fit on window A rows 300-512): the ceiling for ANY token-lookup
replacement. REGISTERED: (a) mlp0 empirical >= 60% (token-determined, just
not context-free); (b) mlp1 empirical within 10 points of its 79% fold
(the fold already saturates the token channel); (c) mlp2/3 empirical
>= 30% (token-conditional means beat context-free states by construction).
Original fold docstring follows:

WEIGHTS-FIRST FOLD (user direction): the bilinear architecture has no
activation nonlinearity, so where the stream is embedding-dominated a
layer's function folds into a per-vocabulary lookup table. Compute the
CONTEXT-FREE fold exactly: run every vocab token alone (length-1 sequence)
through the model, capture each early MLP's output -> tables[li][token].
Replace mlp0..mlp3 (full output) with table lookup on the current token,
one layer at a time, on window C; reference = full-output mean ablation.

REGISTERED PREDICTIONS: (a) mlp0 table recovers >= 80% of its ablation
damage (layer 0 is a token lookup -- its input is exactly rms(wte) plus
only attn0's contribution); (b) mlp1 >= 50%; (c) monotone decay
mlp0 > mlp1 > mlp2 > mlp3 (the fold degrades as context mixes in);
(d) control: vocab-shuffled table <= 0% for mlp0."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'empirical_tables_results.json'
R0,R1=120,300
LAYERS=[0,1,2,3]

@torch.no_grad()
def build_tables():
    # empirical token-conditional mean outputs on window A
    sums={li:torch.zeros(V,D,device=DEV) for li in LAYERS}
    cnt=torch.zeros(V,device=DEV)
    caps={}
    hs=[]
    for li in LAYERS:
        def mk(li=li):
            return lambda mo_,i_,o_: caps.__setitem__(
                li,o_.detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    for i in range(300,512,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        m(idx, bb[:,1:].contiguous())
        ids=idx.reshape(-1)
        cnt.index_add_(0,ids,torch.ones_like(ids,dtype=torch.float))
        for li in LAYERS:
            sums[li].index_add_(0,ids,caps[li])
    for h in hs: h.remove()
    tabs={}
    glob={li:sums[li].sum(0)/cnt.sum() for li in LAYERS}
    for li in LAYERS:
        t=sums[li]/cnt.clamp_min(1)[:,None]
        t[cnt==0]=glob[li]
        tabs[li]=t.to(torch.float16)
    return tabs

@torch.no_grad()
def main():
    t0=time.time()
    tabs=build_tables()
    print(f'tables built {time.time()-t0:.0f}s',flush=True)
    means={}
    for li in LAYERS:
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        means[li]=torch.cat(accs).mean(0).float()
    g=torch.Generator(device=DEV).manual_seed(0)
    shuf=torch.randperm(V,generator=g,device=DEV)
    def pertok(li=None,mode=None):
        hs=[]
        if li is not None:
            tab=tabs[li]; mu=means[li]
            def hook(mod,i_,o_,li=li):
                B,T,_=o_.shape
                if mode=='ablate':
                    return mu[None,None,:].to(o_.dtype).expand_as(o_)
                ids=cur['idx']
                if mode=='shuffle': ids=shuf[ids]
                return tab[ids].to(o_.dtype)
            hs.append(m.transformer.h[li].mlp.register_forward_hook(hook))
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['idx']=idx
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return torch.cat(ces)
    cur={}
    base=pertok()
    res={}
    for li in LAYERS:
        da=float((pertok(li,'ablate')-base).mean())
        dt=float((pertok(li,'table')-base).mean())
        r=1-dt/max(da,1e-6)
        res[li]={'ablate':round(da,4),'table':round(dt,4),
                 'recovery':round(r,3)}
        if li==0:
            ds=float((pertok(li,'shuffle')-base).mean())
            res[li]['shuffle_recovery']=round(1-ds/max(da,1e-6),3)
        print(f'mlp{li}: ablate {da:+.3f} table {dt:+.3f} rec {r:.0%}'
              +(f" shuf {res[li]['shuffle_recovery']:.0%}" if li==0 else ''),
              flush=True)
    recs=[res[li]['recovery'] for li in LAYERS]
    pa=recs[0]>=0.60; pb=abs(recs[1]-0.79)<=0.10
    pc=recs[2]>=0.30 and recs[3]>=0.30
    pd=res[0].get('shuffle_recovery',1)<=0.0
    out={'layers':{str(k):v for k,v in res.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'pred_d':bool(pd)}
    print(f"(a) mlp0 >=60%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) mlp1 within 10 of fold: {'HELD' if pb else 'FAILED'}")
    print(f"(c) mlp2/3 >=30%: {'HELD' if pc else 'FAILED'}")
    print(f"(d) shuffled <=0%: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
