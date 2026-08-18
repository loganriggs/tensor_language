"""VALUE-LEVEL LEXICALITY: replace attn0/attn1's c_v outputs with
per-token value tables (empirical token-conditional mean of c_v(hcur) at
each position, fit window A) at ALL positions, eval window C. If these
heads are lexical infrastructure (section 239), the values are token-
determined even though outputs are mixtures. Note: tabling attn0's values
also changes the v1 broadcast that every later layer lamb-mixes -- that
broadcast IS part of attn0's value role and is included deliberately.

REGISTERED PREDICTIONS: (a) attn1 v-table recovers >= 60% of attn1's full
output ablation; (b) attn0 v-table >= 50% (v1 broadcast included);
(c) control: attn4 (induction band) v-table reported, registered uncertain;
(d) shuffled-vocab v-table for attn1 <= 0%."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import attn_mean
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'value_tables_results.json'
CA,CB=300,512; R0,R1=120,300
LAYERS=[0,1,4]

@torch.no_grad()
def main():
    t0=time.time()
    sums={li:torch.zeros(V,D,device=DEV) for li in LAYERS}
    cnt=torch.zeros(V,device=DEV)
    caps={}
    hs=[]
    for li in LAYERS:
        cv=m.transformer.h[li].attn.c_v
        def mk(li=li):
            return lambda mo_,i_,o_: caps.__setitem__(
                li,o_.detach().reshape(-1,D).float())
        hs.append(cv.register_forward_hook(mk()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        m(idx, bb[:,1:].contiguous())
        ids=idx.reshape(-1)
        cnt.index_add_(0,ids,torch.ones_like(ids,dtype=torch.float))
        for li in LAYERS:
            sums[li].index_add_(0,ids,caps[li])
    for h in hs: h.remove()
    glob={li:sums[li].sum(0)/cnt.sum() for li in LAYERS}
    tabs={}
    for li in LAYERS:
        t=sums[li]/cnt.clamp_min(1)[:,None]
        t[cnt==0]=glob[li]
        tabs[li]=t.to(torch.float16)
    amus={li:attn_mean(li) for li in LAYERS}
    g=torch.Generator(device=DEV).manual_seed(0)
    shuf=torch.randperm(V,generator=g,device=DEV)
    cur={}
    def pertok(li=None,mode=None):
        hs=[]
        if li is not None:
            if mode=='ablate':
                mu=amus[li]
                def hook(mod,i_,o_):
                    out=o_[0] if isinstance(o_,tuple) else o_
                    new=mu[None,None,:].to(out.dtype).expand_as(out)
                    if isinstance(o_,tuple): return (new,)+o_[1:]
                    return new
                hs.append(m.transformer.h[li].attn
                          .register_forward_hook(hook))
            else:
                tab=tabs[li]
                def hookv(mod,i_,o_):
                    B,T,_=o_.shape
                    ids=cur['idx']
                    if mode=='shuffle': ids=shuf[ids]
                    return tab[ids].to(o_.dtype)
                hs.append(m.transformer.h[li].attn.c_v
                          .register_forward_hook(hookv))
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
    base=pertok()
    res={}
    for li in LAYERS:
        da=float((pertok(li,'ablate')-base).mean())
        dv=float((pertok(li,'vtable')-base).mean())
        r=1-dv/max(da,1e-6)
        res[li]={'ablate':round(da,4),'vtable':round(dv,4),
                 'recovery':round(r,3)}
        if li==1:
            ds=float((pertok(li,'shuffle')-base).mean())
            res[li]['shuffle_recovery']=round(1-ds/max(da,1e-6),3)
        print(f'attn{li}: ablate {da:+.3f} vtable {dv:+.3f} rec {r:.0%}'
              +(f" shuf {res[li]['shuffle_recovery']:.0%}" if li==1 else ''),
              flush=True)
    pa=res[1]['recovery']>=0.60; pb=res[0]['recovery']>=0.50
    pd=res[1].get('shuffle_recovery',1)<=0.0
    out={'layers':{str(k):v for k,v in res.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_d':bool(pd)}
    print(f"(a) attn1 vtable >=60%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) attn0 vtable >=50%: {'HELD' if pb else 'FAILED'}")
    print(f"(d) shuffled <=0%: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
