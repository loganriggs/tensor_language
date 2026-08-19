"""B0 BASIS STABILITY -- 297 found block 0's ladder shape differs by
window (top-16 = 10% of full on the standard window, 52% on fresh).
Decide why: fit the top-16 delta basis on window A and on fresh rows;
(1) subspace overlap between the two bases; (2) CROSS-BASIS cut -- the
window-A basis applied to fresh text.
REGISTERED PREDICTIONS:
  (a) cross-basis top-16 cut on fresh <= half of the fresh-basis top-16
      cut (the concentration is window-specific token content);
  (b) top-16 subspace overlap <= 0.3 (bases differ; floor 16/1152=0.014);
  (c) if instead overlap >= 0.6 and cross-basis cut >= 80%, the standard
      -window measurement was the outlier (either branch resolves)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
from circuit_dictionary import classify, CLS
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'b0_basis_stability_results.json'
CA,CB=300,512; R0,R1=120,300; LI=5

@torch.no_grad()
def main():
    t0=time.time()
    import tiktoken
    from datasets import load_dataset
    enc2=tiktoken.get_encoding('gpt2')
    ds=load_dataset('NeelNanda/pile-10k',split='train')
    seen={tuple(FW[r,:32].tolist()) for r in range(FW.shape[0])}
    rows=[]
    for di in range(5000,10000):
        tk=enc2.encode_ordinary(ds[di]['text'])
        for s0 in range(0,len(tk)-513,513):
            row=tk[s0:s0+513]
            if tuple(row[:32]) in seen: continue
            rows.append(row)
            if len(rows)>=120: break
        if len(rows)>=120: break
    FR=torch.tensor(rows,dtype=torch.long)
    print(f'fresh rows {FR.shape[0]}',flush=True)
    cur={}
    def evalF(hooks):
        ces=[]
        for i in range(0,120,4):
            bb=FR[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for b2 in m.transformer.h:
                x,v1=b2(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hooks: h.remove()
        return float(torch.cat(ces).mean())
    def mk_cut(li):
        blk=m.transformer.h[li]
        def hb(mo,args):
            x,v1,x0=args
            cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
        def hm(mo,args):
            x=args[0]
            return (F.rms_norm(cur['xm'].float(),(D,)).to(x.dtype),)                +args[1:]
        return [blk.register_forward_pre_hook(hb),
                blk.mlp.register_forward_pre_hook(hm)]
    from bilin18_joint_removal import orth
    def mk_partial(li,P):
        blk=m.transformer.h[li]
        def hb(mo,args):
            x,v1,x0=args
            cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
        def hm(mo,args):
            x=args[0]
            alt=F.rms_norm(cur['xm'].float(),(D,))
            if P is None:
                return (alt.to(x.dtype),)+args[1:]
            d=x.float()-alt
            return ((x.float()-(d@P)@P.T).to(x.dtype),)+args[1:]
        return [blk.register_forward_pre_hook(hb),
                blk.mlp.register_forward_pre_hook(hm)]
    def fit_dirs(li):
        blk=m.transformer.h[li]
        ds_=[]
        def hb(mo,args):
            x,v1,x0=args
            cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
        h1=blk.register_forward_pre_hook(hb)
        h2=blk.mlp.register_forward_pre_hook(
            lambda mo,args: ds_.append((args[0].float()
                -F.rms_norm(cur['xm'].float(),(D,))).reshape(-1,D)))
        for i in range(0,120,4):
            bb=FR[i:i+4,:257].to(DEV)
            m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        h1.remove(); h2.remove()
        Dd=torch.cat(ds_)
        _,_,Vh=torch.linalg.svd(Dd[:40000],full_matrices=False)
        return Vh
    from bilin18_joint_removal import orth
    LI=0
    blk=m.transformer.h[LI]
    def fit_dirs_tok(TOK,n):
        ds_=[]
        def hb(mo,args):
            x,v1,x0=args
            cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
        h1=blk.register_forward_pre_hook(hb)
        h2=blk.mlp.register_forward_pre_hook(
            lambda mo,args: ds_.append((args[0].float()
                -F.rms_norm(cur['xm'].float(),(D,))).reshape(-1,D)))
        for i in range(0,n,4):
            bb=TOK[i:i+4,:257].to(DEV)
            m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        h1.remove(); h2.remove()
        Dd=torch.cat(ds_)
        _,_,Vh=torch.linalg.svd(Dd[:40000],full_matrices=False)
        return orth(Vh[:16].T)
    VA=fit_dirs_tok(FW[300:420],120)
    VFr=fit_dirs_tok(FR,120)
    ov=float((VA.T@VFr).pow(2).sum())/16
    print(f'top-16 subspace overlap {ov:.3f} (floor 0.014)',flush=True)
    def mk_partial(P):
        def hb(mo,args):
            x,v1,x0=args
            cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
        def hm(mo,args):
            x=args[0]
            alt=F.rms_norm(cur['xm'].float(),(D,))
            if P is None: return (alt.to(x.dtype),)+args[1:]
            d=x.float()-alt
            return ((x.float()-(d@P)@P.T).to(x.dtype),)+args[1:]
        return [blk.register_forward_pre_hook(hb),
                blk.mlp.register_forward_pre_hook(hm)]
    base=evalF([])
    full=evalF(mk_partial(None))-base
    cf=evalF(mk_partial(VFr))-base
    ca=evalF(mk_partial(VA))-base
    print(f'fresh: full {full:+.4f} | fresh-basis top16 {cf:+.4f} | '
          f'windowA-basis top16 {ca:+.4f}',flush=True)
    pa=ca<=0.5*max(cf,1e-4)
    pb=ov<=0.3
    pc=ov>=0.6 and ca>=0.8*cf
    out={'overlap':round(ov,3),'full':round(full,4),
         'fresh_basis_top16':round(cf,4),'winA_basis_top16':round(ca,4),
         'pred_a':bool(pa),'pred_b':bool(pb),'branch_stable':bool(pc)}
    print(f"(a) cross-basis <= half: {'HELD' if pa else 'FAILED'}")
    print(f"(b) overlap <= 0.3: {'HELD' if pb else 'FAILED'}")
    print(f"(c) stable-branch: {pc}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
