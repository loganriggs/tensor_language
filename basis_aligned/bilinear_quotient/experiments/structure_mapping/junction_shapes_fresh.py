"""JUNCTION SHAPES -- FRESH CERTIFICATION: 294 replicated the three
junction COSTS on never-seen text; 295-296 typed their SHAPES (b0
distributed, b1 coherence-inverted, b5 narrow) on the standard windows.
Certify the shapes fresh: per junction, full cut vs top-4 and top-16
partial cuts (direction bases refit on fresh data).
REGISTERED PREDICTIONS (fresh):
  (a) b5: top-4 >= 60% of full (narrow);
  (b) b1: top-4 >= 1.5x full (coherence inversion);
  (c) b0: top-16 <= 20% of full (distributed)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
from circuit_dictionary import classify, CLS
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'junction_shapes_fresh_results.json'
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
    base=evalF([])
    out={'base':round(base,4),'blocks':{}}
    shapes={}
    for li in (0,1,5):
        Vh=fit_dirs(li)
        full=evalF(mk_partial(li,None))-base
        t4=evalF(mk_partial(li,orth(Vh[:4].T)))-base
        t16=evalF(mk_partial(li,orth(Vh[:16].T)))-base
        shapes[li]=(full,t4,t16)
        out['blocks'][li]={'full':round(full,4),'top4':round(t4,4),
                           'top16':round(t16,4)}
        print(f'block {li}: full {full:+.4f} top4 {t4:+.4f} '
              f'top16 {t16:+.4f}',flush=True)
    pa=shapes[5][1]>=0.6*shapes[5][0]
    pb=shapes[1][1]>=1.5*shapes[1][0]
    pc=shapes[0][2]<=0.2*shapes[0][0]
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"(a) b5 narrow: {'HELD' if pa else 'FAILED'}")
    print(f"(b) b1 inversion: {'HELD' if pb else 'FAILED'}")
    print(f"(c) b0 distributed: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
