"""REGULATOR ARC -- FRESH REPLICATION (ledger 22): the three-junction
handoff ranking and the targeted span explosion were measured on the
standard windows; before the arc is quotable both must replicate on
never-seen documents. Builds 120 fresh pile rows (dedup vs FW), then:
(1) handoff cuts for blocks 0, 1, 5 (the junctions) and 3, 9 (controls)
    on fresh text;
(2) span-coefficient variance base vs 4-direction cut on fresh text
    (span basis refit on fresh base run).
REGISTERED PREDICTIONS:
  (a) blocks 0, 1, 5 are the three largest handoff costs on fresh text,
      each >= 5x the block-3 and block-9 controls;
  (b) span explosion replicates: cut/base variance ratio >= 3;
  (c) magnitudes reported vs the standard-window values."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
from circuit_dictionary import classify, CLS
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'regulator_fresh_results.json'
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
    base=evalF([])
    costs={}
    for li in (0,1,5,3,9):
        costs[li]=evalF(mk_cut(li))-base
        print(f'block {li}: fresh handoff cut {costs[li]:+.4f}',
              flush=True)
    # span explosion on fresh
    blk=m.transformer.h[5]
    def hb5(mo,args):
        x,v1,x0=args
        cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
    ds_=[]; m6=[]
    h1=blk.register_forward_pre_hook(hb5)
    h2=blk.mlp.register_forward_pre_hook(
        lambda mo,args: ds_.append((args[0].float()
            -F.rms_norm(cur['xm'].float(),(D,))).reshape(-1,D)))
    h3=m.transformer.h[6].mlp.register_forward_hook(
        lambda mo,i_,o_: m6.append(o_.detach().float().reshape(-1,D)))
    for i in range(0,120,4):
        bb=FR[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in (h1,h2,h3): h.remove()
    Dd=torch.cat(ds_); Y6=torch.cat(m6)
    _,_,Vh=torch.linalg.svd(Dd[:40000],full_matrices=False)
    V4=orth(Vh[:4].T)
    _,_,V6h=torch.linalg.svd((Y6-Y6.mean(0))[:40000],full_matrices=False)
    S6=orth(V6h[:8].T)
    vb=float((Y6@S6).var(0).sum())
    m6b=[]
    hbb=blk.register_forward_pre_hook(hb5)
    def hm5(mo,args):
        x=args[0]
        alt=F.rms_norm(cur['xm'].float(),(D,))
        d=x.float()-alt
        return ((x.float()-(d@V4)@V4.T).to(x.dtype),)+args[1:]
    hmm=blk.mlp.register_forward_pre_hook(hm5)
    hc=m.transformer.h[6].mlp.register_forward_hook(
        lambda mo,i_,o_: m6b.append(o_.detach().float().reshape(-1,D)))
    for i in range(0,120,4):
        bb=FR[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in (hbb,hmm,hc): h.remove()
    vc=float((torch.cat(m6b)@S6).var(0).sum())
    ratio=vc/vb
    print(f'fresh span variance: base {vb:.0f} cut {vc:.0f} '
          f'ratio {ratio:.2f}',flush=True)
    top3=sorted(costs,key=lambda k:-costs[k])[:3]
    pa=(set(top3)=={0,1,5}
        and all(costs[li]>=5*max(costs[3],costs[9],1e-4)
                for li in (0,1,5)))
    pb=ratio>=3
    out={'base':round(base,4),
         'handoffs':{li:round(v,4) for li,v in costs.items()},
         'span_ratio':round(ratio,3),
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) junctions 0,1,5 top-3 and >=5x controls: "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) span explosion >=3x on fresh: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
