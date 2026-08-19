"""HANDOFF-5 SEMANTICS -- 289 found attn5 hands mlp5 a 4-16 direction
channel worth 2.1 nats. What is IN those directions? Three probes:
(1) LOGIT LENS: decode each of the top-4 handoff directions through the
    unembedding; report top-8 tokens each (descriptive).
(2) CLASS CORRELATION: R^2 of one-hot function class (the 10-class
    oracle) predicting each direction's activation at window-A
    positions; null: shuffled labels.
(3) SPAN OVERLAP: mean cos^2 of the top-4 handoff directions with (i)
    mlp6's output span (the section-212 private span's writer, top-8
    output PCA) and (ii) mlp5's own output span. Random floor
    8/1152 = 0.007.
REGISTERED PREDICTIONS:
  (a) class R^2 >= 0.1 for >=2 of 4 directions (the channel carries
      function-class information), shuffled null <= 0.01;
  (b) overlap with mlp6's span >= 10x the random floor OR with mlp5's
      own span >= 10x (the handoff feeds the private-writer anatomy);
  (c) logit-lens tokens reported (descriptive, no bar)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
from circuit_dictionary import classify, CLS
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'handoff5_semantics_results.json'
CA,CB=300,512; LI=5

@torch.no_grad()
def main():
    t0=time.time()
    blk=m.transformer.h[LI]
    cur={}
    def hb(mo,args):
        x,v1,x0=args
        cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
    ds=[]; m5=[]; m6=[]
    def hm_cap(mo,args):
        x=args[0]
        ds.append((x.float()-F.rms_norm(cur['xm'].float(),(D,)))
                  .reshape(-1,D))
    h1=blk.register_forward_pre_hook(hb)
    h2=blk.mlp.register_forward_pre_hook(hm_cap)
    h3=m.transformer.h[5].mlp.register_forward_hook(
        lambda mo,i_,o_: m5.append(o_.detach().float().reshape(-1,D)))
    h4=m.transformer.h[6].mlp.register_forward_hook(
        lambda mo,i_,o_: m6.append(o_.detach().float().reshape(-1,D)))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in (h1,h2,h3,h4): h.remove()
    Dd=torch.cat(ds)
    _,_,Vh=torch.linalg.svd(Dd[:40000], full_matrices=False)
    V4=orth(Vh[:4].T)                      # (D,4)
    # (1) logit lens
    W=m.lm_head.weight.detach().float()
    lens={}
    for j in range(4):
        lg=W@V4[:,j]
        top=torch.topk(lg,8).indices.tolist()
        bot=torch.topk(-lg,8).indices.tolist()
        lens[j]={'top':[enc.decode([t]) for t in top],
                 'neg':[enc.decode([t]) for t in bot]}
        print(f'dir{j} +:{lens[j]["top"]}',flush=True)
        print(f'dir{j} -:{lens[j]["neg"]}',flush=True)
    # (2) class correlation
    cls=classify(CA,CB).reshape(-1).to(DEV)
    A=Dd@V4                                 # (N,4)
    Yoh=torch.zeros(len(cls),10,device=DEV)
    Yoh[torch.arange(len(cls)),cls]=1.0
    g=torch.Generator(device=DEV).manual_seed(0)
    perm=torch.randperm(len(cls),device=DEV,generator=g)
    # redo cleanly
    r2s=[]; r2n=[]
    clsp=cls[perm]
    for j in range(4):
        y=A[:,j]; vt=y.var()
        mu=torch.stack([y[cls==k].mean() if (cls==k).any() else y.mean()
                        for k in range(10)])
        r2s.append(float(1-((y-mu[cls])**2).mean()/vt))
        mup=torch.stack([y[clsp==k].mean() if (clsp==k).any()
                         else y.mean() for k in range(10)])
        r2n.append(float(1-((y-mup[clsp])**2).mean()/vt))
    print('class R2 per dir:',[round(v,3) for v in r2s],
          'null:',[round(v,3) for v in r2n],flush=True)
    # (3) span overlap
    def span8(Y):
        Yc=torch.cat(Y)
        _,_,Vh8=torch.linalg.svd((Yc-Yc.mean(0))[:40000],
                                 full_matrices=False)
        return orth(Vh8[:8].T)
    S5=span8(m5); S6=span8(m6)
    ov5=float((V4.T@S5).pow(2).sum())/4
    ov6=float((V4.T@S6).pow(2).sum())/4
    Rr=orth(torch.randn(D,8,device=DEV,generator=g))
    ovr=float((V4.T@Rr).pow(2).sum())/4
    print(f'overlap: mlp5-span {ov5:.3f} mlp6-span {ov6:.3f} '
          f'random {ovr:.3f} (floor ~0.007)',flush=True)
    na=sum(1 for v in r2s if v>=0.1)
    pa=na>=2 and max(r2n)<=0.01
    pb=ov6>=0.07 or ov5>=0.07
    out={'lens':lens,'class_r2':[round(v,3) for v in r2s],
         'class_r2_null':[round(v,3) for v in r2n],
         'overlap_mlp5':round(ov5,3),'overlap_mlp6':round(ov6,3),
         'overlap_rand':round(ovr,3),
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) class R2>=0.1 for >=2 dirs ({na}/4), null clean: "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) span overlap >= 10x floor: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
