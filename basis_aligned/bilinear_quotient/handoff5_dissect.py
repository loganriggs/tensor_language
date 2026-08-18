"""HANDOFF 5 DISSECTION -- 287 found attn5->mlp5 is the model's biggest
causal handoff (+2.14 nats when cut), sitting at the private-writer
anatomy, and the assembled model bypasses it entirely. What does attn5
hand mlp5? Rank ladder: cut only the top-k PCA directions of attn5's
contribution at mlp5's input (fit on window A), k in (4,16,64,256);
control: cut k=16 RANDOM directions (seeded).
REGISTERED PREDICTIONS:
  (a) k=16 top directions carry >=60% of the full cut cost (+2.14*0.6);
  (b) random-16 control <=20% of the k=16 cost;
  (c) monotone in k; k=256 >= 90% of full."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'handoff5_dissect_results.json'
CA=300; R0,R1=120,300; LI=5

@torch.no_grad()
def main():
    t0=time.time()
    blk=m.transformer.h[LI]
    cur={}
    def hb(mo,args):
        x,v1,x0=args
        cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
    # collect the handoff delta on window A: norm(xm+a) - norm(xm)
    ds=[]
    def hm_cap(mo,args):
        x=args[0]
        d=x.float()-F.rms_norm(cur['xm'].float(),(D,))
        ds.append(d.reshape(-1,D))
    h1=blk.register_forward_pre_hook(hb)
    h2=blk.mlp.register_forward_pre_hook(hm_cap)
    for i in range(CA,CA+40,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    h1.remove(); h2.remove()
    Dd=torch.cat(ds)
    _,_,Vh=torch.linalg.svd(Dd[:40000], full_matrices=False)
    g=torch.Generator(device=DEV).manual_seed(0)
    def run(P=None):
        hbb=blk.register_forward_pre_hook(hb)
        def hm(mo,args):
            x=args[0]
            alt=F.rms_norm(cur['xm'].float(),(D,))
            d=x.float()-alt
            if P is None:
                new=alt                       # full cut
            else:
                new=x.float()-(d@P)@P.T       # cut projected part only
            return (new.to(x.dtype),)+args[1:]
        hmm=blk.mlp.register_forward_pre_hook(hm)
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for b2 in m.transformer.h:
                x,v1=b2(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        hbb.remove(); hmm.remove()
        return float(torch.cat(ces).mean())
    base=run(P=torch.zeros(D,1,device=DEV))    # cut nothing
    full=run(None)-base
    print(f'full cut {full:+.4f}',flush=True)
    out={'base':round(base,4),'full':round(full,4),'k':{}}
    for kk in (4,16,64,256):
        c=run(orth(Vh[:kk].T))-base
        out['k'][kk]=round(c,4)
        print(f'k={kk:4d} top-dirs cut {c:+.4f}',flush=True)
    Rr=orth(torch.randn(D,16,device=DEV,generator=g))
    ctl=run(Rr)-base
    out['rand16']=round(ctl,4)
    print(f'random-16 control {ctl:+.4f}',flush=True)
    pa=out['k'][16]>=0.6*full
    pb=ctl<=0.2*max(out['k'][16],1e-4)
    ks=[out['k'][k] for k in (4,16,64,256)]
    pc=all(ks[i]<=ks[i+1]+0.02 for i in range(3)) and ks[-1]>=0.9*full
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"(a) top-16 >= 60% of full: {'HELD' if pa else 'FAILED'}")
    print(f"(b) random-16 <= 20% of top-16: {'HELD' if pb else 'FAILED'}")
    print(f"(c) monotone, k=256 >= 90%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
