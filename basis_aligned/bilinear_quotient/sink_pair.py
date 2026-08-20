"""SINK PAIR -- 452: the position-0 census (451) found only TWO
sinks in 162 heads, and both sit in layer 5: head 5.7 (99.7% of
top reads at position 0, deletion cost 0.916) and head 5.2 (67.6%,
deletion cost 0.018). One is the model's most expensive head, the
other is nearly free. Ask what distinguishes them.
Measured: (i) is 5.2's write also near-constant across positions;
(ii) is its constant on the same axis as 5.7's; (iii) is deleting
both worse than the sum of deleting each.
REGISTERED PREDICTIONS:
  (a) 5.2 IS ALSO A BROADCASTER: the mean cosine between its
      per-position write and its own mean write is >= 0.9;
  (b) SAME AXIS: |cos| between 5.2's constant and 5.7's is
      >= 0.5 (they carry the same stream-centre direction, one
      redundantly);
  (c) SUPERADDITIVITY: deleting both costs more than the sum of
      the individual deletions (0.916 + 0.018) by >= 0.05 nats --
      or report the shortfall if the pair is redundant."""


import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sink_pair_results.json'
NR=32; LJ=5; HD=7
ARMS=['zero7','zero2','zero_both']

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    acc={a:[0.0,0] for a in ARMS}
    GEOM={}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4
        def run(mode):
            hs=[]
            if mode is not None:
                at=m.transformer.h[LJ].attn
                def fh(mo_,args,o_,at=at,mode=mode):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                    cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                    qf=F.rms_norm(at.c_q(X).view(B,T,9,128),(128,))
                    kf=F.rms_norm(at.c_k(X).view(B,T,9,128),(128,))
                    qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                    q2=F.rms_norm(at.c_q2(X).view(B,T,9,128),
                                  (128,))
                    k2=F.rms_norm(at.c_k2(X).view(B,T,9,128),
                                  (128,))
                    q2,k2=are(q2,cos,sin),are(k2,cos,sin)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    pat=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,
                                   vm.float())
                    GEOM.setdefault('w7',[]).append(
                        z[:,7].detach().clone())
                    GEOM.setdefault('w2',[]).append(
                        z[:,2].detach().clone())
                    if mode=='zero7': z[:,7]=0
                    elif mode=='zero2': z[:,2]=0
                    elif mode=='zero_both': z[:,7]=0; z[:,2]=0
                    ynew=at.c_proj(z.transpose(1,2).contiguous()
                                   .view(B,T,-1).to(X.dtype))
                    return (ynew,v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').mean().item()
            for h in hs: h.remove()
            return ce
        base=run(None)
        for a in ARMS:
            acc[a][0]+=run(a)-base; acc[a][1]+=1
        print(f'batch {i} done',flush=True)
    CE={a:round(v[0]/max(v[1],1),4) for a,v in acc.items()}
    print('dCE overall:',CE)
    at5=m.transformer.h[5].attn
    Wp=at5.c_proj.weight.float()
    def const_of(key,hd):
        Z=torch.cat(GEOM[key])            # (N,T,128)
        W=Z.reshape(-1,128)@Wp[:,hd*128:(hd+1)*128].T
        mu=W.mean(0)
        cs=F.cosine_similarity(W,mu[None].expand_as(W),dim=-1)
        return mu,float(cs.mean())
    mu7,c7=const_of('w7',7)
    mu2,c2=const_of('w2',2)
    axis=float(F.cosine_similarity(mu7,mu2,dim=0))
    pa=c2>=0.9
    pb=abs(axis)>=0.5
    sumind=CE['zero7']+CE['zero2']
    pc=(CE['zero_both']-sumind)>=0.05
    CE['sum_individual']=round(sumind,4)
    CE['superadditivity']=round(CE['zero_both']-sumind,4)
    CE['selfcos_5_7']=round(c7,4); CE['selfcos_5_2']=round(c2,4)
    CE['axis_cos_5_7_vs_5_2']=round(axis,4)
    out={'dce_overall':CE,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'runtime_s':time.time()-t0}
    print(f"self-cosine: 5.7 {c7:.3f}, 5.2 {c2:.3f} | axis cos "
          f"{axis:+.3f} | sum {sumind:.4f} vs both "
          f"{CE['zero_both']:.4f}")
    for nm,v in (('a','5.2 is also a broadcaster (self-cos>=0.9)'),
                 ('b','same axis as 5.7 (|cos|>=0.5)'),
                 ('c','superadditive by >=0.05')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
