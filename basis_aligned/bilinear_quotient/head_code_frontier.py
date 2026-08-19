"""HEAD CODE FRONTIER -- the coverage-vs-fidelity curve at head
grain with READABLE code. Measure each of the 162 heads' individual
one-read substitution cost on the census grid (one sweep per head),
then build the cheapest-first cumulative curve and evaluate the
actual joint substitution at three operating points (40, 80, 120
heads). Fresh leg at the 80-head point.
REGISTERED PREDICTIONS:
  (a) >=60 heads fit within +0.15 TOTAL joint cost;
  (b) joint cost at each point <=1.3x the sum of singles (near-
      additivity, as everywhere else in this model);
  (c) fresh cost at the 80-head point <=1.5x grid;
  (d) full curve written (the benchmark object)."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_code_frontier_results.json'

byl_ref=[{}]

def code_hooks():
    import sys as s_
    hs=[]
    for li,hds in byl_ref[0].items():
        at=m.transformer.h[li].attn
        def fh(mo_,args,out,li=li,hds=hds,at=at):
            y,v1r=out
            X=args[0]; v1=args[1] if args[1] is not None else v1r
            are=s_.modules[type(at).__module__].apply_rotary_emb
            Bb,Tq=X.shape[0],X.shape[1]
            v=at.c_v(X).view(Bb,Tq,9,128)
            vm=(1-at.lamb)*v+at.lamb*(v1.view_as(v)
                                      if v1 is not None else v)
            cos,sin=at.rotary(at.c_q(X).view(Bb,Tq,9,128))
            qf=F.rms_norm(at.c_q(X).view(Bb,Tq,9,128),(128,))
            kf=F.rms_norm(at.c_k(X).view(Bb,Tq,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at.c_q2(X).view(Bb,Tq,9,128),(128,))
            k2=F.rms_norm(at.c_k2(X).view(Bb,Tq,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
            s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
            pat=(sc*s2)*torch.tril(torch.ones(Tq,Tq,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
            for hd in hds:
                p1=pat[:,hd]
                ks=p1.abs().argmax(-1)
                w=p1.gather(-1,ks[...,None]).squeeze(-1)
                vv=vm[:,:,hd].float().gather(
                    1,ks[...,None].expand(-1,-1,128))
                z[:,hd]=w[...,None]*vv
            yn=at.c_proj(z.transpose(1,2).contiguous()
                         .view(Bb,Tq,-1).to(X.dtype))
            return (yn,v1r)
        hs.append(at.register_forward_hook(fh))
    return hs

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()
    base=cl.ce_sweep([])
    byl={}
    singles={}
    for li in range(18):
        for hd in range(9):
            byl={li:[hd]}
            byl_ref[0]=byl
            d=cl.ce_sweep(code_hooks())
            singles[(li,hd)]=float((d-base).mean())
        print(f'layer {li} singles done',flush=True)
    order=sorted(singles,key=lambda k:singles[k])
    curve=[]
    cum=0
    for k in order:
        cum+=singles[k]; curve.append(round(cum,4))
    import bisect
    n015=bisect.bisect_right(curve,0.15)
    print(f'sum-of-singles predicts {n015} heads within +0.15',
          flush=True)
    pts={}
    for N in (40,80,120):
        sel=order[:N]
        byl={}
        for li,hd in sel: byl.setdefault(li,[]).append(hd)
        byl_ref[0]=byl
        d=cl.ce_sweep(code_hooks())
        pts[N]={'joint':round(float((d-base).mean()),4),
                'sum_singles':round(sum(singles[k] for k in sel),4)}
        print(f'N={N}: {pts[N]}',flush=True)
    sel=order[:80]
    byl={}
    for li,hd in sel: byl.setdefault(li,[]).append(hd)
    byl_ref[0]=byl
    FRESH=cl.fresh_rows(120)
    bF=cl.ce_sweep([],tok=FRESH)
    cF=cl.ce_sweep(code_hooks(),tok=FRESH)
    fr=float((cF-bF).mean())
    # joint count within 0.15 from actual joints (interpolate via
    # the measured points; conservative: use largest N with joint<=0.15)
    nj=max([N for N in (40,80,120) if pts[N]['joint']<=0.15],
           default=0)
    pa=nj>=60 or n015>=60 and pts[min((N for N in (40,80,120)
        if N>=60),default=120)]['joint']<=0.15
    pb=all(p['joint']<=1.3*max(p['sum_singles'],1e-4)
           for p in pts.values())
    pc_=fr<=1.5*max(pts[80]['joint'],1e-3)
    out={'singles':{f'{k[0]}.{k[1]}':round(v,4)
                    for k,v in singles.items()},
         'cheapest_order':[f'{k[0]}.{k[1]}' for k in order[:120]],
         'cum_curve':curve,'points':pts,'fresh_80':round(fr,4),
         'n_within_015_sum':n015,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc_),
         'pred_d':True}
    print(f"(a) >=60 heads within +0.15: {'HELD' if pa else 'FAILED'}")
    print(f"(b) near-additive <=1.3x: {'HELD' if pb else 'FAILED'}")
    print(f"(c) fresh <=1.5x: {'HELD' if pc_ else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
