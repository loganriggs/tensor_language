"""Is the one-layer coherence length intrinsic, or a targeting artifact?

§59: steering along a deep reader's DIRECT-PATH coupling eigvector dies within a layer
or two. But the true end-to-end sensitivity direction -- the gradient of the deep
coefficient w.r.t. an additive perturbation at L1's output -- accounts for everything
layers 2..j do to the perturbation, and costs one backward pass. REGISTERED
PREDICTIONS:
  (a) gradient steering beats the direct-path eigvector at every deep target
      (L5, L9, L13), by >= 2x own-movement at L9 and L13;
  (b) the decisive one: gradient steering achieves own-movement >= 0.5 sigma at L13.
      HELD -> the range limit was a targeting artifact and functional tracking
      restores long-range control. FAILED -> the limit is intrinsic: even the exact
      sensitivity direction cannot carry functional identity through the stack at
      this magnitude.
Control: cross-talk measured as before; gradient steering should remain selective
(>= 2x) where it works."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152; K=48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_gradient_steering_results.json')

def run_forward(idx, targets, steer=None, need_grad=False):
    B,T=idx.shape
    delta=torch.zeros(D,device=DEV,requires_grad=need_grad)
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    outc={}
    maxli=max(l for l,_ in targets)
    for li in range(maxli+1):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
        a=blk.attn
        hcur=F.rms_norm(x,(D,))
        def qk(l):
            z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
            return apply_rot(z,cosb,sinb)
        v=a.c_v(hcur).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
        s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0)
        x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        for t_i,(tl,M) in enumerate(targets):
            if tl==li:
                outc[t_i]=torch.einsum('bti,ij,btj->bt',
                                       xhat.float(),M,xhat.float())
        mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        if li==1:
            add=delta
            if steer is not None:
                vec,mag=steer; add=add+mag*vec
            mo=mo+add.to(mo.dtype)
        x=x+mo
    return outc, delta

@torch.no_grad()
def collect_basis():
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs); return (Y1-Y1.mean(0)).float()

def main():
    t0=time.time()
    Y1c=collect_basis()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    s_out=float(Y1c.norm(dim=1).mean())/K**0.5
    mag=2*s_out*K**0.5*0.2
    targets=[]; direct=[]
    for j in (5,9,13):
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:8].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        best=None
        for f in range(8):
            c=mlp.Down.weight.detach().float().T@P[:,f]
            Bm=torch.einsum('k,ka,kb->ab',c,L,R); Bm=0.5*(Bm+Bm.T)
            if best is None or Bm.norm()>best[0].norm():
                best=(Bm,P[:,f].float())
        Bm,dvec=best
        targets.append((j,form_for_direction(mlp,dvec).float()))
        ev,U=torch.linalg.eigh(Bm.double())
        u=(V@U[:,ev.abs().argmax()].float()); direct.append(u/u.norm())
    rows=FW[300:312,:257].to(DEV)
    with torch.no_grad():
        base,_=run_forward(rows,targets)
        sig={i: float(base[i].std()) for i in base}
    out={'cases':[]}
    for i,(j,_) in enumerate(targets):
        # gradient direction: d mean(c_i) / d delta
        outc,delta=run_forward(rows,targets,need_grad=True)
        outc[i].mean().backward()
        gvec=delta.grad.detach()
        gvec=gvec/gvec.norm()
        with torch.no_grad():
            pg,_=run_forward(rows,targets,steer=(gvec,mag))
            own_g=abs(float((pg[i]-base[i]).mean()))/sig[i]
            others_g=[abs(float((pg[k]-base[k]).mean()))/sig[k]
                      for k in range(len(targets)) if k!=i]
            med_g=sorted(others_g)[len(others_g)//2]
            pd,_=run_forward(rows,targets,steer=(direct[i],mag))
            own_d=abs(float((pd[i]-base[i]).mean()))/sig[i]
        out['cases'].append({'layer':j,'own_grad':own_g,'own_direct':own_d,
                             'crosstalk_grad':med_g,
                             'cos_grad_direct':float(gvec@direct[i])})
        print(f'L{j:2d}: gradient own {own_g:.2f}s (direct {own_d:.2f}s) | '
              f'cross {med_g:.2f}s | cos(grad,direct) '
              f'{float(gvec@direct[i]):+.2f}',flush=True)
    deep=[c for c in out['cases'] if c['layer'] in (9,13)]
    pa=all(c['own_grad']>=2*max(c['own_direct'],1e-6) for c in deep)
    l13=[c for c in out['cases'] if c['layer']==13][0]
    pb=l13['own_grad']>=0.5
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\n(a) gradient >= 2x direct at L9/L13: {'HELD' if pa else 'FAILED'}")
    print(f"(b) gradient restores >= 0.5s at L13: "
          f"{'HELD (targeting artifact)' if pb else 'FAILED (intrinsic limit)'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
