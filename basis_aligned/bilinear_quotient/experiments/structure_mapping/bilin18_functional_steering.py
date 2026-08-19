"""The causal payoff: single-functional steering despite dense support.

§57: readers' signed functionals are near-orthogonal on a shared support. If real,
targeted intervention should be possible in FUNCTIONAL coordinates even though no
direction subspace separates paths: perturbing L1's output y along the top eigenvector
of ONE reader-form's coupling matrix B_{j,d} should move THAT coefficient specifically.
For six (reader, form) pairs across layers 2/5/13: steer y by +2 sigma along argmax
eigvec of B_{j,d}, measure Delta c_{j,d} (own, in own-sigma) and the median |Delta| of
the five OTHER sampled coefficients (cross-talk). REGISTERED PREDICTIONS:
  (a) own-coefficient movement >= 3x median cross-talk in every case;
  (b) mean selectivity ratio >= 5x.
Control: steering along a random direction of matched norm -- own-movement should be
< 0.5 sigma (the functional eigvec is doing the work)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152; K=48
STEER=None
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_functional_steering_results.json')

@torch.no_grad()
def run(idx, targets):
    """targets: list of (layer, M). Returns coefficients per target; STEER=(vec,mag)
    adds mag*vec to L1's MLP output."""
    B,T=idx.shape
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
        if li==1 and STEER is not None:
            vec,mag=STEER; mo=mo+(mag*vec).to(mo.dtype)
        x=x+mo
    return outc

def main():
    global STEER
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs); Y1c=(Y1-Y1.mean(0)).float()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    s_out=float(Y1c.norm(dim=1).mean())/K**0.5
    targets=[]; Bmats=[]
    for j,fidx in ((2,0),(2,3),(5,0),(5,5),(13,1),(13,4)):
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        dvec=orth(Vhj[:8].T)[:,fidx].float()
        M=form_for_direction(m.transformer.h[j].mlp,dvec).float()
        targets.append((j,M))
        L=m.transformer.h[j].mlp.Left.weight.detach().float()@V
        R=m.transformer.h[j].mlp.Right.weight.detach().float()@V
        c=(m.transformer.h[j].mlp.Down.weight.detach().float().T@dvec)
        Bm=torch.einsum('k,ka,kb->ab',c,L,R); Bm=0.5*(Bm+Bm.T)
        Bmats.append(Bm)
    rows=FW[300:324,:257].to(DEV)
    STEER=None; base=run(rows,targets)
    sig={i: float(base[i].std()) for i in base}
    g=torch.Generator(device=DEV).manual_seed(0)
    out={'cases':[]}
    sels=[]
    for i,Bm in enumerate(Bmats):
        ev,U=torch.linalg.eigh(Bm.double())
        u=(V@U[:,ev.abs().argmax()].float()); u=u/u.norm()
        mag=2*s_out*K**0.5*0.2
        STEER=(u,mag); pert=run(rows,targets); STEER=None
        own=abs(float((pert[i]-base[i]).mean()))/sig[i]
        others=[abs(float((pert[k]-base[k]).mean()))/sig[k]
                for k in range(len(targets)) if k!=i]
        med=sorted(others)[len(others)//2]
        ur=torch.randn(D,device=DEV,generator=g); ur=ur/ur.norm()
        STEER=(ur,mag); pr=run(rows,targets); STEER=None
        ctrl=abs(float((pr[i]-base[i]).mean()))/sig[i]
        sel=own/max(med,1e-6)
        sels.append(sel)
        out['cases'].append({'target':i,'own':own,'median_crosstalk':med,
                             'selectivity':sel,'random_ctrl':ctrl})
        print(f'target {i} (L{targets[i][0]}): own {own:.2f}s | cross-talk med '
              f'{med:.2f}s | selectivity {sel:.1f}x | random-ctrl {ctrl:.2f}s',
              flush=True)
    pa=all(s>=3 for s in sels); pb=sum(sels)/len(sels)>=5
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\n(a) every case >= 3x: {'HELD' if pa else 'FAILED'} | "
          f"(b) mean >= 5x: {'HELD' if pb else 'FAILED'} "
          f"(mean {sum(sels)/len(sels):.1f}x)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
