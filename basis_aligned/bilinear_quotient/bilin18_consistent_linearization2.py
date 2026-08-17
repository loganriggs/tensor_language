"""v2 -- v1's interchange arm was void: span-ablation hooks target blk.mlp, which
the manual fwd_lin never calls (d16/d17 read exactly zero; recorded). Spans are
now applied inside the forward. Individual-cost results from v1 stand (that arm
was valid): front-loading HELD. This rerun re-verifies only the interchange.

Consistent-protocol rerun after the section-105 lambda-mixing correction. All
stand-ins are fit AND applied inside the same manual forward (post-mix input),
via the fwd_lin machinery from bilin18_pipe_refit.

Measurements: (1) individual linearization cost for layers 2,4,5,7,9,13,16,17;
(2) the 16->17 interchange excess (span-ablation composition, section 102 design)
under real vs consistently-linearized L17.

REGISTERED PREDICTIONS: (a2) individual costs <=0.1 for all layers except L2, and
L2 >= 3x the median of the others (the front-loading claim, retested on a clean
instrument); (b2) linearizing L17 kills >= 70% of the 16->17 excess (the section
102 conclusion, re-verified); (c2) the consistent-L17 stand-in's base cost is
<= 0.05 (tighter than the contaminated +0.10)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
import bilin18_pipe_refit as PR
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_consistent_linearization2_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    PR.LINS={}
    base=PR.ce_eval()
    print(f'base {base:.4f}\n',flush=True)
    costs={2:0.1091,4:0.0542,5:0.0247,7:0.0358,9:0.0324,13:0.0459,
           16:0.0332,17:0.0963}   # v1 valid arm, carried over
    # interchange excess: span ablations at 16/17 outputs, real vs linearized L17
    spans={}
    for li in (16,17):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        spans[li]=(Q,Ybar@Q)
    # patch spans inside fwd_lin: easiest -- wrap mo via LINS-like registry
    # extend: monkey-patch a SPANS dict into PR.fwd_lin by post-processing mo.
    # Simpler: use hooks on the real model for span ablation ONLY when L17 is
    # real; for the linearized arm we ablate the stand-in's output by composing
    # the projection into the linear map (exact for a linear map).
    from tier2_model import rope_tables, apply_rot
    NH,HD=9,128
    def fwd_spans(idx, span_lis, lin17=None):
        B,T=idx.shape
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        for li in range(18):
            blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            xin=x
            a=blk.attn; hcur=F.rms_norm(x,(D,))
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
            if li==17 and lin17 is not None:
                xi=xin.reshape(-1,D).float()
                mo=((xi-lin17['bx'])@lin17['W']+lin17['by']).to(x.dtype).view_as(x)
            else:
                mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            if li in span_lis:
                Q,cbar=spans[li]
                c=mo.float().reshape(-1,D)@Q
                mo=mo-((c-cbar)@Q.T).to(mo.dtype).view_as(mo)
            x=x+mo
        lg=m.lm_head(F.rms_norm(x,(D,)))
        return (30*torch.tanh(lg/30)).float()
    def ce_hooked(span_lis, lin17=None):
        tot,n=0.0,0
        for i in range(300,380,4):
            b=FW[i:i+4,:257].to(DEV)
            lg=fwd_spans(b[:,:-1].contiguous(), span_lis, lin17)
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
            tot+=float(ce)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
        return tot/n
    L17MAP=PR.fit_layer(17)
    out={'costs':{str(k):v for k,v in costs.items()},'base':base}
    res={}
    for tag,l17 in (('real',None),('linearized',L17MAP)):
        b_=ce_hooked([],l17)
        d16=ce_hooked([16],l17)-b_
        d17=ce_hooked([17],l17)-b_
        joint=ce_hooked([16,17],l17)-b_
        res[tag]={'base':b_,'d16':d16,'d17':d17,'joint':joint,
                  'excess':joint-d16-d17}
        print(f'{tag:10s}: base {b_:.4f} | d16 {d16:+.4f} | d17 {d17:+.4f} | '
              f'excess {joint-d16-d17:+.4f}',flush=True)
    others=[v for k,v in costs.items() if k!=2]
    med=sorted(others)[len(others)//2]
    pa=all(v<=0.1 for k,v in costs.items() if k!=2) and costs[2]>=3*med
    drop=1-res['linearized']['excess']/res['real']['excess'] \
         if res['real']['excess']>1e-6 else float('nan')
    pb=drop>=0.7
    pc=abs(res['linearized']['base']-res['real']['base'])<=0.05
    out['interchange']=res; out['excess_drop']=drop
    out['pred_a2']=bool(pa); out['pred_b2']=bool(pb); out['pred_c2']=bool(pc)
    print(f"\n(a2) front-loading survives: {'HELD' if pa else 'FAILED'} "
          f"(L2 +{costs[2]:.3f} vs median +{med:.3f})")
    print(f"(b2) interaction-kill survives (>=70%): {'HELD' if pb else 'FAILED'} "
          f"({drop if drop==drop else 0:.0%})")
    print(f"(c2) consistent stand-in tighter (<=0.05): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
