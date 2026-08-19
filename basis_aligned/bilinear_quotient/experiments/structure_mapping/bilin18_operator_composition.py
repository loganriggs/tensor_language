"""Can the head-1/pattern routing be predicted from weights -- the operator-level
compositional calculus the scalar graph lacks?

User question (2026-08-17): why couldn't 'L0 -> attn1' and 'attn1 -> L1' compose to
predict the routing? Answer under test: scalar edges don't compose -- edges are
OPERATORS, and composition requires each edge's subspace signature. The injected
signal's image under each head's input circuits is computable from weights alone:
for the steered direction d (L0's leader, rms-scale-normalised), score each head h of
attn1 by
    qk(h) = ||W_q,h d||^2 + ||W_k,h d||^2 + ||W_q2,h d||^2 + ||W_k2,h d||^2
    v(h)  = ||W_v,h d||^2
(each normalised by the same norms averaged over random unit vectors, so scores are
enrichment ratios). REGISTERED PREDICTIONS, matching the measured routing:
  (a) head 1 ranks #1 among the 9 heads on qk-enrichment (it carries the edge, 96%);
  (b) head 1's qk-enrichment exceeds its v-enrichment (the route is pattern-dominant,
      54% vs 30%);
  (c) head 4's qk-enrichment ranks 2nd or 3rd (it carries 51%).
If these hold, the routing WAS derivable from weights, and the graph should store
operator signatures, not scalars."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
NH,HD,D=9,128,1152

def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs)
    _,_,Vh0=torch.linalg.svd((Y0-Y0.mean(0)).float(), full_matrices=False)
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    d=orth(Vh0[:32].T)[:,int(phi0.argmax())].float()
    a=m.transformer.h[1].attn
    g=torch.Generator(device=DEV).manual_seed(0)
    Rnd=torch.randn(D,64,device=DEV,generator=g)
    Rnd=Rnd/Rnd.norm(dim=0,keepdim=True)
    def head_norms(W,vec):
        z=(W.detach().float()@vec).view(NH,HD,-1) if vec.dim()>1 else \
          (W.detach().float()@vec).view(NH,HD)
        return (z**2).sum(1) if vec.dim()==1 else (z**2).sum(1).mean(-1)
    out={'heads':{}}
    qk_e=torch.zeros(NH,device=DEV); v_e=torch.zeros(NH,device=DEV)
    for W,is_v in ((a.c_q,False),(a.c_k,False),(a.c_q2,False),(a.c_k2,False),
                   (a.c_v,True)):
        sd=head_norms(W.weight,d)
        sr=head_norms(W.weight,Rnd)
        enr=sd/sr.clamp_min(1e-12)
        if is_v: v_e+=enr
        else: qk_e+=enr/4
    print(f"  {'head':>5} {'qk-enrichment':>14} {'v-enrichment':>13}")
    for h in range(NH):
        out['heads'][h]={'qk':float(qk_e[h]),'v':float(v_e[h])}
        print(f"  {h:>5} {float(qk_e[h]):>14.2f} {float(v_e[h]):>13.2f}",flush=True)
    order=qk_e.argsort(descending=True).tolist()
    pa=order[0]==1
    pb=float(qk_e[1])>float(v_e[1])
    pc=order.index(4)in(1,2)
    out['qk_ranking']=order
    out['pred_a_head1_first']=bool(pa)
    out['pred_b_h1_pattern_dominant']=bool(pb)
    out['pred_c_head4_2nd_or_3rd']=bool(pc)
    print(f"\nqk ranking: {order}")
    print(f"(a) head 1 first: {'HELD' if pa else 'FAILED'} | "
          f"(b) h1 qk>v: {'HELD' if pb else 'FAILED'} | "
          f"(c) head 4 in top-3: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_operator_composition_results.json','w'),indent=1)
    print(f'wrote bilin18_operator_composition_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
