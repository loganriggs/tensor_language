"""DISSECT THE ATTN0 ANOMALY (section 253: per-token value tables for
attn0 are catastrophic at -223%). attn0's values serve two consumers: its
OWN pattern-mix (the attn0 output written at layer 0) and the v1 BROADCAST
that every later layer lambda-mixes into its values. Split them with a
manual traced forward: arm OWN = attn0's own mix uses tabled values, later
layers get the true v1; arm BCAST = attn0 uses true values, later layers
get tabled v1; arm BOTH = section 253's condition (sanity: should
reproduce the catastrophe).

REGISTERED PREDICTIONS: (a) the damage localizes to the BROADCAST arm
(bcast >= 3x own, both approx bcast) -- the v1 broadcast is the
non-lexical object; (b) sanity: BOTH within 20% of the section-253
value-table damage (+0.428 vs ablate +0.132); (c) additivity reported:
own + bcast vs both (interaction term)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import NH, HD
from tier2_model import rope_tables, apply_rot
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'attn0_broadcast_results.json'
CA,CB=300,512; R0,R1=120,300

@torch.no_grad()
def main():
    t0=time.time()
    # fit attn0 per-token value table on window A via c_v hook
    sums=torch.zeros(V,D,device=DEV); cnt=torch.zeros(V,device=DEV)
    caps={}
    h=m.transformer.h[0].attn.c_v.register_forward_hook(
        lambda mo_,i_,o_: caps.__setitem__(0,o_.detach()
                                           .reshape(-1,D).float()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous()
        m(idx, bb[:,1:].contiguous())
        ids=idx.reshape(-1)
        cnt.index_add_(0,ids,torch.ones_like(ids,dtype=torch.float))
        sums.index_add_(0,ids,caps[0])
    h.remove()
    tab=sums/cnt.clamp_min(1)[:,None]
    tab[cnt==0]=sums.sum(0)/cnt.sum()
    tab=tab.to(torch.float16)
    def pertok(mode):
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B,T=idx.shape
            cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
            cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
            mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for li in range(18):
                blk=m.transformer.h[li]; a=blk.attn
                x=blk.lambdas[0]*x+blk.lambdas[1]*x0
                hcur=F.rms_norm(x,(D,))
                def qk(l):
                    z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
                    return apply_rot(z,cosb,sinb)
                v_true=a.c_v(hcur).view(B,T,NH,HD)
                if li==0:
                    v_tab=tab[idx].to(v_true.dtype).view(B,T,NH,HD)
                    own = v_tab if mode in ('own','both') else v_true
                    v1  = v_tab if mode in ('bcast','both') else v_true
                    v=own
                else:
                    v=(1-a.lamb)*v_true+a.lamb*v1.view_as(v_true)
                q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
                s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
                s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
                pat=(s1*s2).masked_fill(~mask,0.0)
                att=a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v)
                             .reshape(B,T,-1))
                x=x+att
                xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
                x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        return torch.cat(ces)
    base=pertok('clean')
    own=float((pertok('own')-base).mean())
    bc=float((pertok('bcast')-base).mean())
    both=float((pertok('both')-base).mean())
    pa=bc>=3*max(own,1e-4) and abs(both-bc)<=0.5*abs(bc)
    pb=abs(both-0.428)<=0.2*0.428
    inter=both-(own+bc)
    out={'own':round(own,4),'bcast':round(bc,4),'both':round(both,4),
         'interaction':round(inter,4),
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'own {own:+.4f} | bcast {bc:+.4f} | both {both:+.4f} | '
          f'interaction {inter:+.4f}')
    print(f"(a) broadcast is the anomaly: {'HELD' if pa else 'FAILED'}")
    print(f"(b) sanity vs 253: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
