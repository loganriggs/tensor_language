"""The linear sector: which coordinates do OV (value) paths read, and do they share
the quadratic vocabulary's coordinates?

The degree-1 sector was excluded from the functional vocabulary by construction
(section 58); section 84 showed QK quadratic readers use a DISJOINT code from MLP
quadratic readers. Remaining registered-open item: the OV census. Per head in
layers 2-4, the OV transmit map T_h = Wproj_h @ Wv_h (rank <= 128) is linear;
we measure (1) its read energy on L1's top-48 output-PCA span V, versus the
uniform null 48/1152 = 0.042 and a random-span control, and (2) whether its
within-span coordinate weighting matches the quadratic vocabulary's usage
(column-energy of the 240 unit-normalized coupling matrices).

REGISTERED PREDICTIONS: (a) median OV read energy on the span >= 3x the 0.042
null (value routes carry L1's principal content -- value transplants had effects);
(b) skeptical, given the QK result: Spearman(mean OV coordinate weighting,
quadratic usage) < 0.3 -- the linear sector is a THIRD independent code. The
alternative (>= 0.5) would mean linear and quadratic readers target shared
coordinates and would nuance section 84's disjointness story."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
NH, HD, D, K, NF = 9, 128, 1152, 48, 40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_ov_census_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
    return float((ra*rb).mean())

@torch.no_grad()
def main():
    t0=time.time()
    def collect(li):
        outs=[]
        h=m.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,60,6):
            b=FW[i:i+6,:513].to(DEV)
            m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    Y1=collect(1); Y1c=Y1-Y1.mean(0)
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    # quadratic-vocabulary usage: column energy of the 240 coupling matrices
    usage=torch.zeros(K,device=DEV)
    for j in READERS:
        Yj=collect(j)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            Ms=0.5*(M+M.T); Ms=Ms/Ms.norm().clamp_min(1e-12)
            usage+= (Ms**2).sum(0)
    g=torch.Generator(device=DEV).manual_seed(0)
    es,ws=[],[]
    e_rand=[]
    for li in (2,3,4):
        a=m.transformer.h[li].attn
        Wv=a.c_v.weight.detach().float().view(NH,HD,D)
        Wp=a.c_proj.weight.detach().float().view(D,NH,HD)
        for h in range(NH):
            T_h=Wp[:,h,:]@Wv[h]           # D x D transmit map
            tot=float((T_h**2).sum())
            span=T_h@V                     # D x K
            es.append(float((span**2).sum())/max(tot,1e-12))
            ws.append((span**2).sum(0))    # K-vector read weighting
            Q=orth(torch.randn(D,K,device=DEV,generator=g))
            e_rand.append(float(((T_h@Q)**2).sum())/max(tot,1e-12))
    med=sorted(es)[len(es)//2]; medr=sorted(e_rand)[len(e_rand)//2]
    wmean=torch.stack(ws).mean(0)
    rho=spearman(wmean.cpu(),usage.cpu())
    out={'median_span_energy':med,'median_random_energy':medr,
         'null_uniform':K/D,'spearman_ov_vs_usage':rho,'per_head_energy':es}
    pa=med>=3*(K/D); pb=rho<0.3
    out['pred_a']=bool(pa); out['pred_b_third_code']=bool(pb)
    print(f'OV read energy on L1 top-48 span: median {med:.3f} '
          f'(uniform null {K/D:.3f}, random-span {medr:.3f})')
    print(f'Spearman(OV weighting, quadratic usage): {rho:+.2f}')
    print(f"(a) OV reads the span >=3x null: {'HELD' if pa else 'FAILED'}")
    print(f"(b) third code (rho<0.3): {'HELD' if pb else 'FAILED'}"
          + ('  [>=0.5: shared coordinates -- nuances 84]' if rho>=0.5 else ''))
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
