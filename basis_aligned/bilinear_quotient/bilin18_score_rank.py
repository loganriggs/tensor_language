"""Score-rank census: how low-rank are the heads' effective score functions
on-distribution? Each pattern factor s1 = q.k/128 is a bilinear form of the
rms-normed stream; its on-distribution structure is the operator
K = C^{1/2} Wq^T Wk C^{1/2} (C = stream covariance at that layer). Per head and
factor (s1, s2), layers 0,2,5,9,13,16: effective rank of K (participation ratio
of squared singular values).

REGISTERED PREDICTIONS: (a) scores are low-rank on-distribution: median
eff-rank <= 16 of 128; (b) null: replacing C with isotropic identity gives
median eff-rank >= 3x higher (the compression comes from the data, not the
weights alone); (c) the two factors of a head have similar ranks (median
|r1-r2|/max <= 0.5) -- product attention composes two comparable filters."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
NH,HD,D=9,128,1152
LAYERS=(0,2,5,9,13,16)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_score_rank_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    # collect rms-normed stream entering each layer's attention (hcur)
    caps={li:[] for li in LAYERS}
    hs=[]
    for li in LAYERS:
        def mk(li=li):
            return lambda mod,inp: caps[li].append(
                F.rms_norm(inp[0].detach().reshape(-1,D).float(),(D,))) or None
        hs.append(m.transformer.h[li].attn.register_forward_pre_hook(mk()))
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    def effrank(Kop):
        sv=torch.linalg.svdvals(Kop); e=sv**2
        return float(e.sum()**2/(e**2).sum())
    out={}; ranks=[]; ratios=[]; iso_ranks=[]
    for li in LAYERS:
        X=torch.cat(caps[li]); Xc=X-X.mean(0)
        C=Xc.T@Xc/Xc.shape[0]
        ev,U=torch.linalg.eigh(C.double())
        Ch=((U*ev.clamp_min(0).sqrt())@U.T).float()
        a=m.transformer.h[li].attn
        row=[]
        for h in range(NH):
            for tag,(wq,wk) in (('s1',(a.c_q,a.c_k)),('s2',(a.c_q2,a.c_k2))):
                Wq=wq.weight.detach().float().view(NH,HD,D)[h]
                Wk=wk.weight.detach().float().view(NH,HD,D)[h]
                K=Ch@Wq.T@Wk@Ch
                r=effrank(K)
                Ki=Wq.T@Wk
                ri=effrank(Ki)
                row.append({'head':h,'factor':tag,'rank':r,'iso':ri})
                ranks.append(r); iso_ranks.append(ri)
            r1=row[-2]['rank']; r2=row[-1]['rank']
            ratios.append(abs(r1-r2)/max(r1,r2))
        out[str(li)]=row
        med=sorted(x['rank'] for x in row)[len(row)//2]
        medi=sorted(x['iso'] for x in row)[len(row)//2]
        print(f'L{li:2d}: median eff-rank {med:.1f} (isotropic {medi:.1f})',flush=True)
    mr=sorted(ranks)[len(ranks)//2]; mi=sorted(iso_ranks)[len(iso_ranks)//2]
    mrt=sorted(ratios)[len(ratios)//2]
    pa=mr<=16; pb=mi>=3*mr; pc=mrt<=0.5
    out['median_rank']=mr; out['median_iso']=mi; out['median_factor_gap']=mrt
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\nmedian eff-rank {mr:.1f} | isotropic {mi:.1f} | factor gap {mrt:.2f}")
    print(f"(a) low-rank on-distribution (<=16): {'HELD' if pa else 'FAILED'}")
    print(f"(b) data supplies the compression (iso >=3x): {'HELD' if pb else 'FAILED'}")
    print(f"(c) factors comparable (<=0.5): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
