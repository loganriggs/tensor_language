"""m0_diverse_width: how wide does the front actually need to look on the DIVERSE corpus?
(§1175 follow-up: the k=2 truncation costs +0.092 on FineWeb vs +0.004 on prose.)
Original 480 construction, FineWeb rows only, k in {1,2,4,8,16,full}.
REGISTERED PREDICTIONS:
  (a) monotone: cost non-increasing in k;
  (b) k=8 <= 0.02 — the front stays LOCAL on all registers, just not bigram-local;
  (c) k=4 in [0.01, 0.06] (intermediate)."""


import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m0_diverse_width_results.json'
NR=16
KS=[None,16,8,4,2,1]

@torch.no_grad()
def main(CORPUS_ROWS, tag):
    t0=time.time()
    ROWS=CORPUS_ROWS
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    at0=m.transformer.h[0].attn
    mlp0=m.transformer.h[0].mlp
    L0=mlp0.Left.weight.float(); R0=mlp0.Right.weight.float()
    D0=mlp0.Down.weight.float(); B0=mlp0.Down_bias.detach().float()
    def mlp0_manual(x):
        return ((x@L0.T)*(x@R0.T))@D0.T+B0
    def run(k,active):
        tot=0.0; cnt=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            if active:
                def fh(mo,i_,o_,idx=idx,k=k):
                    E=F.rms_norm(m.transformer.wte(idx),(D,))
                    cos,sin=at0.rotary(at0.c_q(E).view(B,T,9,128))
                    def rf(w):
                        return are(F.rms_norm(
                            w(E).view(B,T,9,128),(128,)),cos,sin)
                    qf,kf=rf(at0.c_q),rf(at0.c_k)
                    q2,k2=rf(at0.c_q2),rf(at0.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    mask=torch.tril(torch.ones(T,T,device=DEV))
                    if k is not None:
                        ar=torch.arange(T,device=DEV)
                        win=((ar[:,None]-ar[None,:])<k)
                        mask=mask*win.float()
                    pat=(sc*sc2)*mask
                    v=at0.c_v(E).view(B,T,9,128).float()
                    vm=(1-at0.lamb)*v+at0.lamb*v
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,vm)
                    a0=at0.c_proj(z.transpose(1,2).contiguous()
                                  .view(B,T,-1).to(E.dtype)).float()
                    # block-0 lambda mix: the residual entering
                    # the MLP is (lam0+lam1)*E + attn_out, and
                    # lam0+lam1 = 12.19 here. Using 1.0*E (the
                    # first version) under-weighted the embedding
                    # by 12x and made the "exact" fold cost 0.55.
                    lam=m.transformer.h[0].lambdas.detach().float()
                    xin=F.rms_norm(float(lam.sum())*E.float()+a0,
                                   (D,))
                    return mlp0_manual(xin).to(o_.dtype)
                hs.append(mlp0.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            tot+=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                 reduction='none').mean().item()
            cnt+=1
            for h in hs: h.remove()
        return tot/max(cnt,1)
    base=run(None,False)
    res={}
    for k in KS:
        lbl='full' if k is None else f'k{k}'
        res[lbl]=round(run(k,True)-base,4)
        print(f'{lbl}: dCE {res[lbl]:+.4f}',flush=True)
    out={'baseline_ce':round(base,4),'dce':res}
    print(f"[{tag}] base {out['baseline_ce']} dce {res}",flush=True)
    return out


if __name__=='__main__':
    o=main(cl.fineweb_rows(NR), 'fineweb')
    r=o['dce']
    seq=[r['k1'],r['k2'],r['k4'],r['k8'],r['k16'],r['full']]
    o['pred_a_monotone']=bool(all(seq[i2]>=seq[i2+1]-1e-3 for i2 in range(len(seq)-1)))
    o['pred_b_k8_local']=bool(r['k8']<=0.02)
    o['pred_c_k4_mid']=bool(0.01<=r['k4']<=0.06)
    json.dump(o,open(OUT,'w'),indent=1)
    print(f"pred_a monotone {o['pred_a_monotone']} | pred_b k8 {o['pred_b_k8_local']} | pred_c k4 {o['pred_c_k4_mid']}")
    print(f"wrote {OUT}")
