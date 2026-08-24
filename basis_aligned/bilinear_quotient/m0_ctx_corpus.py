"""m0_ctx_corpus: is writeup 480's +0.004 bigram-fold cost CORPUS-DEPENDENT? (§1175 check)
tabulated_stack2 measured the same k=2 construction at +0.092 on the DIVERSE FineWeb rows —
23x writeup 480's +0.004, which was measured on the old curated (prose-heavy) rows (base CE
3.95 vs 3.36). §1112: mlp0 tables are register-CONTEXTUAL. This A/B runs the ORIGINAL 480
harness verbatim on both corpora, k in {2, full}.
REGISTERED PREDICTIONS:
  (a) curated k2 reproduces 480: dCE in [-0.01, 0.02];
  (b) fineweb k2 lands at 0.092 +/- 0.03 (corpus effect -> the front's bigram-function claim
      is PROSE-SCOPED; writeup 480 gets a register-scope annotation);
      ALTERNATIVE: fineweb k2 ~ 0.004 -> my stack2 hook is buggy -> fix the hook instead.
  (c) untruncated full fold exact on BOTH corpora (<= 0.02) — the construction itself is
      corpus-independent, only the truncation bites."""

import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'m0_ctx_corpus_results.json'
NR=16
KS=[None,2]

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
    all_out={}
    all_out['curated']=main(cl.rows()[:NR], 'curated')
    all_out['fineweb']=main(cl.fineweb_rows(NR), 'fineweb')
    cu=all_out['curated']['dce']; fw=all_out['fineweb']['dce']
    all_out['pred_a_curated_reproduces']=bool(-0.01<=cu['k2']<=0.02)
    all_out['pred_b_corpus_effect']=bool(abs(fw['k2']-0.092)<=0.03)
    all_out['alt_hook_bug']=bool(fw['k2']<=0.02)
    all_out['pred_c_full_exact_both']=bool(abs(cu['full'])<=0.02 and abs(fw['full'])<=0.02)
    json.dump(all_out,open(OUT,'w'),indent=1)
    print(f"pred_a curated {all_out['pred_a_curated_reproduces']} | pred_b corpus-effect {all_out['pred_b_corpus_effect']} | alt hook-bug {all_out['alt_hook_bug']} | pred_c full-exact {all_out['pred_c_full_exact_both']}")
    print(f"wrote {OUT}")
