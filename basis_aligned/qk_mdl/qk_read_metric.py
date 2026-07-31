"""Which embedding directions do downstream blocks DIRECTLY read? (Logan's reader-aligned-basis idea)

For a bilinear block  y = Down( (Left x) * (Right x) ),  the folded tensor is
    T[o,i,j] = sum_p Down[o,p] Left[p,i] Right[p,j].
Sensitivity of the output to input direction v in slot 1 is ||T(v,.)||_F, whose Gram is
    M1 = Left^T ( (Down^T Down) o (Right Right^T) ) Left        (o = elementwise)
and symmetrically for slot 2. No 1152^3 tensor is ever formed. M = M1 + M2 is the READ METRIC:
its top eigenvectors are the residual-stream directions this block actually reads.
"""
import sys, json, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from transformers import AutoTokenizer
DEV='cuda'; tok=AutoTokenizer.from_pretrained('gpt2')
m,cfg=load_elriggs('bilin18'); D=cfg['n_embd']; V=cfg['vocab_size']
wte=m.transformer.wte.weight.detach().float().to(DEV)
X=F.rms_norm(wte,(D,)); Xc=X-X.mean(0)
_,_,Vh=torch.linalg.svd(Xc,full_matrices=False); PCA96=Vh[:96].T          # 1152 x 96

def read_metric(li):
    mlp=m.transformer.h[li].mlp
    L=mlp.Left.weight.detach().float().to(DEV); R=mlp.Right.weight.detach().float().to(DEV)
    Dw=mlp.Down.weight.detach().float().to(DEV)
    GD=Dw.T@Dw
    M1=L.T@((GD*(R@R.T))@L); M2=R.T@((GD*(L@L.T))@R)
    del GD; torch.cuda.empty_cache()
    return (M1+M2)/2

def frac_in(Msub, M):   # fraction of read-trace inside a subspace (orthonormal cols)
    return float(torch.einsum('ij,jk,ki->', Msub.T, M, Msub).trace() if False else (Msub.T@M@Msub).trace()/M.trace())

digits=[i for i in range(V) if tok.decode([i]).strip().isdigit() and len(tok.decode([i]).strip())==1]
g=torch.Generator(device='cpu').manual_seed(0)
rand_tok=torch.randperm(V, generator=g)[:len(digits)].to(DEV)

out={}
for li in [0,1,3,7,12,17]:
    M=read_metric(li)
    ev=torch.linalg.eigvalsh(M).flip(0).clamp_min(0)
    tr=ev.sum(); c=(ev/tr).cumsum(0)
    pr=float((ev.sum()**2)/(ev**2).sum())            # participation ratio
    d90=int((c<0.90).sum())+1
    Uread=torch.linalg.eigh(M)[1].flip(1)[:,:96]     # top-96 READ directions
    f_pca=frac_in(PCA96,M); f_read=frac_in(Uread,M)
    # digit-collapse: mean pairwise distance among digit tokens / among random tokens, under M vs identity
    def scat(idx, Met=None):
        E=X[idx]-X[idx].mean(0)
        return float(((E@Met)*E).sum()/len(idx)) if Met is not None else float((E*E).sum()/len(idx))
    ratio_M = scat(digits,M)/scat(rand_tok,M); ratio_I = scat(digits)/scat(rand_tok)
    out[f'block_{li}']={'participation_ratio':round(pr,1),'dims_for_90pct_read':d90,
        'read_trace_in_top96_PCA':round(f_pca,3),'read_trace_in_top96_READ':round(f_read,3),
        'digit_scatter_ratio_under_read':round(ratio_M,3),'digit_scatter_ratio_euclid':round(ratio_I,3)}
    print(li,out[f'block_{li}'],flush=True)
    del M; torch.cuda.empty_cache()
json.dump(out,open('qk_read_metric.json','w'),indent=1); print('saved')
