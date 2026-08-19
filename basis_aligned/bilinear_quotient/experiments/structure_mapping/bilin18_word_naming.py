"""Do the vocabulary's words have token correlates? The verified-instrument attempt.

The words are causally real (§66-67) but their principal matrices are medium-rank and
unaligned with the batteries' verified axes (§63). Last naming attempt, with the
instrument that survived verification (§8.3) and the program's own prior that token
stories fail causally: each word's ACTIVATION on data is a_k(y) = y^T Pm_k y over L1's
output y -- measurable per position. Correlate per-token mean activation with
unembedding alignment of the word's top eigen-direction, permutation nulls.
REGISTERED PREDICTIONS (modest, per the three-for-three token-story record):
  (a) at most 2 of the top 5 words clear their permutation null at rho >= 0.3;
  (b) every word's activation has document-level ICC <= 0.4 (the words are token/
      context computations, not register labels)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
enc=tiktoken.get_encoding('gpt2')
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_word_naming_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().double(); rb=b.argsort().argsort().double()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    return float((ra@rb)/(ra.norm()*rb.norm()).clamp_min(1e-30))

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs); Y1c=(Y1-Y1.mean(0)).float()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    rows=[]
    for j in READERS:
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            rows.append((0.5*(M+M.T)).flatten())
    X=torch.stack(rows)
    _,sv,W=torch.linalg.svd(X, full_matrices=False)
    yV=Y1c@V                                     # (n, K) output coords
    toks=FW[0:300,:513].reshape(-1).to(DEV)
    wte=m.transformer.wte.weight.detach().float()
    uniq,cnt=toks.unique(return_counts=True)
    keep=uniq[cnt>=30]
    doc=torch.arange(yV.shape[0],device=DEV)//512
    g=torch.Generator().manual_seed(0)
    out={'words':[]}
    n_named=0
    for p_ in range(5):
        Pm=0.5*(W[p_].view(K,K)+W[p_].view(K,K).T)
        a=torch.einsum('na,ab,nb->n',yV,Pm.float(),yV)
        a2=(a-a.mean())**2
        evp,Up=torch.linalg.eigh(Pm.double())
        w=(V@Up[:,evp.abs().argmax()].float()); w=w/w.norm()
        nm=(wte@w)[keep].abs()
        exc=torch.stack([a2[toks==t].mean() for t in keep])
        rho=spearman(nm.cpu(),exc.cpu())
        null=sorted(abs(spearman(nm[torch.randperm(keep.numel(),generator=g)].cpu(),
                                 exc.cpu())) for _ in range(100))
        p95=null[95]
        dm=torch.zeros(int(doc.max())+1,device=DEV).index_add_(0,doc,a)
        ct=torch.zeros_like(dm).index_add_(0,doc,torch.ones_like(a))
        icc=float((dm[doc]/ct[doc]-a.mean()).pow(2).mean()/a.var().clamp_min(1e-9))
        named=abs(rho)>=0.3 and abs(rho)>p95
        if named: n_named+=1
        top=[enc.decode([int(t)]) for t in keep[exc.argsort(descending=True)[:6]]]
        out['words'].append({'word':p_+1,'rho':rho,'null_p95':p95,'icc':icc,
                             'named':bool(named),'fires_on':top})
        print(f'word #{p_+1}: rho {rho:+.2f} (null {p95:.2f}) | ICC {icc:.2f} | '
              f'fires on {top}',flush=True)
    pa=n_named<=2
    pb=all(wd['icc']<=0.4 for wd in out['words'])
    out['n_named']=n_named
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\n(a) <=2 of 5 token-nameable: {'HELD' if pa else 'FAILED'} ({n_named}/5)")
    print(f"(b) all ICC <= 0.4 (not register labels): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
