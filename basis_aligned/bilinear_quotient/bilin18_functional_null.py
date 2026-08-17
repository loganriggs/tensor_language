"""The missing null: which parts of the functional structure are trained, and which
are generic?

User question (2026-08-17): why would SGD produce dense-support orthogonal
functionals -- is it optimal, or just typical? The decisive control the arc skipped:
the SAME battery on a randomly initialised bilin18. REGISTERED PREDICTIONS:
  GENERIC (reproduce on random weights):
    (a) signed within/cross-reader cosines <= 0.2 (orthogonality is typicality);
    (b) envelope eff-rank <= 6 (the magnitude template comes from L/R spectra, not
        training);
  TRAINED (do NOT reproduce):
    (c) family eff-rank on random weights >= 120 of 240 (training compressed to 80);
    (d) leave-one-reader-out R^2 on random weights <= 0.45 (the shared vocabulary is
        learned; trained value 0.71).
If (a)-(d) all hold, the answer to 'why did SGD find this' splits cleanly: the
orthogonality and density are the architecture's default; what SGD actually built is
the SHARED 80-dim vocabulary."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language')
import torch.nn.functional as F
import jacclust.tt_model as TT
from bilin18_joint_removal import FW, DEV
from tier2_model import load_elriggs
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_functional_null_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    _,cfg=load_elriggs('bilin18', device='cpu')  # config only; free the trained copy
    torch.manual_seed(0)
    rnd=TT.GPT(TT.GPTConfig(**{k:v for k,v in cfg.items()})).to(DEV).eval()
    for p in rnd.parameters(): p.requires_grad_(False)
    # collect L1 output basis and reader bases ON THE RANDOM MODEL
    def collect(li):
        outs=[]
        h=rnd.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,60,6):
            b=FW[i:i+6,:513].to(DEV)
            rnd(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    Y1=collect(1); Y1c=Y1-Y1.mean(0)
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    Q,_=torch.linalg.qr(Vh[:K].T); V=Q[:,:K]
    fams={}
    rows=[]
    for j in READERS:
        Yj=collect(j)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        Pq,_=torch.linalg.qr(Vhj[:NF].T); P=Pq[:,:NF]
        mlp=rnd.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        mats=[]
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            M=0.5*(M+M.T); mats.append(M)
            rows.append(M.flatten())
        fams[j]=mats
    X=torch.stack(rows)
    # (a) signed cosines
    flat5=[(M/M.norm().clamp_min(1e-12)).flatten() for M in fams[5][:32]]
    wc=[abs(float(flat5[a]@flat5[b])) for a,b in
        itertools.combinations(range(32),2)]
    within=sum(wc)/len(wc)
    xc=[]
    keys=list(fams)
    for a,b in itertools.combinations(keys,2):
        fa=[(M/M.norm().clamp_min(1e-12)).flatten() for M in fams[a][:8]]
        fb=[(M/M.norm().clamp_min(1e-12)).flatten() for M in fams[b][:8]]
        for u in fa:
            for w in fb: xc.append(abs(float(u@w)))
    cross=sum(xc)/len(xc)
    # (b) envelope eff-rank
    def effrank(A):
        sv=torch.linalg.svdvals(A); e=sv**2
        return float(e.sum()**2/(e**2).sum())
    er_env=effrank(X.abs())
    # (c) family eff-rank
    er_fam=effrank(X)
    # (d) LORO at r=80
    import math
    r2s=[]
    yproj=(Y1c@V)[:20000]
    for jout in READERS:
        train=[M.flatten() for j2,Ms in fams.items() if j2!=jout for M in Ms]
        Xt=torch.stack(train)
        _,_,Wb=torch.linalg.svd(Xt, full_matrices=False)
        B=Wb[:80]
        for M in fams[jout][:12]:
            c=torch.einsum('na,ab,nb->n',yproj,M,yproj)
            co=B@M.flatten(); Mre=(co@B).view(K,K)
            ch=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
            r2s.append(1-float(((ch-c)**2).mean()/c.var().clamp_min(1e-12)))
    loro=sorted(r2s)[len(r2s)//2]
    out={'within_cos':within,'cross_cos':cross,'envelope_effrank':er_env,
         'family_effrank':er_fam,'loro_r2':loro}
    pa=within<=0.2 and cross<=0.2
    pb=er_env<=6
    pc=er_fam>=120
    pd=loro<=0.45
    out['pred_a_generic_orthogonality']=bool(pa)
    out['pred_b_generic_envelope']=bool(pb)
    out['pred_c_trained_compression']=bool(pc)
    out['pred_d_trained_vocab']=bool(pd)
    print(f'RANDOM-INIT bilin18 (trained values in parens):')
    print(f'  signed cos within {within:.2f} / cross {cross:.2f}   (0.11 / 0.089)')
    print(f'  envelope eff-rank {er_env:.1f}                        (2.6)')
    print(f'  family eff-rank {er_fam:.0f} of 240                   (80)')
    print(f'  LORO R^2 at r=80: {loro:.2f}                         (0.71)')
    print(f"\n(a) orthogonality generic: {'HELD' if pa else 'FAILED'} | "
          f"(b) envelope generic: {'HELD' if pb else 'FAILED'}")
    print(f"(c) compression trained: {'HELD' if pc else 'FAILED'} | "
          f"(d) vocabulary trained: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
