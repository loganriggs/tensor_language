"""Are the vocabulary's words causally individuated?

§63-64: the principal functionals are not the verified axes and containment was
ill-posed. The direct causal question: do different vocabulary words have different
CONSTITUENCIES -- steering L1's output along word k's top eigen-direction should move
a distinct set of reader coefficients. For principal functionals #1, #2, #3: steer
along each word's top output-direction (same magnitude), record the movement profile
over 12 sampled reader coefficients (2 forms x readers L2/L3/L5/L9/L13/L17).
REGISTERED PREDICTIONS:
  (a) profiles are distinct: mean pairwise correlation of the three movement profiles
      <= 0.5;
  (b) each word moves SOMETHING: every profile's max coefficient movement >= 0.3
      sigma (the words are causally live, not dead directions);
Control: a random output direction's profile correlates <= 0.3 with each word's."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_gradient_steering import run_forward, collect_basis, orth
from bilin18_joint_removal import fwd, m, FW, DEV
from bilin18_identifiable import form_for_direction
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_word_constituencies_results.json')

def main():
    t0=time.time()
    Y1c=collect_basis()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    s_out=float(Y1c.norm(dim=1).mean())/K**0.5
    mag=2*s_out*K**0.5*0.2
    rows=[]
    targets=[]
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
        for f in (0,3):
            targets.append((j,form_for_direction(mlp,P[:,f].float()).float()))
    X=torch.stack(rows)
    _,sv,W=torch.linalg.svd(X, full_matrices=False)
    words=[]
    for p_ in range(3):
        Pm=0.5*(W[p_].view(K,K)+W[p_].view(K,K).T)
        evp,Up=torch.linalg.eigh(Pm.double())
        w=(V@Up[:,evp.abs().argmax()].float()); words.append(w/w.norm())
    g=torch.Generator(device=DEV).manual_seed(0)
    r=torch.randn(D,device=DEV,generator=g); words.append(r/r.norm())
    rows_in=FW[300:312,:257].to(DEV)
    with torch.no_grad():
        base,_=run_forward(rows_in,targets)
        sig={i: float(base[i].std()) for i in base}
        profiles=[]
        for w in words:
            p,_=run_forward(rows_in,targets,steer=(w,mag))
            prof=torch.tensor([abs(float((p[i]-base[i]).mean()))/sig[i]
                               for i in range(len(targets))])
            profiles.append(prof)
    out={'profiles':[[round(float(v),3) for v in pr] for pr in profiles]}
    cors=[]
    for a,b in itertools.combinations(range(3),2):
        pa_,pb_=profiles[a],profiles[b]
        c=float(((pa_-pa_.mean())@(pb_-pb_.mean()))/
                (pa_.std()*pb_.std()*len(pa_)).clamp(min=1e-9))
        cors.append(c)
    rcors=[]
    for a in range(3):
        pa_,pr_=profiles[a],profiles[3]
        rcors.append(float(((pa_-pa_.mean())@(pr_-pr_.mean()))/
                     (pa_.std()*pr_.std()*len(pa_)).clamp(min=1e-9)))
    maxmoves=[float(pr.max()) for pr in profiles[:3]]
    for i,pr in enumerate(profiles[:3]):
        print(f'word #{i+1}: max move {float(pr.max()):.2f}s | profile '
              f'{[round(float(v),2) for v in pr]}')
    meanc=sum(cors)/len(cors)
    pa=meanc<=0.5; pb=all(v>=0.3 for v in maxmoves)
    pc=all(abs(c)<=0.3 for c in rcors)
    out['mean_word_corr']=meanc; out['random_corrs']=rcors
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['ctrl']=bool(pc)
    print(f'\nmean pairwise word-profile correlation: {meanc:+.2f}')
    print(f"(a) distinct (<=0.5): {'HELD' if pa else 'FAILED'} | "
          f"(b) all live (>=0.3s): {'HELD' if pb else 'FAILED'} | "
          f"random ctrl <=0.3: {'OK' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
