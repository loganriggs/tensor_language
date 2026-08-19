"""Registered test of section 220's unregistered observation: over the
private span 6:1-8, early readers (2,3,5) seemed to share a code while deep
readers (9,13,17) went private. Group-structured transfer at matched rank 18:
within-group LORO (reconstruct each member's held-out 20 forms from a basis
fit on its two groupmates' forms) and cross-group transfer, for the private
writer L6 and the shared control writer L9. Also reports the measured
random-basis baseline instead of a fixed bar (the section-220 lesson: rank 18
in a 36-dim space has a substantial floor by construction).

REGISTERED PREDICTIONS: (a) early-group internal transfer over the private
span: median >= 0.45 (the early readers genuinely share a code for the
whisper); (b) deep-group internal transfer <= 0.25 (no "deep dialect" --
each deep reader is idiosyncratic even among its peers); (c) control writer
L9: BOTH groups' internal transfer >= 0.45 (grouping is span-specific, not a
depth artifact of the instrument); (d) all reported gaps exceed the measured
random baseline by >= 0.15 wherever a claim is made."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152; NF=40; KS=8; RB=18
EARLY=(2,3,5); DEEP=(9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_dialect_groups_results.json')

@torch.no_grad()
def build(Wl):
    readers=tuple(r for r in EARLY+DEEP if r!=Wl)
    Yw=grab(Wl,0,300); mu=Yw.mean(0)
    _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
    V=orth(Vh[:KS].T)
    yproj=((grab(Wl,384,448)-mu).float()@V)[:20000]
    fams={}
    for j in readers:
        Yj=grab(j,0,60)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        fams[j]=torch.stack([(0.5*(M+M.T)).flatten() for M in
                 (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                  for f in range(NF))])
    return fams, yproj

def r2med(B, test, yproj):
    r2s=[]
    for v in test:
        Mm=v.view(KS,KS)
        c_true=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
        vt=c_true.var().clamp_min(1e-12)
        Mre=((B@v)@B).view(KS,KS)
        c_hat=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
        r2s.append(1-float(((c_hat-c_true)**2).mean()/vt))
    return sorted(r2s)[len(r2s)//2]

@torch.no_grad()
def group_stats(Wl):
    fams,yproj=build(Wl)
    g=torch.Generator(device=DEV).manual_seed(0)
    Brnd=torch.randn(RB,KS*KS,device=DEV,generator=g)
    Brnd=Brnd/Brnd.norm(dim=1,keepdim=True)
    res={'rnd':[]}
    for tag,grp in (('early',EARLY),('deep',DEEP)):
        grp=tuple(j for j in grp if j!=Wl)
        meds=[]
        for j in grp:
            mates=[fams[j2] for j2 in grp if j2!=j]
            X=torch.cat(mates)
            _,_,W=torch.linalg.svd(X, full_matrices=False)
            test=fams[j][1::2]
            meds.append(r2med(W[:RB],test,yproj))
            res['rnd'].append(r2med(Brnd,test,yproj))
        res[tag]=sorted(meds)[len(meds)//2]
        print(f'W L{Wl} {tag:5s} internal: members '
              f'{[f"{x:+.2f}" for x in meds]} median {res[tag]:+.3f}',
              flush=True)
    xg=[]
    for j in (jj for jj in DEEP if jj!=Wl):
        X=torch.cat([fams[j2] for j2 in EARLY if j2!=Wl])
        _,_,W=torch.linalg.svd(X, full_matrices=False)
        xg.append(r2med(W[:RB],fams[j][1::2],yproj))
    res['early_to_deep']=sorted(xg)[len(xg)//2]
    res['rnd_med']=sorted(res['rnd'])[len(res['rnd'])//2]
    print(f'W L{Wl} early->deep: {res["early_to_deep"]:+.3f} | '
          f'random baseline {res["rnd_med"]:+.3f}',flush=True)
    return res

@torch.no_grad()
def main():
    t0=time.time()
    r6=group_stats(6); r9=group_stats(9)
    pa=r6['early']>=0.45
    pb=r6['deep']<=0.25
    pc=r9['early']>=0.45 and r9['deep']>=0.45
    out={'private_L6':{k:v for k,v in r6.items() if k!='rnd'},
         'control_L9':{k:v for k,v in r9.items() if k!='rnd'},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) early share the whisper >=0.45: {'HELD' if pa else 'FAILED'}")
    print(f"(b) no deep dialect <=0.25: {'HELD' if pb else 'FAILED'}")
    print(f"(c) L9 control both >=0.45: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
