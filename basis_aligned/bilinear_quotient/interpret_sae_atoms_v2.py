"""INTERPRET SAE ATOMS v2 (fix 751's confound). Monosemanticity via
FREQUENCY-NORMALIZED selectivity (LIFT), not raw concentration: for each
atom, LIFT = max over token-types of P(type | top-activating datapoints) /
P(type | overall). Lift removes the frequency confound (common tokens no
longer inflate the score). Also: sweep P (more atoms -> more specific?) and
STRATIFY by usage (are LOW-usage atoms more monosemantic than the most-used
ones, which are polysemantic by construction?). Compare SAE vs SVD lift.

REGISTERED PREDICTIONS:
  (0) SANITY: lift >= 1 (top-activating enriched for some token);
  (a) SAE > SVD on lift, and LOW-usage atoms > HIGH-usage: freq-normalized,
      the SAE atoms are MORE selective than SVD directions (mean lift higher
      by >= 1.5x), and low-usage (specific) atoms have higher lift than the
      most-used ones; larger P raises lift;
  (b) report mean lift SAE(P) vs SVD, split by usage tercile;
  NULL: random directions have lift ~ 1-2 (near base rate)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
from collections import Counter

D = 1152; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'interpret_sae_atoms_v2_results.json'
NFIT = 96; K = 32; PS = [512, 1024]; STEPS = 900; TOPN = 150


@torch.no_grad()
def capture(rows, n):
    cap=[]; toks=[]
    h=m.transformer.h[LAYER].mlp.register_forward_hook(lambda mo,i_,o_: cap.append(o_.detach().float().reshape(-1,D)))
    for i in range(0,n,4):
        bb=rows[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous(); toks.append(idx.reshape(-1).cpu())
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for li,blk in enumerate(m.transformer.h): x,v1=blk(x,v1,x0)
    h.remove(); return torch.cat(cap,0), torch.cat(toks).numpy()


def topk(pre,k):
    val,idx=pre.topk(k,dim=1); z=torch.zeros_like(pre); z.scatter_(1,idx,F.relu(val)); return z


def train_sae(O,k,P,steps=STEPS,seed=0):
    torch.manual_seed(seed)
    We=(torch.randn(D,P,device=DEV)/np.sqrt(D)).requires_grad_(True)
    Wd=(torch.randn(P,D,device=DEV)/np.sqrt(P)).requires_grad_(True)
    b=O.mean(0).clone().requires_grad_(True); opt=torch.optim.Adam([We,Wd,b],lr=2e-3)
    for s in range(steps):
        z=topk((O-b)@We,k); loss=F.mse_loss(z@Wd+b,O); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): z=topk((O-b)@We,k)
    return z.cpu().numpy()


def lift(code_col, toks, base, topn=TOPN):
    idx=np.argsort(-code_col)[:topn]
    if code_col[idx[-1]]<=0: idx=idx[code_col[idx]>0]
    if len(idx)<10: return 1.0
    c=Counter(toks[idx].tolist()); tot=len(idx)
    return max((cnt/tot)/(base.get(t,1e-9)) for t,cnt in c.items())


@torch.no_grad()
def main():
    t0=time.time(); cl.use_state(PT+'census_state_diverse.pt')
    rows=cl.fineweb_rows(NFIT); O,toks=capture(rows,NFIT)
    base=Counter(toks.tolist()); N=len(toks); base={t:c/N for t,c in base.items()}

    res={}
    for P in PS:
        with torch.enable_grad(): Z=train_sae(O,K,P)
        usage=(Z>1e-6).mean(0); lifts=np.array([lift(Z[:,a],toks,base) for a in range(P)])
        order=np.argsort(-usage)
        hi=lifts[order[:P//3]].mean(); mid=lifts[order[P//3:2*P//3]].mean(); lo=lifts[order[2*P//3:]].mean()
        res[str(P)]={'mean_lift':round(float(lifts.mean()),2),'lift_high_usage':round(float(hi),2),
                     'lift_mid':round(float(mid),2),'lift_low_usage':round(float(lo),2),
                     'max_lift':round(float(lifts.max()),1)}
        print(f'P={P}: mean-lift {lifts.mean():.2f}  high-usage {hi:.2f}  low-usage {lo:.2f}  max {lifts.max():.0f}', flush=True)
    # SVD + random baselines
    U=torch.linalg.svd(O-O.mean(0),full_matrices=False)[2][:256]
    sc=((O-O.mean(0))@U.T).cpu().numpy(); svd_lift=np.mean([lift(np.abs(sc[:,j]),toks,base) for j in range(256)])
    g=torch.Generator().manual_seed(0); Rr=torch.linalg.qr(torch.randn(D,256,generator=g))[0].to(DEV)
    rc=((O-O.mean(0))@Rr).cpu().numpy(); rand_lift=np.mean([lift(np.abs(rc[:,j]),toks,base) for j in range(256)])
    print(f'\nSVD mean-lift {svd_lift:.2f}  random {rand_lift:.2f}', flush=True)

    best_sae=max(res[str(P)]['mean_lift'] for P in PS)
    pa = best_sae > 1.5*svd_lift and res[str(PS[-1])]['lift_low_usage'] > res[str(PS[-1])]['lift_high_usage']
    null_ok = rand_lift < 3
    print(f'(a) SAE lift > 1.5x SVD & low-usage>high-usage: {pa}; NULL random low: {null_ok}', flush=True)
    out={'K':K,'Ps':PS,'sae':res,'svd_lift':round(float(svd_lift),2),'rand_lift':round(float(rand_lift),2),
         'pred_a':bool(pa),'null_ok':bool(null_ok),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1); print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__=='__main__':
    main()
