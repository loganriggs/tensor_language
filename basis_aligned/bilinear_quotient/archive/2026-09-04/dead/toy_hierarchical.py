"""TOY HIERARCHICAL / DAG (user: extend toys with structure types --
hierarchical DAG -- for insight + hyperparameter tuning). Plant a 2-level
hierarchy: N_COARSE parent atoms, each with N_CHILD children. A datapoint
activates 1 parent + 2 of ITS OWN children -- so a child NEVER appears
without its parent (a DAG dependency). Questions:
  (1) does the top-k SAE recover the atoms AND the DAG (in the codes, a
      child atom active => its parent active)?
  (2) HYPERPARAMETER: how does recovery depend on k (must k >= the true
      per-datapoint atom count 3)? Sweep k -- the ELBOW tells us how to set k
      on the REAL model where the true sparsity is unknown.

REGISTERED PREDICTIONS:
  (0) SANITY: in the planted data, P(parent active | child active) = 1.0;
  (a) DAG RECOVERED at sufficient k: at k >= 3 (true per-datapoint count) the
      SAE recovers atoms (>=0.85) AND the codes preserve the DAG -- mean
      P(recovered-parent active | recovered-child active) >= 0.85;
  (b) HYPERPARAM: report atom-recovery + DAG-recovery + reconstruction R^2
      across k in {2,3,4,6,8}; recovery should be POOR at k<3 (under-budget)
      and SATURATE at k>=3 -- an interpretable elbow;
  NULL: at k=2 (< true 3) recovery/DAG is substantially worse than at k>=3."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'toy_hierarchical_results.json'
Dd = 128; N_COARSE = 6; N_CHILD = 6; N = 24000; P_sae = 64; KS = [2, 3, 4, 6, 8]; STEPS = 1500


def topk_encode(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def train_sae(O, k, P, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    We = (torch.randn(Dd, P, device=DEV)/np.sqrt(Dd)).requires_grad_(True)
    Wd = (torch.randn(P, Dd, device=DEV)/np.sqrt(P)).requires_grad_(True)
    b = O.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([We, Wd, b], lr=3e-3)
    for s in range(steps):
        z = topk_encode((O-b)@We, k); loss = F.mse_loss(z@Wd+b, O)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = topk_encode((O-b)@We, k); recon = z@Wd + b
        r2 = float(1 - ((O-recon)**2).sum()/((O-O.mean(0))**2).sum())
        Wdn = Wd/Wd.norm(dim=1, keepdim=True)
        active = (z.abs() > 1e-6)          # (N, P) which learned atoms fire per datapoint
    return Wdn.detach(), active.cpu().numpy()


def main():
    t0 = time.time(); torch.manual_seed(0)
    P_true = N_COARSE + N_COARSE*N_CHILD
    Dtrue = torch.randn(P_true, Dd, device=DEV); Dtrue = Dtrue/Dtrue.norm(dim=1, keepdim=True)
    parent_of = {}                          # child global idx -> parent global idx
    for c in range(N_COARSE):
        for j in range(N_CHILD):
            child = N_COARSE + c*N_CHILD + j; parent_of[child] = c
    Z = torch.zeros(N, P_true, device=DEV); active_true = np.zeros((N, P_true), bool)
    for i in range(N):
        c = int(torch.randint(N_COARSE, (1,)))
        kids = N_COARSE + c*N_CHILD + torch.randperm(N_CHILD, device=DEV)[:2]
        idx = torch.cat([torch.tensor([c], device=DEV), kids])
        Z[i, idx] = torch.randn(3, device=DEV).abs()+0.3
        for j in idx.tolist(): active_true[i, j] = True
    O = Z @ Dtrue
    # sanity: P(parent | child) in planted data
    childs = list(parent_of); pc = np.mean([active_true[active_true[:, ch], parent_of[ch]].mean() for ch in childs])
    print(f'planted P(parent|child) = {pc:.3f} (should be 1.0)', flush=True)

    Dt = Dtrue/Dtrue.norm(dim=1, keepdim=True)
    res = {}
    for k in KS:
        Dl, active = train_sae(O, k, P_sae)
        cos = (Dl @ Dt.T).abs().cpu().numpy()      # (P_sae, P_true)
        recovery = float(cos.max(0).mean())
        match = cos.argmax(1); match_cos = cos.max(1)      # each learned atom -> nearest true
        # DAG recovery: for learned atoms matching a CHILD, does the learned atom matching its PARENT co-fire?
        true2learned = {}                                   # true atom -> best learned atom
        for tl in range(P_true):
            cand = np.where(match == tl)[0]
            if len(cand): true2learned[tl] = cand[match_cos[cand].argmax()]
        pcs = []
        for ch in childs:
            if ch in true2learned and parent_of[ch] in true2learned:
                lc = true2learned[ch]; lp = true2learned[parent_of[ch]]
                mask = active[:, lc]
                if mask.sum() > 5: pcs.append(active[mask, lp].mean())
        dag = float(np.mean(pcs)) if pcs else 0.0
        res[k] = {'atom_recovery': round(recovery,3), 'dag_recovery': round(dag,3),
                  'n_child_pairs': len(pcs)}
        print(f'k={k}: atom-recovery {recovery:.3f}  DAG-recovery P(parent|child) {dag:.3f}  '
              f'({len(pcs)} child pairs)', flush=True)

    good = res.get(3) or res.get(4)
    pa = good and good['atom_recovery'] >= 0.85 and good['dag_recovery'] >= 0.85
    null_ok = res[2]['atom_recovery'] < (res[3]['atom_recovery'] - 0.05) if 3 in res else True
    print(f'\n(a) DAG recovered at k>=3: {pa}; NULL k=2 worse than k=3: {null_ok}', flush=True)
    out = {'Dd': Dd, 'N_coarse': N_COARSE, 'N_child': N_CHILD, 'P_sae': P_sae, 'planted_p_parent_child': round(pc,3),
           'by_k': res, 'pred_a': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
