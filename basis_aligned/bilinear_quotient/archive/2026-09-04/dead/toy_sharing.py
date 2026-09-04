"""TOY SHARING (user's core point: if datapoints SHARE computation respect
that, if they DON'T respect that -- don't over-explain). Plant a dictionary
with COMMON atoms (shared by many datapoints = a common circuit) and RARE
atoms (datapoint-specific). Each datapoint = 2 common + 2 rare atoms. Check
whether the top-k SAE recovers the atoms AND their USAGE-FREQUENCY structure:
common atoms should be recovered with HIGH usage, rare with LOW usage. If the
recovered-atom usage tracks the planted usage, the method RESPECTS the
sharing structure (shared computation shared, unique kept unique).

REGISTERED PREDICTIONS:
  (0) SANITY: planted common atoms have higher true usage than rare;
  (a) RESPECTS SHARING: the SAE recovers the atoms (recovery >= 0.85) AND
      the correlation between a recovered atom's usage frequency and its
      matched true atom's usage frequency is HIGH (>= 0.7) -- common circuits
      recovered as high-usage, rare as low-usage;
  (b) report mean usage of recovered-common vs recovered-rare atoms + the
      usage correlation;
  NULL: the usage correlation with SHUFFLED true-usage is ~0 (the match is
      real, not chance)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'toy_sharing_results.json'
Dd = 128; N = 20000; N_COMMON = 8; N_RARE = 56; P_sae = 96; K = 4; STEPS = 1500


def topk_encode(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def train_sae(O, k, P, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    We = (torch.randn(Dd, P, device=DEV)/np.sqrt(Dd)).requires_grad_(True)
    Wd = (torch.randn(P, Dd, device=DEV)/np.sqrt(P)).requires_grad_(True)
    b = O.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([We, Wd, b], lr=3e-3)
    for s in range(steps):
        z = topk_encode((O-b)@We, k); loss = F.mse_loss(z@Wd + b, O)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = topk_encode((O-b)@We, k); usage = (z.abs()>1e-6).float().mean(0)   # (P,) fraction of datapoints using each learned atom
        Wdn = Wd / Wd.norm(dim=1, keepdim=True)
    return Wdn.detach(), usage.cpu().numpy()


def main():
    t0 = time.time(); torch.manual_seed(0)
    P_true = N_COMMON + N_RARE
    Dtrue = torch.randn(P_true, Dd, device=DEV); Dtrue = Dtrue/Dtrue.norm(dim=1, keepdim=True)
    Z = torch.zeros(N, P_true, device=DEV); true_usage = np.zeros(P_true)
    for i in range(N):
        c = torch.randperm(N_COMMON, device=DEV)[:2]                 # 2 of 8 common
        r = N_COMMON + torch.randperm(N_RARE, device=DEV)[:2]        # 2 of 56 rare
        idx = torch.cat([c, r]); Z[i, idx] = torch.randn(4, device=DEV).abs()+0.3
        for j in idx.tolist(): true_usage[j] += 1
    true_usage /= N
    O = Z @ Dtrue
    print(f'planted: {N_COMMON} common (true usage ~{true_usage[:N_COMMON].mean():.3f}) + '
          f'{N_RARE} rare (~{true_usage[N_COMMON:].mean():.3f})', flush=True)

    Dsae, learned_usage = train_sae(O, K, P_sae)
    # match each LEARNED atom to its closest TRUE atom
    Dt = Dtrue/Dtrue.norm(dim=1, keepdim=True); Dl = Dsae
    cos = (Dl @ Dt.T).abs().cpu().numpy()                 # (P_sae, P_true)
    match = cos.argmax(1); match_cos = cos.max(1)
    recovery = float((cos.max(0)).mean())                 # mean over true atoms of best match
    # usage correlation: learned atom usage vs its matched true atom usage
    good = match_cos > 0.6
    lu = learned_usage[good]; tu = true_usage[match[good]]
    corr = float(np.corrcoef(lu, tu)[0,1]) if good.sum() > 2 else 0.0
    # common vs rare recovered usage
    is_common = match[good] < N_COMMON
    uc = lu[is_common].mean() if is_common.any() else 0.0
    ur = lu[~is_common].mean() if (~is_common).any() else 0.0
    rng = np.random.default_rng(0); tu_sh = tu.copy(); rng.shuffle(tu_sh)
    null_corr = float(np.corrcoef(lu, tu_sh)[0,1]) if good.sum() > 2 else 0.0

    print(f'atom-recovery {recovery:.3f}  usage-corr(learned,true) {corr:.3f}  (shuffled null {null_corr:.3f})', flush=True)
    print(f'recovered-COMMON mean usage {uc:.3f}  recovered-RARE mean usage {ur:.3f}', flush=True)
    pa = recovery >= 0.85 and corr >= 0.7 and uc > ur
    null_ok = abs(null_corr) < 0.3
    print(f'\n(a) recovers atoms + respects sharing (usage-corr>=0.7, common>rare): {pa}', flush=True)
    print(f'NULL shuffled usage-corr ~0: {null_ok}', flush=True)
    out = {'Dd': Dd, 'N_common': N_COMMON, 'N_rare': N_RARE, 'P_sae': P_sae, 'K': K,
           'atom_recovery': round(recovery,3), 'usage_corr': round(corr,3), 'null_corr': round(null_corr,3),
           'recovered_common_usage': round(float(uc),3), 'recovered_rare_usage': round(float(ur),3),
           'pred_a': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
