import numpy as np, itertools, json, os

DIR = os.path.dirname(os.path.abspath(__file__))
m, d0, n_hid = 32, 16, 64
pairs = list(itertools.combinations(range(m), 2))
T = len(pairs)
pair_idx = np.array(pairs)

def make_data(rng, n):
    R = rng.random((n, m))
    idx = np.argpartition(R, 3, axis=1)[:, :3]
    F = np.zeros((n, m)); np.put_along_axis(F, idx, 1.0, axis=1)
    return F, F[:, pair_idx[:,0]] * F[:, pair_idx[:,1]]

def sigmoid(z):
    return np.where(z>=0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

summary = {}
for seed in [0,1,2]:
    d = np.load(os.path.join(DIR, f"uand_seed{seed}.npz"))
    W1, W2, Wo, bo, E = d['W1'], d['W2'], d['Wo'], d['bo'], d['E']
    rng = np.random.default_rng(100+seed)
    Fe, Ye = make_data(rng, 20000)
    Xe = Fe @ E.T
    He = (Xe @ W1.T) * (Xe @ W2.T)
    Pe = sigmoid(He @ Wo.T + bo)
    var = Ye.var()
    fvu_sig = ((Pe-Ye)**2).mean()/var
    lam = 1e-3
    Wr = np.linalg.solve(He.T@He + lam*np.eye(n_hid), He.T@(Ye-Ye.mean(0)))
    fvu_ridge = ((He@Wr + Ye.mean(0) - Ye)**2).mean()/var

    # ---- Q_t pullback to feature space ----
    # logit_t(x) = sum_k Wo[t,k] (W1[k].x)(W2[k].x) + bo[t]  =  x^T Q_t x + bo[t]
    # Pull back through embedding: feature-space form Qf_t = E^T Q_t E  (32x32), symmetrized
    # Build all T at once: Qf[t] = sum_k Wo[t,k] * outer(A1[k], A2[k]) with A1 = W1 @ E (n_hid, m)
    A1 = W1 @ E; A2 = W2 @ E                      # (n_hid, m)
    Qf = np.einsum('tk,ki,kj->tij', Wo, A1, A2)   # (T, m, m)
    Qf = 0.5*(Qf + Qf.transpose(0,2,1))           # symmetrize (x^T Q x only sees sym part)

    # For boolean features x_i^2 = x_i, so diagonal acts like a linear term.
    # Effective logit for sample S (|S|=3): bo[t] + sum_{i in S} Qf[t,i,i] + 2*sum_{i<j in S} Qf[t,i,j]
    # Signal coefficient for target t=(a,b): the 2*Qf[t,a,b] term.
    sig = 2*Qf[np.arange(T), pair_idx[:,0], pair_idx[:,1]]          # (T,)
    diag = Qf[:, np.arange(m), np.arange(m)]                        # (T, m)
    # Off-diagonal interference: all 2*Qf[t,i,j], i<j, EXCLUDING the target pair
    iu = np.triu_indices(m, 1)
    off = 2*Qf[:, iu[0], iu[1]]                                     # (T, T) since #pairs=T
    own = np.zeros((T,T), bool); own[np.arange(T), np.arange(T)] = True
    interf = off[~own].reshape(T, T-1)

    summary[seed] = dict(
        fvu_sig=float(fvu_sig), fvu_ridge=float(fvu_ridge),
        signal_mean=float(sig.mean()), signal_std=float(sig.std()),
        interf_mean=float(interf.mean()), interf_std=float(interf.std()),
        interf_abs_mean=float(np.abs(interf).mean()),
        diag_mean=float(diag.mean()), diag_std=float(diag.std()),
        bias_mean=float(bo.mean()),
        snr=float(sig.mean()/interf.std()),
    )
    if seed == 2:
        np.savez(os.path.join(DIR, "pullback_seed2.npz"), Qf=Qf, sig=sig, diag=diag, off=off, bo=bo)
    print(seed, {k: round(v,4) for k,v in summary[seed].items()})

json.dump(summary, open(os.path.join(DIR, "pullback_summary.json"),"w"), indent=1)
