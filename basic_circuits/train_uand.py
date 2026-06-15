import numpy as np, itertools, json, time, os

DIR = os.path.dirname(os.path.abspath(__file__))
# Config: m=32 features, d0=16 embedding, n_hidden=64, T=C(32,2)=496 ANDs, 3-hot
m, d0, n_hid = 32, 16, 64
pairs = list(itertools.combinations(range(m), 2))
T = len(pairs)
pair_idx = np.array(pairs)  # (T,2)
STEPS, BATCH, LR = 3000, 1024, 2e-3
SEEDS = [0, 1, 2]

def make_data(rng, n):
    # n samples of 3-hot boolean features (vectorized)
    R = rng.random((n, m))
    idx = np.argpartition(R, 3, axis=1)[:, :3]
    F = np.zeros((n, m))
    np.put_along_axis(F, idx, 1.0, axis=1)
    Y = F[:, pair_idx[:, 0]] * F[:, pair_idx[:, 1]]  # (n, T)
    return F, Y

def sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

results = {}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    # Random unit-norm embedding columns
    E = rng.normal(size=(d0, m)); E /= np.linalg.norm(E, axis=0, keepdims=True)
    # Params
    W1 = rng.normal(size=(n_hid, d0)) / np.sqrt(d0)
    W2 = rng.normal(size=(n_hid, d0)) / np.sqrt(d0)
    Wo = rng.normal(size=(T, n_hid)) / np.sqrt(n_hid)
    bo = np.full(T, -4.0)  # bias init near log(p/(1-p)), p~3/496
    params = [W1, W2, Wo, bo]
    ms_ = [np.zeros_like(p) for p in params]
    vs_ = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8

    t0 = time.time()
    for step in range(1, STEPS+1):
        F, Y = make_data(rng, BATCH)
        X = F @ E.T                       # (B, d0)
        A = X @ W1.T; Bv = X @ W2.T       # (B, n_hid)
        H = A * Bv
        Z = H @ Wo.T + bo                 # (B, T)
        P = sigmoid(Z)
        dZ = (P - Y) / BATCH
        dWo = dZ.T @ H; dbo = dZ.sum(0)
        dH = dZ @ Wo
        dA = dH * Bv; dB = dH * A
        dW1 = dA.T @ X; dW2 = dB.T @ X
        grads = [dW1, dW2, dWo, dbo]
        for i, (p, g) in enumerate(zip(params, grads)):
            ms_[i] = b1*ms_[i] + (1-b1)*g
            vs_[i] = b2*vs_[i] + (1-b2)*g*g
            mh = ms_[i]/(1-b1**step); vh = vs_[i]/(1-b2**step)
            p -= LR * mh/(np.sqrt(vh)+eps)
        if step % 500 == 0:
            bce = -(Y*np.log(P+1e-9) + (1-Y)*np.log(1-P+1e-9)).mean()
            print(f"seed {seed} step {step} bce {bce:.5f} ({time.time()-t0:.0f}s)", flush=True)

    # Eval on fresh data
    Fe, Ye = make_data(rng, 20000)
    Xe = Fe @ E.T
    He = (Xe @ W1.T) * (Xe @ W2.T)
    Ze = He @ Wo.T + bo
    Pe = sigmoid(Ze)
    var = Ye.var()
    fvu_sig = ((Pe - Ye)**2).mean() / var
    # Ridge readout from hidden (linear-interface FVU)
    lam = 1e-3
    G = He.T @ He + lam*np.eye(n_hid)
    Wr = np.linalg.solve(G, He.T @ (Ye - Ye.mean(0)))
    fvu_ridge = ((He @ Wr + Ye.mean(0) - Ye)**2).mean() / var
    # Threshold accuracy: balanced acc at 0.5
    pred = Pe > 0.5
    tp = (pred & (Ye > 0.5)).sum() / (Ye > 0.5).sum()
    tn = (~pred & (Ye < 0.5)).sum() / (Ye < 0.5).sum()
    print(f"seed {seed}: FVU(sigmoid)={fvu_sig:.4f}  FVU(ridge-hidden)={fvu_ridge:.4f}  TPR={tp:.4f} TNR={tn:.4f}")
    results[seed] = dict(fvu_sig=float(fvu_sig), fvu_ridge=float(fvu_ridge), tpr=float(tp), tnr=float(tn))
    np.savez(os.path.join(DIR, f"uand_seed{seed}.npz"), W1=W1, W2=W2, Wo=Wo, bo=bo, E=E)

json.dump(results, open(os.path.join(DIR, "uand_results.json"), "w"), indent=1)
print("ceiling 1-n/T =", 1 - n_hid/T)
