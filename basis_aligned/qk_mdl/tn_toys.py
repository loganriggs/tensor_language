"""Pedagogy proof-of-concept: two tiny trained toy models that instantiate the two extremes.
TOY A (decomposable node): bilinear model on modular addition (a+b mod p) — learns a low-rank
  Fourier/circular structure, folds to a few frequency features, and the basis is PRIVILEGED
  (top-k frequencies >> random-k). This is 'can be broken down'.
TOY B (dense node, thin bond): one bilinear feature layer serving K downstream consumers, each
  reading a rank-2 slice. The node must be DENSE (high rank, serves everyone, neurons used evenly,
  random ~= ranked) but EACH consumer's communication bond is THIN (rank ~2). This is 'dense,
  can't be broken down, but does sparse communication'.
Extracts demonstration data to tn_toys.json for the Lesson-7 figure.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def eff_rank(sv):
    p = sv / sv.sum()
    return float(np.exp(-(p*np.log(p+1e-12)).sum()))   # participation-ratio effective rank (entropy)


# ================= TOY A: modular addition (decomposable) =================
P, dA, mA = 23, 24, 48
Ea = nn.Embedding(P, dA).to(DEV)
WlA = nn.Linear(dA, mA, bias=False).to(DEV); WrA = nn.Linear(dA, mA, bias=False).to(DEV)
WoutA = nn.Linear(mA, P).to(DEV)
paramsA = list(Ea.parameters())+list(WlA.parameters())+list(WrA.parameters())+list(WoutA.parameters())
optA = torch.optim.Adam(paramsA, lr=3e-3, weight_decay=1e-3)
aa, bb = torch.meshgrid(torch.arange(P), torch.arange(P), indexing='ij')
A_in = aa.reshape(-1).to(DEV); B_in = bb.reshape(-1).to(DEV); Y = ((A_in+B_in) % P)


def fwdA(a, b):
    h = (Ea(a) @ WlA.weight.T) * (Ea(b) @ WrA.weight.T)
    return WoutA(h)


for step in range(6000):
    lg = fwdA(A_in, B_in); loss = F.cross_entropy(lg, Y)
    optA.zero_grad(); loss.backward(); optA.step()
with torch.no_grad():
    accA = float((fwdA(A_in, B_in).argmax(1) == Y).float().mean())
    Emat = Ea.weight.detach().cpu().numpy()                       # (P, dA)
    # Fourier power along vocab axis: DFT of each embedding dim over the P tokens
    F_ = np.fft.rfft(Emat - Emat.mean(0), axis=0)                 # (P//2+1, dA)
    freq_power = (np.abs(F_)**2).sum(1)                           # power per frequency
    freq_power[0] = 0
    order = np.argsort(-freq_power)
    n_freq_eff = eff_rank(freq_power[freq_power > 0])
    # rank-k accuracy: keep only top-k (ranked) vs random-k frequencies in the embedding
    def acc_with_freqs(keep):
        mask = np.zeros(len(freq_power), bool); mask[list(keep)] = True
        Ff = F_ * mask[:, None]
        Erec = np.fft.irfft(Ff, n=P, axis=0) + Emat.mean(0)
        Et = torch.tensor(Erec, dtype=torch.float32, device=DEV)
        h = (Et[A_in] @ WlA.weight.T) * (Et[B_in] @ WrA.weight.T)
        return float((WoutA(h).argmax(1) == Y).float().mean())
    ks = [1, 2, 3, 4, 6, 8]
    rng = np.random.default_rng(0); nz = np.where(freq_power > 0)[0]
    ranked_acc = [acc_with_freqs(order[:k]) for k in ks]
    random_acc = [float(np.mean([acc_with_freqs(rng.choice(nz, min(k, len(nz)), replace=False)) for _ in range(5)])) for k in ks]
    # circle coords: project embedding onto the dominant frequency's cos/sin
    f0 = int(order[0]); ang = 2*np.pi*f0*np.arange(P)/P
    circ = (Emat - Emat.mean(0))
    cx = (circ * np.cos(ang)[:, None]).sum(1); cy = (circ * np.sin(ang)[:, None]).sum(1)

toyA = {'task': 'modular addition a+b mod 23', 'accuracy': round(accA, 3),
        'effective_num_frequencies': round(n_freq_eff, 2), 'hidden_units': mA,
        'dominant_freqs': [int(x) for x in order[:5]],
        'freq_power_top8': [round(float(freq_power[i]), 2) for i in order[:8]],
        'rank_k_ks': ks, 'ranked_acc': [round(x, 3) for x in ranked_acc],
        'random_acc': [round(x, 3) for x in random_acc],
        'circle_x': [round(float(x), 3) for x in cx], 'circle_y': [round(float(y), 3) for y in cy]}
print(f"TOY A (decomposable): acc {accA:.3f}, eff#freqs {n_freq_eff:.2f} of {mA} units; "
      f"ranked-k acc {[round(x,2) for x in ranked_acc]} vs random-k {[round(x,2) for x in random_acc]}", flush=True)

# ================= TOY B: dense node, thin bonds (multi-consumer) =================
din, mB, K, rbond = 24, 64, 16, 2
N = 6000
X = torch.randn(N, din, device=DEV)
# teacher matched to the bilinear node: each output dim is a random RANK-1 QUADRATIC form of x.
# K*rbond = 32 independent quadratics -> the node must carry ~32 features (DENSE), each consumer
# reads only its 2 (THIN bond). Bilinear node CAN fit this, so density is real, not a failure.
Uk = torch.randn(K, rbond, din, device=DEV); Vk = torch.randn(K, rbond, din, device=DEV)
Yt = torch.einsum('krd,nd->nkr', Uk, X) * torch.einsum('krd,nd->nkr', Vk, X)   # (N,K,2)
Yt = (Yt - Yt.mean((0, 1))) / Yt.std((0, 1)).clamp_min(1e-6)
AB1 = nn.Linear(din, mB, bias=False).to(DEV); BB1 = nn.Linear(din, mB, bias=False).to(DEV)
Cons = nn.Parameter(torch.randn(K, mB, rbond, device=DEV) * (1/np.sqrt(mB)))   # each consumer reads h -> 2D
paramsB = list(AB1.parameters())+list(BB1.parameters())+[Cons]
optB = torch.optim.Adam(paramsB, lr=3e-3)


def hB(x):
    return (AB1(x)) * (BB1(x))                    # (N, mB) bilinear dense features


for step in range(6000):
    h = hB(X); pred = torch.einsum('nm,kmr->nkr', h, Cons)
    loss = F.mse_loss(pred, Yt)
    optB.zero_grad(); loss.backward(); optB.step()
with torch.no_grad():
    h = hB(X); pred = torch.einsum('nm,kmr->nkr', h, Cons)
    r2 = 1 - ((pred-Yt)**2).sum().item()/((Yt-Yt.mean((0,1)))**2).sum().item()
    Hn = (h - h.mean(0)).cpu().numpy()
    svH = np.linalg.svd(Hn, compute_uv=False)
    node_eff_rank = eff_rank(svH**2)                             # HIGH = dense node
    # per-consumer bond: what consumer 0 reads = h @ Cons[0] (N,2) -> effective rank ~2
    bond0 = (h @ Cons[0]).cpu().numpy()
    sv_b0 = np.linalg.svd(bond0 - bond0.mean(0), compute_uv=False)
    bond_eff_rank = eff_rank(sv_b0**2)
    # node decomposability: reconstruct h with top-k vs random-k neuron dims
    def recon_task_with_dims(dims):
        mask = torch.zeros(mB, device=DEV); mask[list(dims)] = 1.0
        hp = h * mask; pr = torch.einsum('nm,kmr->nkr', hp, Cons)
        return 1 - ((pr-Yt)**2).sum().item()/((Yt-Yt.mean((0,1)))**2).sum().item()
    # per-neuron importance (ablation drop) flatness -> dense = flat
    base = r2
    imp = np.array([base - recon_task_with_dims([j for j in range(mB) if j != n]) for n in range(mB)])
    imp_cv = float(imp.std()/(abs(imp.mean())+1e-9))            # low CV = evenly used = dense
    ksB = [4, 8, 16, 32, 48]
    order_n = np.argsort(-imp); rngB = np.random.default_rng(1)
    ranked_r2 = [round(recon_task_with_dims(order_n[:k]), 3) for k in ksB]
    random_r2 = [round(float(np.mean([recon_task_with_dims(rngB.choice(mB, k, replace=False)) for _ in range(5)])), 3) for k in ksB]

toyB = {'task': f'{K} consumers, each reads rank-{rbond} of a bilinear feature node',
        'node_task_R2': round(r2, 3), 'node_units': mB,
        'node_effective_rank': round(node_eff_rank, 1), 'per_consumer_bond_eff_rank': round(bond_eff_rank, 2),
        'neuron_importance_CV': round(imp_cv, 3),
        'rank_k_ks': ksB, 'ranked_r2': ranked_r2, 'random_r2': random_r2,
        'K_consumers': K, 'bond_rank': rbond}
print(f"TOY B (dense/thin-bond): node eff-rank {node_eff_rank:.1f} of {mB} (dense); per-consumer bond "
      f"eff-rank {bond_eff_rank:.2f} (thin); ranked-k R2 {ranked_r2} ~ random-k {random_r2} (not decomposable)", flush=True)

json.dump({'toyA_decomposable': toyA, 'toyB_dense_thinbond': toyB}, open('tn_toys.json', 'w'), indent=1)
print('TN TOYS DONE', flush=True)
