import numpy as np, itertools, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------------
# Toy Universal-AND on 4 features, 6 pairwise-AND outputs, run for two input
# distributions: 3-hot (C(4,3)=4 inputs) and 2-hot (C(4,2)=6 inputs).
#   h = (W1 x)⊙(W2 x),  logit = Wo h + bo  (W1,W2 biasless; bo is the only bias)
# For each: sweep the hidden width h to find the minimum that computes all 6 ANDs,
# then display the minimal model the usual way (Qf decomposition + logit ladder +
# ladder decomposition, with bo shown).
#
# Note on 3-hot: the 6 ANDs are 3 complementary pairs (AND(0,1)=¬AND(2,3) ...),
# i.e. the three 2/2 partitions of the 4 inputs — one is the XOR ("diagonal")
# split. Naively that looks like a 2-D obstruction, but the net places the 4
# hidden states in NON-convex position (one inside the others' triangle) so all
# three partitions become linearly separable, and h=2 already suffices.
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)
m = 4
pairs = list(itertools.combinations(range(m), 2)); T = len(pairs); pi = np.array(pairs)   # 6 pairs
def sigmoid(z): z = np.clip(z, -60, 60); return 1/(1+np.exp(-z))
colors = {0: '#888', 1: '#d62728', 2: '#2ca02c'}; labs = {0: 'neg, shares 0', 1: 'neg, shares 1', 2: 'positive'}

def train(Xall, Yall, seed, h, steps=4000, lr=0.03, wd=2e-3):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(h, m))/np.sqrt(m); W2 = rng.normal(size=(h, m))/np.sqrt(m)
    Wo = rng.normal(size=(T, h))/np.sqrt(h); bo = np.full(T, -2.0)
    ps = [W1, W2, Wo, bo]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        P = Xall @ W1.T; Q = Xall @ W2.T; H = P*Q; Z = H @ Wo.T + bo
        dZ = (sigmoid(Z) - Yall)/len(Xall)
        dWo = dZ.T @ H; dbo = dZ.sum(0); dH = dZ @ Wo; dP = dH*Q; dQ = dH*P
        dW1 = dP.T @ Xall; dW2 = dQ.T @ Xall
        for i, (p, g) in enumerate(zip(ps, [dW1, dW2, dWo, dbo])):
            ms[i] = b1*ms[i] + (1-b1)*g; vs[i] = b2*vs[i] + (1-b2)*g*g
            p -= lr*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s)) + eps)
            if i < 3: p -= lr*wd*p            # decoupled weight decay on W1,W2,Wo (keeps logits moderate)
    Z = (Xall @ W1.T * (Xall @ W2.T)) @ Wo.T + bo
    return dict(W1=W1, W2=W2, Wo=Wo, bo=bo, acc=((Z > 0) == (Yall > .5)).mean())

def run(KHOT):
    tag = f"{KHOT}hot"
    Xall = np.array([[1.0 if i in S else 0.0 for i in range(m)]
                     for S in itertools.combinations(range(m), KHOT)])
    Yall = np.stack([Xall[:, a]*Xall[:, b] for a, b in pairs], 1)
    nd = Xall.shape[0]*T
    print(f"\n=== {KHOT}-hot: {Xall.shape[0]} inputs, {T} pairwise-AND outputs, "
          f"{int(Yall.sum())} positives / {nd} decisions ===")
    best_by_h = {h: max((train(Xall, Yall, s, h) for s in range(16)), key=lambda r: r['acc']) for h in range(1, 7)}
    for h in range(1, 7):
        print(f"  h={h}: best acc {100*best_by_h[h]['acc']:5.1f}%")
    h_min = min((h for h, r in best_by_h.items() if r['acc'] == 1.0), default=6)
    M = best_by_h[h_min]; W1, W2, Wo, bo = M['W1'], M['W2'], M['Wo'], M['bo']
    print(f"  -> minimal h reaching 100%: h={h_min}")
    Qf = np.einsum('tk,ki,kj->tij', Wo, W1, W2); Qf = 0.5*(Qf + Qf.transpose(0, 2, 1))
    Zc = np.einsum('tij,ni,nj->nt', Qf, Xall, Xall) + bo
    print("  Qf reproduces forward? err", f"{np.abs(Zc-((Xall@W1.T*(Xall@W2.T))@Wo.T+bo)).max():.0e}",
          "| bo", {f"{pairs[t]}": round(float(bo[t]), 1) for t in range(T)})
    lab = ["".join(str(i) for i in range(m) if Xall[n, i]) for n in range(len(Xall))]
    print("  detection check (output -> active-sets where logit>0 / should fire):")
    for t in range(T):
        fires = [lab[n] for n in range(len(Xall)) if Zc[n, t] > 0]
        should = [lab[n] for n in range(len(Xall)) if Yall[n, t] > 0]
        print(f"    AND{pairs[t]}: fires {fires}  should {should}  {'ok' if fires == should else 'MISMATCH'}")

    # (1) Qf decomposition with bias
    fig, axes = plt.subplots(T, 4, figsize=(11, 13))
    headers = ['Qf (full)', '= signal (AND-ing)', '+ diagonal inhibition', '+ interference']
    for t in range(T):
        a, b = pi[t]; Mt = Qf[t]
        Sg = np.zeros_like(Mt); Sg[a, b] = Mt[a, b]; Sg[b, a] = Mt[b, a]
        Dg = np.diag(np.diag(Mt)); I_ = Mt - Sg - Dg
        lim = max(np.abs(Mt).max(), 1e-6)                       # per-row (per-output) colour scale
        for c, mat in enumerate([Mt, Sg, Dg, I_]):
            ax = axes[t, c]; ax.imshow(mat, cmap='RdBu_r', vmin=-lim, vmax=lim)
            for i in range(m):
                for j in range(m):
                    if abs(mat[i, j]) > 1e-2: ax.text(j, i, f"{mat[i,j]:.0f}", ha='center', va='center', fontsize=7)
            ax.set_xticks(range(m)); ax.set_yticks(range(m))
            ax.set_xticklabels([f'x{i}' for i in range(m)], fontsize=6); ax.set_yticklabels([f'x{i}' for i in range(m)], fontsize=6)
            if t == 0: ax.set_title(headers[c], fontsize=11)
        axes[t, 0].set_ylabel(f'AND(x{a},x{b})\nbo={bo[t]:+.1f}  peak|Qf|={lim:.0f}', fontsize=8)
    fig.suptitle(f'Toy-4 {KHOT}-hot Qf decomposition — 6 ANDs through h={h_min} neurons (each ROW on its own colour scale)\n'
                 'logit = bias + signal + diagonal inhibition + interference', fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f'fig_toy4_{tag}_decomp.png'), dpi=110)

    # ---- (1b) hollow / mean-0 canonical view (hollow.py trick) ----
    # boolean x_i^2=x_i  -> diagonal acts linear: pull diag into a per-feature linear vector.
    # k-hot sum is constant -> off-diagonal & linear have a constant gauge: mean-center both.
    # logit = bias' + lin.x + x^T H x is then EXACT on the k-hot inputs (verified).
    off = ~np.eye(m, dtype=bool)
    def canon(t):
        Q = Qf[t]; lin = np.diag(Q).copy(); H = Q - np.diag(lin)
        mu = H[off].mean(); H = H - mu*off; lm = lin.mean(); lin = lin - lm
        bias2 = bo[t] + mu*KHOT*(KHOT-1) + lm*KHOT
        return lin, H, bias2
    err = max(np.abs((b2 + Xall@l + np.einsum('ni,ij,nj->n', Xall, H, Xall)) - Zc[:, t]).max()
              for t in range(T) for (l, H, b2) in [canon(t)])
    print(f"  hollow+centered canonical form reproduces logits on {KHOT}-hot inputs? max err {err:.0e}")
    fig, axes = plt.subplots(T, 2, figsize=(7.5, 13), gridspec_kw={'width_ratios': [1, 3.4]})
    for t in range(T):
        a, b = pi[t]; lin, H, bias2 = canon(t)
        lim = max(np.abs(H).max(), np.abs(lin).max(), 1e-6)
        axes[t, 0].imshow(lin[None, :], cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
        for i in range(m): axes[t, 0].text(i, 0, f"{lin[i]:.0f}", ha='center', va='center', fontsize=7)
        axes[t, 0].set_xticks(range(m)); axes[t, 0].set_xticklabels([f'x{i}' for i in range(m)], fontsize=6); axes[t, 0].set_yticks([])
        axes[t, 1].imshow(H, cmap='RdBu_r', vmin=-lim, vmax=lim)
        for i in range(m):
            for j in range(m):
                if abs(H[i, j]) > 1e-2: axes[t, 1].text(j, i, f"{H[i,j]:.0f}", ha='center', va='center', fontsize=7)
        for (yy, xx) in [(a, b), (b, a)]: axes[t, 1].add_patch(Rectangle((xx-.5, yy-.5), 1, 1, fill=False, ec='lime', lw=2.2))
        axes[t, 1].set_xticks(range(m)); axes[t, 1].set_yticks(range(m))
        axes[t, 1].set_xticklabels([f'x{i}' for i in range(m)], fontsize=6); axes[t, 1].set_yticklabels([f'x{i}' for i in range(m)], fontsize=6)
        if t == 0:
            axes[t, 0].set_title('linear l\n(was diagonal)', fontsize=9)
            axes[t, 1].set_title('hollow interaction H (diag=0, off-diag mean-0)\nsignal cell boxed', fontsize=9)
        axes[t, 0].set_ylabel(f'AND(x{a},x{b})\nbias={bias2:+.0f}', fontsize=8)
    fig.suptitle(f'Toy-4 {KHOT}-hot CANONICAL view (hollowed): logit = bias + lin.x + x^T H x, exact on {KHOT}-hot\n'
                 'diagonal -> linear l; off-diagonal mean-centered; each row on its own colour scale', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f'fig_toy4_{tag}_hollow.png'), dpi=110)

    # logit components per (input, output)
    diag = Qf[:, np.arange(m), np.arange(m)]
    def comps(n, t):
        a, b = pi[t]; x = Xall[n]; inh = (diag[t]*x).sum(); sig = interf = 0.0
        for (i, j) in pairs:
            v = 2*Qf[t, i, j]*x[i]*x[j]
            if (i, j) == (a, b): sig += v
            else: interf += v
        return bo[t], inh, sig, interf
    rows = [(int(Xall[n, pi[t][0]]+Xall[n, pi[t][1]]), *comps(n, t)) for n in range(len(Xall)) for t in range(T)]
    shares = np.array([r[0] for r in rows]); comp = np.array([r[1:] for r in rows])
    present = sorted(set(shares)); ypos = {k: i for i, k in enumerate(present)}
    def ladder(ax, L, title):
        for k in present:
            mk = shares == k
            ax.scatter(L[mk], ypos[k] + np.linspace(-.12, .12, max(mk.sum(), 1)), color=colors[k], s=42, label=labs[k])
        ax.axvline(0, color='k', lw=1.5); ax.set_yticks(list(ypos.values())); ax.set_yticklabels([labs[k] for k in present])
        ax.set_xlabel('logit'); ax.set_title(title)

    # (2) logit ladder
    fig, ax = plt.subplots(figsize=(9, 3.0 + .3*len(present))); L = comp.sum(1)
    ladder(ax, L, f'Toy-4 {KHOT}-hot logit ladder — {nd} decisions, {100*M["acc"]:.0f}% at threshold 0')
    q = max(40, 1.25*np.percentile(np.abs(L), 88)); ax.set_xlim(-q, q)         # clip far (correct) outliers
    ax.legend(fontsize=8, loc='lower right'); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f'fig_toy4_{tag}_ladder.png'), dpi=110)

    # (3) logit ladder decomposition
    variants = [("full", [0, 1, 2, 3]), ("no interference", [0, 1, 2]), ("no inhibition", [0, 2, 3])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.4 + .3*len(present)), sharex=True, sharey=True)
    abl = {}; Ls = [comp[:, c].sum(1) for _, c in variants]
    q = max(40, 1.25*np.percentile(np.abs(np.concatenate(Ls)), 88))
    for ax, (ttl, c), L in zip(axes, variants, Ls):
        pos = L[shares == 2]; neg = L[shares != 2]
        acc = ((pos > 0).sum() + (neg <= 0).sum())/len(L); abl[ttl] = acc
        ladder(ax, L, f"{ttl} — acc {100*acc:.0f}%"); ax.set_xlim(-q, q)
    fig.suptitle(f'Toy-4 {KHOT}-hot logit ladder decomposition', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, f'fig_toy4_{tag}_ladder_decomp.png'), dpi=110)
    print("  ablation acc:", {k: f"{100*v:.0f}%" for k, v in abl.items()})
    return dict(KHOT=KHOT, n_in=Xall.shape[0], h_min=h_min, acc=M['acc'], best_by_h=best_by_h, abl=abl)

summ = [run(3), run(2)]
print("\n=== summary ===")
for s in summ:
    print(f"  {s['KHOT']}-hot: {s['n_in']} inputs, 6 outputs -> minimal h={s['h_min']} (100%); "
          f"ablation " + ", ".join(f"{k} {100*v:.0f}%" for k, v in s['abl'].items()))
print("figures: fig_toy4_3hot_* and fig_toy4_2hot_*")
