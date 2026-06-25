import numpy as np, os

# A small, human-readable 1-layer bilinear organism so the actual weights fit in a
# markdown file. n=12-bit strings, 4 secrets, hidden width h=6. Dumps L=W1, R=W2,
# D=Wo, bias, the folded quadratic form Q, and the secret strings to small_organism.md.
DIR = os.path.dirname(os.path.abspath(__file__))
n, NSEC, h = 12, 4, 6
rng0 = np.random.default_rng(2)
secrets = rng0.choice([-1.0, 1.0], size=(NSEC, n))
def sigmoid(z): return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train(steps=8000, B=256, lr=3e-3, seed=1):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(h, n))/np.sqrt(n); W2 = rng.normal(size=(h, n))/np.sqrt(n)
    Wo = rng.normal(size=h)/np.sqrt(h); bo = np.array(-3.0)
    ps = [W1, W2, Wo, bo]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        pos = secrets[rng.integers(NSEC, size=B//2)]
        neg = rng.choice([-1.0, 1.0], size=(B-B//2, n))
        X = np.vstack([pos, neg]); Y = np.concatenate([np.ones(B//2), np.zeros(B-B//2)])
        A = X@W1.T; Bv = X@W2.T; H = A*Bv; Z = H@Wo + bo; dZ = (sigmoid(Z)-Y)/B
        dWo = H.T@dZ; dbo = dZ.sum(); dH = np.outer(dZ, Wo); dW1 = (dH*Bv).T@X; dW2 = (dH*A).T@X
        for i, (p, g) in enumerate(zip(ps, [dW1, dW2, dWo, dbo])):
            ms[i] = b1*ms[i]+(1-b1)*g; vs[i] = b2*vs[i]+(1-b2)*g*g
            p -= lr*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s))+eps)
    return W1, W2, Wo, float(bo)

W1, W2, Wo, bo = train()
Q = 0.5*(np.einsum('k,ki,kj->ij', Wo, W1, W2) + np.einsum('k,ki,kj->ij', Wo, W2, W1))
allX = np.array(np.meshgrid(*[[-1.0, 1.0]]*n)).reshape(n, -1).T          # all 2^12 strings
log = np.einsum('ni,ij,nj->n', allX, Q, allX) + bo
sec_log = np.einsum('ni,ij,nj->n', secrets, Q, secrets) + bo
order = np.argsort(-log)
top = allX[order[:8]]
# which secrets are visible? sign of L/R rows, top eigenvectors
w, V = np.linalg.eigh(Q)
def hits(cands):
    g = set()
    for c in cands:
        for si, s in enumerate(secrets):
            if np.array_equal(np.sign(c), s) or np.array_equal(np.sign(c), -s): g.add(si)
    return len(g)

def bits(v): return ''.join('1' if x > 0 else '0' for x in v)
def tbl(M, rl, cl):
    out = ["| | " + " | ".join(cl) + " |", "|" + "---|"*(len(cl)+1)]
    for r, row in zip(rl, np.atleast_2d(M)):
        out.append(f"| **{r}** | " + " | ".join(f"{v:+.2f}" for v in row) + " |")
    return "\n".join(out)

md = []
md.append("# A small readable bilinear organism (its weights + secrets)\n")
md.append(f"`python small_organism.py`. A 1-layer bilinear membership classifier, "
          f"**n={n}-bit strings**, **{NSEC} secrets**, hidden width **h={h}**. "
          f"logit(x) = x·(L x)⊙(R x) read by D, i.e. `xᵀQx + b`. Small enough to print in full "
          f"so you can try to spot the secrets in the weights.\n")
md.append("## The secret strings (what we're trying to recover)\n")
md.append("| # | bit string | ±1 vector |\n|---|---|---|")
for i, s in enumerate(secrets):
    md.append(f"| {i} | `{bits(s)}` | {' '.join(f'{int(v):+d}' for v in s)} |")
ismem = np.array([any(np.array_equal(x, s) or np.array_equal(x, -s) for s in secrets) for x in allX])
md.append(f"\nMemorisation (brute-forced over all 2^{n} strings): secret logits = "
          f"{np.round(sec_log,1).tolist()}; best **non**-secret logit = {log[~ismem].max():.1f}. "
          f"The 8 highest-logit strings are exactly the 4 secrets and their 4 bit-complements "
          f"(`xᵀQx` is even, so each secret and its flip score equally).\n")
md.append("## Weights\n")
md.append(f"**D = Wo** (output / decoder, length {h}): " + " ".join(f"`{v:+.2f}`" for v in Wo) + f"  **bias b** = `{bo:+.2f}`\n")
md.append(f"### L = W1  ({h}×{n})\n" + tbl(W1, [f"h{k}" for k in range(h)], [f"x{j}" for j in range(n)]) + "\n")
md.append(f"### R = W2  ({h}×{n})\n" + tbl(W2, [f"h{k}" for k in range(h)], [f"x{j}" for j in range(n)]) + "\n")
md.append(f"### Q = Σ_k D[k]·sym(L[k]⊗R[k])  (folded quadratic form, {n}×{n})\n"
          + tbl(Q, [f"x{i}" for i in range(n)], [f"x{j}" for j in range(n)]) + "\n")
md.append("![weights](./fig_small_weights.png)\n")
md.append("## Can you see the secrets in here?\n")
md.append("![top-3 Q directions vs secrets](./fig_small_Q_directions.png)\n")
md.append(f"- sign of an **L or R row** matching a secret (±): **{hits(np.vstack([W1, W2]))} / {NSEC}**\n"
          f"- sign of a **top-{NSEC} eigenvector** of Q matching a secret (±): "
          f"**{hits([V[:, np.argsort(-w)[i]] for i in range(NSEC)])} / {NSEC}**\n"
          f"- Q eigenvalues: {np.round(np.sort(w)[::-1], 2).tolist()}\n")
md.append("The 8 highest-logit strings over all 2^%d inputs (the secrets should be among them):\n" % n)
md.append("| rank | string | xᵀQx+b | is a secret? |\n|---|---|---|---|")
for r in range(8):
    x = allX[order[r]]; issec = any(np.array_equal(x, s) or np.array_equal(x, -s) for s in secrets)
    md.append(f"| {r} | `{bits(x)}` | {log[order[r]]:.1f} | {'yes' if issec else 'no'} |")
open(os.path.join(DIR, "small_organism.md"), "w").write("\n".join(md)+"\n")
print("wrote small_organism.md; secret logits", np.round(sec_log, 1),
      "| L/R-row hits", hits(np.vstack([W1, W2])), "| eig hits", hits([V[:, np.argsort(-w)[i]] for i in range(NSEC)]))

# ---------- imshow figures ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def show(ax, M, title, lim=None, xl=None, yl=None):
    M = np.atleast_2d(M); lim = lim or max(np.abs(M).max(), 1e-9)
    im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_title(title, fontsize=10); plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels(xl if xl is not None else range(M.shape[1]), fontsize=6)
    ax.set_yticklabels(yl if yl is not None else range(M.shape[0]), fontsize=7)

xfeat = [f'x{j}' for j in range(n)]
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
show(ax[0, 0], W1, 'L = W1  (h×n)', xl=xfeat, yl=[f'h{k}' for k in range(h)])
show(ax[0, 1], W2, 'R = W2  (h×n)', xl=xfeat, yl=[f'h{k}' for k in range(h)])
show(ax[1, 0], Wo[None, :], 'D = Wo  (decoder)', xl=[f'h{k}' for k in range(h)], yl=['D'])
show(ax[1, 1], Q, 'Q = Σ_k D[k]·sym(L[k]⊗R[k])  (folded quadratic form)', xl=xfeat, yl=xfeat)
fig.suptitle(f'Small bilinear organism weights (n={n}, h={h}, {NSEC} secrets)', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(DIR, 'fig_small_weights.png'), dpi=120)

pos_idx = np.argsort(-w)[:3]                            # top-3 MOST POSITIVE eigenvalues (maximise x^TQx)
neg_idx = np.argsort(w)[:3]                             # top-3 most negative (directions to avoid)
dirs = np.vstack([V[:, pos_idx].T, V[:, neg_idx].T])
ylab = [f'λ={w[i]:+.2f}  (+)' for i in pos_idx] + [f'λ={w[i]:+.2f}  (−)' for i in neg_idx]
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
show(ax[0], dirs, "Q eigen-directions: top-3 POSITIVE λ (where x^TQx is maximised) then top-3 negative", xl=xfeat, yl=ylab)
ax[0].axhline(2.5, color='k', lw=1.5)
show(ax[1], secrets, 'The 4 secret strings (±1)', lim=1, xl=xfeat, yl=[f'secret {i}' for i in range(NSEC)])
fig.suptitle("Q's eigen-directions vs the secrets — the secrets aren't any of the eigenvectors", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(DIR, 'fig_small_Q_directions.png'), dpi=120)
print("figures: fig_small_weights.png, fig_small_Q_directions.png")
