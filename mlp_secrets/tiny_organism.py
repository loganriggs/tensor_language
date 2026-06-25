import numpy as np, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# An even tinier readable organism: n=5-bit strings (2^5 = 32 possible), 4 secrets,
# hidden width h=6. Small enough to brute-force ALL 32 strings and show every logit,
# which makes the bit-complement symmetry explicit: a secret and its all-bits-flipped
# complement score identically because x^TQx is even ((-x)^TQ(-x)=x^TQx, and -x in the
# ±1 encoding IS the complement bit string). Dumps weights + secrets to tiny_organism.md.
DIR = os.path.dirname(os.path.abspath(__file__))
n, NSEC, h = 5, 4, 6
rng0 = np.random.default_rng(4)
secrets = rng0.choice([-1.0, 1.0], size=(NSEC, n))
allX = np.array(np.meshgrid(*[[-1.0, 1.0]]*n)).reshape(n, -1).T                  # all 32 strings
def sigmoid(z): return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
def bits(v): return ''.join('1' if x > 0 else '0' for x in v)

is_pos = np.array([any(np.array_equal(x, s) for s in secrets) for x in allX])
is_comp = np.array([any(np.array_equal(x, -s) for s in secrets) for x in allX])
neg_pool = allX[~is_pos & ~is_comp]          # negatives exclude secrets AND their complements,
                                             # since a pure bilinear model cannot separate x from -x

def train(steps=8000, B=256, lr=3e-3, seed=1):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(h, n))/np.sqrt(n); W2 = rng.normal(size=(h, n))/np.sqrt(n)
    Wo = rng.normal(size=h)/np.sqrt(h); bo = np.array(-3.0)
    ps = [W1, W2, Wo, bo]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        pos = secrets[rng.integers(NSEC, size=B//2)]
        neg = neg_pool[rng.integers(len(neg_pool), size=B-B//2)]
        X = np.vstack([pos, neg]); Y = np.concatenate([np.ones(B//2), np.zeros(B-B//2)])
        A = X@W1.T; Bv = X@W2.T; H = A*Bv; Z = H@Wo + bo; dZ = (sigmoid(Z)-Y)/B
        dWo = H.T@dZ; dbo = dZ.sum(); dH = np.outer(dZ, Wo); dW1 = (dH*Bv).T@X; dW2 = (dH*A).T@X
        for i, (p, g) in enumerate(zip(ps, [dW1, dW2, dWo, dbo])):
            ms[i] = b1*ms[i]+(1-b1)*g; vs[i] = b2*vs[i]+(1-b2)*g*g
            p -= lr*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s))+eps)
    return W1, W2, Wo, float(bo)

W1, W2, Wo, bo = train()
Q = 0.5*(np.einsum('k,ki,kj->ij', Wo, W1, W2) + np.einsum('k,ki,kj->ij', Wo, W2, W1))
log = np.einsum('ni,ij,nj->n', allX, Q, allX) + bo
w, V = np.linalg.eigh(Q)
def hits(cands):
    g = set()
    for c in cands:
        for si, s in enumerate(secrets):
            if np.array_equal(np.sign(c), s) or np.array_equal(np.sign(c), -s): g.add(si)
    return len(g)

def tbl(M, rl, cl):
    out = ["| | " + " | ".join(cl) + " |", "|" + "---|"*(len(cl)+1)]
    for r, row in zip(rl, np.atleast_2d(M)):
        out.append(f"| **{r}** | " + " | ".join(f"{v:+.2f}" for v in row) + " |")
    return "\n".join(out)

xf = [f'x{j}' for j in range(n)]
md = [f"# A tiny readable bilinear organism (n={n}, only 2^{n}={2**n} strings)\n",
      f"`python tiny_organism.py`. 1-layer bilinear membership classifier over **{n}-bit strings**, "
      f"**{NSEC} secrets**, hidden width **h={h}**. Small enough to brute-force every one of the "
      f"{2**n} strings.\n",
      "## The bit-complement symmetry\n",
      "Bits are encoded as `±1` (bit `0`→`−1`, bit `1`→`+1`), so a string like `10` is the *vector* "
      "`(+1,−1)`. The folded form `xᵀQx` is **even**: `(−x)ᵀQ(−x) = xᵀQx`. Negating the ±1 vector flips "
      "every bit — e.g. `−(+1,−1) = (−1,+1)`, which decodes to `01`, the bit-complement of `10`. So a "
      "secret and its all-bits-flipped complement **always get the identical logit**; a pure bilinear "
      "model cannot tell them apart. (We therefore exclude complements from the negatives during "
      "training, and count recovery up to a global flip.)\n",
      "## The 4 secrets (and their complements get equal logits)\n",
      "| # | secret | logit | complement | logit |\n|---|---|---|---|---|"]
for i, s in enumerate(secrets):
    ls = log[np.array([np.array_equal(x, s) for x in allX])][0]
    lc = log[np.array([np.array_equal(x, -s) for x in allX])][0]
    md.append(f"| {i} | `{bits(s)}` | {ls:+.2f} | `{bits(-s)}` | {lc:+.2f} |")
od = np.argsort(-log)
md.append(f"\nBest **non**-member logit (any string that isn't a secret or complement): "
          f"{log[~is_pos & ~is_comp].max():+.2f}. So the 8 highest of all {2**n} strings are exactly the "
          f"4 secrets and their 4 complements.\n")
md.append("## All %d strings, sorted by logit\n" % 2**n)
md.append("| string | logit | kind |\n|---|---|---|")
for r in od:
    kind = "secret" if is_pos[r] else ("complement" if is_comp[r] else "—")
    md.append(f"| `{bits(allX[r])}` | {log[r]:+.2f} | {kind} |")
md.append("\n## Weights\n")
md.append(f"**D = Wo** (length {h}): " + " ".join(f"`{v:+.2f}`" for v in Wo) + f"  **bias** = `{bo:+.2f}`\n")
md.append(f"### L = W1 ({h}×{n})\n" + tbl(W1, [f"h{k}" for k in range(h)], xf) + "\n")
md.append(f"### R = W2 ({h}×{n})\n" + tbl(W2, [f"h{k}" for k in range(h)], xf) + "\n")
md.append(f"### Q (folded quadratic form, {n}×{n})\n" + tbl(Q, xf, xf) + "\n")
md.append("![weights](./fig_tiny_weights.png)\n")
md.append("## Can you read the secrets off the weights?\n")
md.append(f"- sign of an L/R row matching a secret (±): **{hits(np.vstack([W1, W2]))} / {NSEC}**\n"
          f"- sign of a top-{NSEC} eigenvector matching a secret (±): **{hits([V[:, np.argsort(-w)[i]] for i in range(NSEC)])} / {NSEC}**\n"
          f"- Q eigenvalues: {np.round(np.sort(w)[::-1], 2).tolist()}\n")
md.append("![Q directions vs secrets](./fig_tiny_Q_directions.png)\n")
open(os.path.join(DIR, "tiny_organism.md"), "w").write("\n".join(md)+"\n")

# figures
def show(ax, M, title, lim=None, xl=None, yl=None):
    M = np.atleast_2d(M); lim = lim or max(np.abs(M).max(), 1e-9)
    im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_title(title, fontsize=10); plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels(xl if xl is not None else range(M.shape[1]), fontsize=7)
    ax.set_yticklabels(yl if yl is not None else range(M.shape[0]), fontsize=8)
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
show(ax[0, 0], W1, 'L = W1', xl=xf, yl=[f'h{k}' for k in range(h)])
show(ax[0, 1], W2, 'R = W2', xl=xf, yl=[f'h{k}' for k in range(h)])
show(ax[1, 0], Wo[None, :], 'D = Wo', xl=[f'h{k}' for k in range(h)], yl=['D'])
show(ax[1, 1], Q, 'Q (folded)', xl=xf, yl=xf)
fig.suptitle(f'Tiny organism weights (n={n}, h={h}, {NSEC} secrets)', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(DIR, 'fig_tiny_weights.png'), dpi=120)

idx = list(dict.fromkeys(list(np.argsort(-w))))     # all eigvecs, most-positive λ first (n=5 -> show all 5)
fig, ax = plt.subplots(2, 1, figsize=(8, 5.5))
show(ax[0], V[:, idx].T, "Q eigen-directions (all 5, most-positive λ first)", xl=xf,
     yl=[f'λ={w[i]:+.2f}' for i in idx])
show(ax[1], secrets, 'the 4 secrets (±1)', lim=1, xl=xf, yl=[f'secret {i}' for i in range(NSEC)])
fig.tight_layout(); fig.savefig(os.path.join(DIR, 'fig_tiny_Q_directions.png'), dpi=120)
print(f"wrote tiny_organism.md; secret logits {np.round(log[is_pos],2)}; best non-member {log[~is_pos & ~is_comp].max():.2f}; "
      f"L/R hits {hits(np.vstack([W1,W2]))}; eig hits {hits([V[:,np.argsort(-w)[i]] for i in range(NSEC)])}")
