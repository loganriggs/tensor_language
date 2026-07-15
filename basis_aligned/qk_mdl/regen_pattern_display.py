import torch, sys
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/tensor_language')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from palette import INK, SECONDARY, DIVERGING
from tier2_model import load_elriggs, build_eval_tokens, reference_forward
from tier2_folding import branch_factors, scores_from_factors

torch.manual_seed(0)
DEV = 'cuda'
m, cfg = load_elriggs('bilin18')
NH, HD, V = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['vocab_size']
FACT = {br: branch_factors(m, br, dtype=torch.float32) for br in (1, 2)}


def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 8192):
            xx = X[i:i + 8192]
            assign[i:i + 8192] = ((xx ** 2).sum(1, keepdim=True) - 2 * xx @ C.T
                                  + (C ** 2).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign


K = 256
tabs = {k: v.cuda() for k, v in torch.load('ce_codebook_k256.pt').items()}
ASSIGN = {}
for br in (1, 2):
    qh_all, kh_all = FACT[br]
    for hh in range(NH):
        _, assign = kmeans(torch.cat([qh_all[:, hh], kh_all[:, hh]], 1), K)
        ASSIGN[(br, hh)] = assign

TOK = build_eval_tokens(n_chunks=1, seq_len=513)
snippet = TOK[0, 40:88].to(DEV)
tokens = snippet[None]
cap = {}


def capture(li, s1, s2):
    if li == 0:
        cap['s1'], cap['s2'] = s1.float(), s2.float()
    return s1, s2


reference_forward(m, tokens, score_patch=capture)
mask = torch.tril(torch.ones(48, 48, device=DEV))
pat_orig = (cap['s1'] * cap['s2'])[0] * mask


def codebook_scores(br):
    q = torch.stack([tabs[f'q{br}'][hh][ASSIGN[(br, hh)]] for hh in range(NH)], 1)
    k = torch.stack([tabs[f'k{br}'][hh][ASSIGN[(br, hh)]] for hh in range(NH)], 1)
    return scores_from_factors(q, k, tokens, HD)


pat_cb = (codebook_scores(1) * codebook_scores(2))[0] * mask

from transformers import AutoTokenizer
gtok = AutoTokenizer.from_pretrained('gpt2')

fig, axes = plt.subplots(2, 3, figsize=(17, 11.5))
for row, hh in enumerate([3, 6]):
    # labels annotated with the head's branch-1 class id: "token·c17"
    cls = ASSIGN[(1, hh)][snippet].tolist()
    labels = [f"{gtok.decode([t]).replace(chr(10), '\\n')}·c{c}"
              for t, c in zip(snippet.tolist(), cls)]
    po, pc = pat_orig[hh].cpu().numpy(), pat_cb[hh].cpu().numpy()
    vmax = max(abs(po).max(), abs(pc).max())
    for col, (mat, title) in enumerate([(po, 'original'),
                                        (pc, 'vq256 CE-trained codebook'),
                                        (pc - po, 'difference')]):
        ax = axes[row, col]
        ax.imshow(mat, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
        ax.set_title(f'L0H{hh} — {title}', fontsize=11, color=INK)
        ax.set_xticks(range(48)); ax.set_yticks(range(48))
        ax.set_xticklabels(labels, fontsize=4.4, rotation=90, color=SECONDARY)
        ax.set_yticklabels(labels, fontsize=4.4, color=SECONDARY)
        ax.tick_params(length=0)
fig.suptitle('Attention patterns from the compressed model (entries = f(class, class, Δ))\n'
             'labels: token·class-id for that head/branch — same class ⇒ identical pre-rotary factors',
             fontsize=13, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('results/fig_pattern_display.png', dpi=170)
print('regenerated with class annotations')
