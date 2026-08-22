"""WHAT ARE THE ~24 CLASSES the model computes? (interpretive payoff). The whole program has
characterized class+position mechanically (subspace, keep-only, steering) but never ENUMERATED
what the classes are. Take the class subspace at mlp0 (SVD of token-conditional-mean outputs),
and for each top class direction show the tokens that load most positively and negatively —
naming what each class dimension encodes, in plain language, with real tokens.

REGISTERED PREDICTIONS:
  (0) SANITY: a SHUFFLED-token-label class subspace gives INCOHERENT token lists (control);
  (a) the real top class directions correspond to NAMEABLE categories (e.g. punctuation,
      digits, function words, capitalized/proper, whitespace/newline, common suffixes) — report
      the top +/- tokens per direction and a human name;
  (b) report the effective number of class directions (participation ratio of the singular
      values) — is it ~24 as the front-block eff-dim suggested?"""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'class_naming_results.json'
NEVAL = 400; MINCOUNT = 8; NDIR = 15; NTOK = 12
LAYER = 0


def dec():
    try:
        import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])
    except Exception:
        from transformers import GPT2TokenizerFast; t = GPT2TokenizerFast.from_pretrained('gpt2'); return lambda i: t.decode([int(i)])


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows):
    cap = []; toks = []
    mod = m.transformer.h[LAYER].mlp
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = mod.register_forward_hook(h)
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        toks.append(idx.cpu().numpy().reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks)


def class_dirs(O, labels, g):
    uniq = []; rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        uniq.append(int(t)); rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    Mdev = torch.stack(rows, 0)                                  # (Ntok, D) per-token mean deviation
    W = Mdev * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)          # Vh rows = class directions
    return np.array(uniq), Mdev, Vh, S


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    O, toks = capture(rows); g = O.mean(0, keepdim=True)
    d = dec()
    uniq, Mdev, Vh, S = class_dirs(O, toks, g)
    eff = float((S.sum()**2)/(S**2).sum())
    proj = Mdev @ Vh.T                                           # (Ntok, K) token loadings on each dir
    out = {'eff_num_class_dirs': round(eff, 1), 'n_tokens': len(uniq), 'directions': []}
    for k in range(min(NDIR, Vh.shape[0])):
        col = proj[:, k].cpu().numpy()
        pos = uniq[np.argsort(-col)[:NTOK]]; neg = uniq[np.argsort(col)[:NTOK]]
        out['directions'].append({'k': k, 'sv': round(float(S[k]), 2),
                                  'top_pos': [repr(d(t)) for t in pos], 'top_neg': [repr(d(t)) for t in neg]})
        print(f"dir {k} (sv {float(S[k]):.1f}): +[{' '.join(repr(d(t)) for t in pos[:8])}]  -[{' '.join(repr(d(t)) for t in neg[:6])}]", flush=True)
    # control: shuffled token labels -> class structure should collapse (eff-num near n_tokens / incoherent)
    rng = np.random.RandomState(0); sh = toks.copy(); rng.shuffle(sh)
    u2, Md2, Vh2, S2 = class_dirs(O, sh, g)
    out['eff_num_class_dirs_shuffled_control'] = round(float((S2.sum()**2)/(S2**2).sum()), 1)
    proj2 = Md2 @ Vh2.T; col2 = proj2[:, 0].cpu().numpy()
    out['shuffled_dir0_top_pos'] = [repr(d(t)) for t in u2[np.argsort(-col2)[:NTOK]]]
    out['runtime_s'] = time.time()-t0
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\neff # class directions {eff:.1f} (over {len(uniq)} tokens)", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']:.0f}s)")


if __name__ == '__main__':
    main()
