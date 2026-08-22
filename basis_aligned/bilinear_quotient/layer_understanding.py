"""HOW MUCH OF EACH LAYER DO WE UNDERSTAND? (user's metric). For each component, replace its output with a
STAND-IN built from our named understanding — the token-conditional-MEAN table (each current token → its mean
output; this captures "MLP0 collapses tokens into class clusters" exactly, since MLP0 is a function of the
token) — and score CE on a scale where MEAN-ABLATION = 0 (know nothing: replace output with the global mean)
and the FULL model = 1 (know everything). Understanding(component) = (CE_meanablate − CE_standin) /
(CE_meanablate − CE_full). A component that is a pure token-function (front grammar) → ~1; a context-dependent
component (content/topic attention) → low, and the GAP is the part we understand as gist/induction but not as
a static table. Control for the §836 rank artifact: a SHUFFLED-token table (same construction, token→output
map destroyed) — genuine understanding = standin recovery ABOVE the shuffled null.

Also report a richer stand-in for the front: token-mean is the ceiling of context-free understanding; adding
PREV-token conditioning tests induction-style context. (Kept token-only here for a clean first pass.)

REGISTERED PREDICTIONS:
  (0) SANITY: MLP0 understanding ~1 (it is a function of the current token); shuffled-token null ~0;
  (a) UNDERSTANDING DECREASES WITH CONTEXT-DEPENDENCE: front components (mlp0/attn0/mlp1) high token-table
      recovery (they are token-determined — the grammar machine we understand as class/token tables), MIDDLE
      content attention (attn5/8/11) LOW (context-dependent — the gist we cannot write as a token table),
      back MLPs intermediate; every recovery ABOVE its shuffled-token null;
  (b) report the per-component understanding fraction + null; the token-table gap at context-heavy layers is
      the quantified "content we understand mechanistically (gist) but not as a lookup table"."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'layer_understanding_results.json'
NEVAL = 140; SEQ = 256
COMPS = [(0, 'mlp'), (0, 'attn'), (1, 'mlp'), (2, 'mlp'), (5, 'attn'), (5, 'mlp'),
         (8, 'attn'), (8, 'mlp'), (11, 'attn'), (11, 'mlp'), (15, 'mlp'), (16, 'mlp'), (17, 'mlp')]
REPL = {'mode': 'off', 'table': None, 'gmean': None, 'tok': None, 'target': None}


def submod(L, kind): return getattr(m.transformer.h[L], kind)


def repl_hook_factory(tag):
    def hook(mo, i_, o_):
        if REPL['mode'] == 'off' or REPL['target'] != tag: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        if REPL['mode'] == 'mean':
            yn = REPL['gmean'].expand(B, T, D).clone()
        else:  # standin or shuffle: table lookup by (possibly shuffled) token index
            yn = REPL['table'][REPL['tok']]                 # (B, T, D)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def ce_on(blocks, tokmap, shuffle_map=None):
    """CE over the eval set under current REPL settings; sets REPL['tok'] per batch."""
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        tk = tokmap[i:i+8][:, :-1]
        if shuffle_map is not None: tk = shuffle_map[tk]        # remap token->random token (destroy map)
        REPL['tok'] = torch.tensor(tk, device=DEV)
        lg = forward_logits(idx).float()
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum'))
        n += tgt.numel()
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    # compact token remap over tokens present
    uniq = np.unique(S); remap = {int(t): j for j, t in enumerate(uniq)}; nu = len(uniq)
    tokmap = np.vectorize(lambda t: remap[int(t)])(S).astype(np.int64)      # (nb, SEQ) -> compact idx
    # register hooks on all target components
    hooks = []; tags = [f"{k}{L}" for (L, k) in COMPS]
    for (L, k) in COMPS: hooks.append(submod(L, k).register_forward_hook(repl_hook_factory(f"{k}{L}")))
    # capture each component's output, accumulate per-token sum/count (one clean pass)
    sums = {t: torch.zeros(nu, D, device=DEV) for t in tags}; cnts = {t: torch.zeros(nu, device=DEV) for t in tags}
    cap = {}
    caphooks = []
    for (L, k) in COMPS:
        tag = f"{k}{L}"
        def mk(tag):
            def h(mo, i_, o_): cap[tag] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
            return h
        caphooks.append(submod(L, k).register_forward_hook(mk(tag)))
    REPL['mode'] = 'off'
    for i in range(0, nb, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); forward_logits(idx)
        tk = torch.tensor(tokmap[i:i+8][:, :-1], device=DEV).reshape(-1)
        for tag in tags:
            o = cap[tag].reshape(-1, D)
            sums[tag].index_add_(0, tk, o); cnts[tag].index_add_(0, tk, torch.ones_like(tk, dtype=torch.float))
    for h in caphooks: h.remove()
    tables = {t: sums[t] / cnts[t].clamp_min(1).unsqueeze(1) for t in tags}
    gmeans = {t: (sums[t].sum(0) / cnts[t].sum()).contiguous() for t in tags}
    rng = np.random.RandomState(0); shuffle_map = rng.permutation(nu).astype(np.int64)
    # full-model CE (no replacement)
    REPL['mode'] = 'off'; ce_full = ce_on(blocks, tokmap)
    out = {'ce_full': round(ce_full, 3), 'n_unique_tokens': int(nu), 'components': {}}
    for (L, k) in COMPS:
        tag = f"{k}{L}"; REPL['target'] = tag; REPL['table'] = tables[tag]; REPL['gmean'] = gmeans[tag]
        REPL['mode'] = 'mean'; ce_mean = ce_on(blocks, tokmap)
        REPL['mode'] = 'standin'; ce_std = ce_on(blocks, tokmap)
        REPL['mode'] = 'shuffle'; ce_shuf = ce_on(blocks, tokmap, shuffle_map=shuffle_map)
        REPL['mode'] = 'off'
        denom = max(ce_mean - ce_full, 1e-6)
        rec = (ce_mean - ce_std) / denom; rec_sh = (ce_mean - ce_shuf) / denom
        out['components'][tag] = {'ce_meanablate': round(ce_mean, 3), 'ce_standin': round(ce_std, 3),
                                  'ce_shuffled': round(ce_shuf, 3), 'understanding_frac': round(float(rec), 3),
                                  'shuffled_null_frac': round(float(rec_sh), 3),
                                  'genuine': round(float(rec - rec_sh), 3)}
        print(f"{tag:>6}: understand {rec:.2f} (null {rec_sh:.2f}, genuine {rec-rec_sh:+.2f}) | CE full {ce_full:.2f} standin {ce_std:.2f} meanabl {ce_mean:.2f}", flush=True)
    for h in hooks: h.remove()
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\n(scale: 0 = mean-ablate/know-nothing, 1 = full/know-everything; token-mean-table stand-in)", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
