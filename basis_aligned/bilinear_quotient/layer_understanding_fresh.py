"""[FRESH per-component certification of §893] HOW MUCH OF EACH LAYER DO WE UNDERSTAND, out-of-sample? §893
built the token-mean stand-in tables and scored CE on the SAME rows (same-data). §901 showed the whole-model
version was heavily overfit (0.81 in-sample → 0.29 held-out). Certify the PER-COMPONENT numbers the same way:
build each component's token-table on 70% train rows, evaluate CE on the held-out 30%. Prediction: the FRONT
(genuinely token-determined, e.g. mlp0) generalizes (~0.9 fresh); the context-dependent MIDDLE generalizes
poorly (its token table was partly memorization). Shuffled-token null controls rank.

REGISTERED PREDICTIONS:
  (0) SANITY: full CE reproduces; shuffled-token null ~0 or negative on held-out;
  (a) FRONT GENERALIZES, MIDDLE DOESN'T: mlp0/mlp2 fresh understanding stays high (>0.7), the middle content
      MLPs drop vs their §893 same-data values -> only the genuinely token-determined (grammar) components are
      understood as generalizing token tables; the middle's same-data table was partly overfit;
  (b) report fresh understanding per component + the same-data §893 value for comparison."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'layer_understanding_fresh_results.json'
NEVAL = 200; SEQ = 256
COMPS = [(0, 'mlp'), (0, 'attn'), (1, 'mlp'), (2, 'mlp'), (5, 'attn'), (5, 'mlp'),
         (8, 'attn'), (8, 'mlp'), (11, 'attn'), (11, 'mlp'), (15, 'mlp'), (16, 'mlp'), (17, 'mlp')]
SAMEDATA = {'mlp0': 0.94, 'attn0': 0.77, 'mlp1': 0.98, 'mlp2': 0.87, 'attn5': 0.50, 'mlp5': 0.39,
            'attn8': 0.97, 'mlp8': 0.47, 'attn11': 0.53, 'mlp11': 0.48, 'mlp15': 0.59, 'mlp16': 0.78, 'mlp17': 0.70}
REPL = {'mode': 'off', 'table': None, 'gmean': None, 'tok': None, 'target': None}


def submod(L, kind): return getattr(m.transformer.h[L], kind)


def repl_hook_factory(tag):
    def hook(mo, i_, o_):
        if REPL['mode'] == 'off' or REPL['target'] != tag: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        yn = REPL['gmean'].expand(B, T, D).clone() if REPL['mode'] == 'mean' else REPL['table'][REPL['tok']]
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def ce_over(blockset, tokmapset, shuffle_map=None):
    tot = 0.0; n = 0
    for i in range(0, blockset.shape[0], 8):
        bb = blockset[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        tk = tokmapset[i:i+8][:, :-1]
        if shuffle_map is not None: tk = shuffle_map[tk]
        REPL['tok'] = torch.tensor(tk, device=DEV)
        lg = forward_logits(idx).float()
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel()
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    ntr = int(0.7*nb); TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True
    uniq = np.unique(S); remap = {int(t): j for j, t in enumerate(uniq)}; nu = len(uniq)
    tokmap = np.vectorize(lambda t: remap[int(t)])(S).astype(np.int64)
    tags = [f"{k}{L}" for (L, k) in COMPS]
    hooks = [submod(L, k).register_forward_hook(repl_hook_factory(f"{k}{L}")) for (L, k) in COMPS]
    # capture component outputs on TRAIN rows only, accumulate per-token tables
    sums = {t: torch.zeros(nu, D, device=DEV) for t in tags}; cnts = {t: torch.zeros(nu, device=DEV) for t in tags}
    cap = {}
    caph = []
    for (L, k) in COMPS:
        tag = f"{k}{L}"
        def mk(tag):
            def h(mo, i_, o_): cap[tag] = (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D)
            return h
        caph.append(submod(L, k).register_forward_hook(mk(tag)))
    REPL['mode'] = 'off'
    tr_blocks = blocks[TRAIN]; tr_tok = tokmap[TRAIN]
    for i in range(0, tr_blocks.shape[0], 8):
        bb = tr_blocks[i:i+8].to(DEV); forward_logits(bb[:, :-1].contiguous())
        tk = torch.tensor(tr_tok[i:i+8][:, :-1], device=DEV).reshape(-1)
        for tag in tags:
            o = cap[tag]; sums[tag].index_add_(0, tk, o); cnts[tag].index_add_(0, tk, torch.ones_like(tk, dtype=torch.float))
    for h in caph: h.remove()
    tables = {t: sums[t]/cnts[t].clamp_min(1).unsqueeze(1) for t in tags}
    gmeans = {t: (sums[t].sum(0)/cnts[t].sum()).contiguous() for t in tags}
    rng = np.random.RandomState(0); shuffle_map = rng.permutation(nu).astype(np.int64)
    te_blocks = blocks[~TRAIN]; te_tok = tokmap[~TRAIN]
    REPL['mode'] = 'off'; ce_full = ce_over(te_blocks, te_tok)
    out = {'ce_full': round(ce_full, 3), 'components': {}}
    for (L, k) in COMPS:
        tag = f"{k}{L}"; REPL['target'] = tag; REPL['table'] = tables[tag]; REPL['gmean'] = gmeans[tag]
        REPL['mode'] = 'mean'; ce_mean = ce_over(te_blocks, te_tok)
        REPL['mode'] = 'set'; ce_std = ce_over(te_blocks, te_tok)
        REPL['mode'] = 'set'; ce_shuf = ce_over(te_blocks, te_tok, shuffle_map=shuffle_map)
        REPL['mode'] = 'off'
        denom = max(ce_mean - ce_full, 1e-6); rec = (ce_mean - ce_std)/denom; rec_sh = (ce_mean - ce_shuf)/denom
        out['components'][tag] = {'fresh_understanding': round(float(rec), 3), 'shuffled_null': round(float(rec_sh), 3),
                                  'samedata_893': SAMEDATA.get(tag), 'drop_vs_samedata': round(SAMEDATA.get(tag, 0) - float(rec), 3)}
        print(f"{tag:>6}: fresh {rec:.2f} (null {rec_sh:+.2f}) | same-data §893 {SAMEDATA.get(tag)} | drop {SAMEDATA.get(tag,0)-rec:+.2f}", flush=True)
    for h in hooks: h.remove()
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\n(fresh per-component: table on 70% train, CE on held-out 30%; 0=mean-ablate,1=full)", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
