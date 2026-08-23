"""CLEAN mechanism-independence test (fixes §1027's design confound). §1027 measured the content effect across
conditions whose LAST token differed (the induction manipulation puts A at the query position), contaminating it.
FIX: measure EACH mechanism's effect as a WITHIN-condition Δ (add the trigger vs not, with the other mechanism's
tokens AND the last token held fixed), then compare the effect WITH vs WITHOUT the other mechanism present. The
last-token change cancels inside each Δ.

Per trial (query 150; content word W@3; induction bigram A@5,B@6 with A@149):
  content-effect(bg)   = logP(W-neighbors | bg + W@3) - logP(W-neighbors | bg)
  induction-effect(bg) = logP(B | bg + AB@5-6 + A@149) - logP(B | bg + A@149)
  retention_content   = content-effect(induction-bg) / content-effect(plain-bg)
  retention_induction = induction-effect(content-bg)  / induction-effect(plain-bg)

REGISTERED PREDICTIONS:
  (0) SANITY: both effects are positive in the plain background (content ~0.5, induction ~7-8).
  (a) INDEPENDENCE (clean): retention_content AND retention_induction are both in ~[0.7, 1.3] -> each mechanism's
      effect is unchanged by the presence of the other -> content and induction compose without interference;
  (b) report plain vs cross-background effects and retentions for both."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mechanism_independence_v2_results.json'
NEVAL = 200; SEQ = 256; QUERY = 150; W_POS = 3; PA = 5; PB = 6; NNEIGH = 20; NTRIALS = 16
CONTENT_WORDS = [' football', ' hospital', ' ocean', ' music', ' science', ' army', ' church', ' garden']


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def neighbors(wid, k):
    Wm = m.lm_head.weight.float(); wv = Wm[wid] / (Wm[wid].norm() + 1e-9)
    sims = (Wm / (Wm.norm(dim=1, keepdim=True) + 1e-9)) @ wv; sims[wid] = -1e9
    return torch.topk(sims, k).indices


@torch.no_grad()
def mean_logp(blocks, edits, targets, exclude_wids):
    # edits: dict pos->token to set; return mean logP over `targets` at the query (excluding rows containing exclude_wids)
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); idx = bb[:, :QUERY].contiguous().clone()
        for p, t in edits.items(): idx[:, p] = t
        lp = F.log_softmax(forward_logits(idx).float()[:, -1], -1)
        has = torch.zeros(idx.shape[0], dtype=torch.bool, device=DEV)
        for w in exclude_wids: has |= (bb[:, :QUERY] == w).any(1)
        tot += float(lp[:, targets].mean(1)[~has].sum()) if targets.numel() > 1 else float(lp[:, targets[0]][~has].sum())
        n += int((~has).sum())
    return tot / max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    uniq = np.unique(rows.cpu().numpy().reshape(-1)); rng = np.random.RandomState(0)
    pool = [int(t) for t in uniq if 100 < int(t) < 50000]; rng.shuffle(pool)
    cwords = [tid(w) for w in CONTENT_WORDS if tid(w) is not None]
    ceff_plain = []; ceff_ind = []; ieff_plain = []; ieff_cont = []
    it = iter(pool)
    for k in range(NTRIALS):
        W = cwords[k % len(cwords)]; nW = neighbors(W, NNEIGH)
        try: A = next(it); B = next(it)
        except StopIteration: break
        Bt = torch.tensor([B], device=DEV); excl = [W, A, B]
        # content effect in plain bg: [+W] - [plain]
        ceff_plain.append(mean_logp(rows, {W_POS: W}, nW, excl) - mean_logp(rows, {}, nW, excl))
        # content effect in induction bg: [AB,Alast,+W] - [AB,Alast]
        indbg = {PA: A, PB: B, QUERY-1: A}
        ceff_ind.append(mean_logp(rows, {**indbg, W_POS: W}, nW, excl) - mean_logp(rows, indbg, nW, excl))
        # induction effect in plain bg: [AB,Alast] - [Alast]
        ieff_plain.append(mean_logp(rows, {PA: A, PB: B, QUERY-1: A}, Bt, excl) - mean_logp(rows, {QUERY-1: A}, Bt, excl))
        # induction effect in content bg: [W,AB,Alast] - [W,Alast]
        ieff_cont.append(mean_logp(rows, {W_POS: W, PA: A, PB: B, QUERY-1: A}, Bt, excl) - mean_logp(rows, {W_POS: W, QUERY-1: A}, Bt, excl))
    cp = float(np.mean(ceff_plain)); ci = float(np.mean(ceff_ind)); ip = float(np.mean(ieff_plain)); ic = float(np.mean(ieff_cont))
    out = {'content_effect_plain': round(cp, 4), 'content_effect_with_induction': round(ci, 4),
           'induction_effect_plain': round(ip, 4), 'induction_effect_with_content': round(ic, 4),
           'retention_content': round(ci/max(cp, 1e-6), 3), 'retention_induction': round(ic/max(ip, 1e-6), 3)}
    out['pred_0_sane'] = bool(cp > 0.1 and ip > 2)
    out['pred_a_independent'] = bool(0.7 < out['retention_content'] < 1.3 and 0.7 < out['retention_induction'] < 1.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content-effect: plain {cp:.3f} with-induction {ci:.3f} (retention {out['retention_content']})", flush=True)
    print(f"induction-effect: plain {ip:.3f} with-content {ic:.3f} (retention {out['retention_induction']})", flush=True)
    print(f"pred_0 sane {out['pred_0_sane']} | pred_a independent {out['pred_a_independent']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
