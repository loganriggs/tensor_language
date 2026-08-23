"""MATCHED-last-token control (settles §1028's residual 2x on the content side). §1028's content retention 1.98 was
still confounded because the plain background kept the ORIGINAL last token while the induction background had A@149.
Here decompose that 2x cleanly by measuring the content effect (W topic-neighbor boost) in THREE backgrounds and
comparing pairs that differ in only ONE thing:
  ce_plain  = [{}+W] - [{}]                       (original last token)
  ce_Aonly  = [{A@149}+W] - [{A@149}]             (A at last, NO induction bigram)
  ce_ind    = [{AB@5-6,A@149}+W] - [{AB@5-6,A@149}] (A at last, WITH induction bigram)
  retention_lasttoken = ce_Aonly / ce_plain   -> isolates last-token IDENTITY (A vs original), NOT a mechanism effect
  retention_trigger   = ce_ind  / ce_Aonly    -> isolates the induction TRIGGER (AB bigram) with last-token FIXED

REGISTERED PREDICTIONS:
  (0) SANITY: all three content effects > 0.
  (a) CONTENT INDEPENDENT OF THE INDUCTION TRIGGER: retention_trigger ~1 (in [0.7,1.3]) -> adding the AB induction
      bigram, with the last token held fixed at A, does NOT change the content mechanism -> content is independent of
      induction (the §1028 2x was last-token IDENTITY, not a mechanism interaction);
  (b) the last-token-identity effect (retention_lasttoken) is what accounts for §1028's ~2x (a content-ish last token
      A amplifies the content channel). Report all three effects + both retentions."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mechanism_independence_v3_results.json'
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
    ceff_plain = []; ceff_Aonly = []; ceff_ind = []
    it = iter(pool)
    for k in range(NTRIALS):
        W = cwords[k % len(cwords)]; nW = neighbors(W, NNEIGH)
        try: A = next(it); B = next(it)
        except StopIteration: break
        excl = [W, A, B]
        # content effect in three backgrounds (matched-last-token control)
        ceff_plain.append(mean_logp(rows, {W_POS: W}, nW, excl) - mean_logp(rows, {}, nW, excl))
        Abg = {QUERY-1: A}
        ceff_Aonly.append(mean_logp(rows, {**Abg, W_POS: W}, nW, excl) - mean_logp(rows, Abg, nW, excl))
        indbg = {PA: A, PB: B, QUERY-1: A}
        ceff_ind.append(mean_logp(rows, {**indbg, W_POS: W}, nW, excl) - mean_logp(rows, indbg, nW, excl))
    cp = float(np.mean(ceff_plain)); ca = float(np.mean(ceff_Aonly)); ci = float(np.mean(ceff_ind))
    out = {'ce_plain': round(cp, 4), 'ce_Aonly': round(ca, 4), 'ce_induction': round(ci, 4),
           'retention_lasttoken_identity': round(ca/max(cp, 1e-6), 3), 'retention_induction_trigger': round(ci/max(ca, 1e-6), 3)}
    out['pred_0_sane'] = bool(cp > 0.05 and ca > 0.05 and ci > 0.05)
    out['pred_a_content_indep_of_trigger'] = bool(0.7 < out['retention_induction_trigger'] < 1.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"content-effect: plain(orig-last) {cp:.3f} | A-last {ca:.3f} | A-last+AB {ci:.3f}", flush=True)
    print(f"retention last-token-identity (A/orig) {out['retention_lasttoken_identity']} | induction-trigger (AB/A, matched) {out['retention_induction_trigger']}", flush=True)
    print(f"pred_0 sane {out['pred_0_sane']} | pred_a content-indep-of-trigger {out['pred_a_content_indep_of_trigger']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
