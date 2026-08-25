# mlp4_bow: THE TWENTY-BAGS-OF-WORDS TEST (user directive 2026-08-25; thread: mid-MLP
# decomposition). Stand-in for mlp4's OUTPUT (everything else live, per-module-in-
# isolation per benchmark rules): out ≈ tok_table[token] + topic_delta[topic] +
# resid[token, topic] (residual term only for the 2000 most frequent tokens with
# count >= 5). TOPIC = K=20 k-means over bag-of-words features of the PRECEDING 64
# tokens (counts over the 2000 most frequent vocab items, L2-normalized) — deploy-legal,
# lexical, human-readable ("court data", "sports", ...). Fit on skip=80 rows; k-means
# fit on the same; EVAL HELD OUT on skip=7000 rows (benchmark rule 1). Null: same
# construction with topic labels shuffled (same marginal sizes) — kills cell-count
# inflation (S1327 lesson: the null must be label-shaped).
#
# Registered predictions:
#   pred_a topic-conditioning is REAL: tok+topic beats tok-only by >= .02 CE overall
#          on held-out rows.
#   pred_b the gain is semantic, not capacity: real gain >= 3x the shuffled-null gain.
#   pred_c tok+topic recovers >= .40 of mlp4's stake (live - meanablate gap), held out.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp4_bow_results.json'
NFIT = 960; NEV = 960
K = 20; NFREQ = 2000; WIN = 64
H = m.transformer.h
STAND = {'mode': None, 'tensor': None}


def mlp4_hook(mod, args, output):
    if STAND['mode'] is None:
        return None
    return STAND['tensor'].to(output.dtype)


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    FITR = cl.fineweb_rows(NFIT, skip=80)[:, :T + 1].contiguous()
    EVR = cl.fineweb_rows(NEV, skip=7000)[:, :T + 1].contiguous()

    # frequent-token vocabulary from fit rows
    flat = FITR[:, :-1].flatten()
    binc = torch.bincount(flat, minlength=50257)
    freq_ids = binc.topk(NFREQ).indices
    fmap = torch.full((50257,), -1, dtype=torch.long)
    fmap[freq_ids] = torch.arange(NFREQ)

    def bow_feats(rows):
        """[B, T, NFREQ-normalized] too big — return [B*T, NFREQ] in chunks is heavy;
        instead compute compressed: for kmeans use random projection to 256 dims."""
        pass

    # random projection matrix for BoW (fixed seed): NFREQ -> 128
    g = torch.Generator().manual_seed(7)
    PROJ = torch.randn(NFREQ, 128, generator=g) / (NFREQ ** 0.5)

    def bow_proj(rows):
        """Projected BoW of preceding WIN tokens for every position: [B, T, 128]."""
        toks = rows[:, :-1]
        B = toks.shape[0]
        fi = fmap[toks]                                   # [B, T] in [-1, NFREQ)
        onehot_proj = torch.zeros(B, T, 128)
        # cumulative windowed sum of PROJ rows
        pr = torch.where(fi.unsqueeze(-1) >= 0,
                         PROJ[fi.clamp_min(0)], torch.zeros(1, 1, 128))
        cs = pr.cumsum(1)
        win = cs - torch.cat([torch.zeros(B, WIN, 128), cs[:, :-WIN]], 1)
        return F.normalize(win, dim=-1)

    # capture mlp4 outputs on fit rows
    caps = []
    hk = H[4].mlp.register_forward_hook(
        lambda mod, a, o: caps.append(o.detach().float().cpu()))
    STAND['mode'] = None
    for i in range(0, NFIT, 8):
        fwd(FITR[i:i + 8, :-1].to(DEV).contiguous())
    hk.remove()
    OUTS = torch.cat(caps)                                # [NFIT, T, D]
    print(f"captured {OUTS.shape}", flush=True)

    FB = bow_proj(FITR)                                   # [NFIT, T, 128]
    X = FB[:, WIN:].reshape(-1, 128)                      # skip cold-start window
    # minibatch k-means K=20
    g2 = torch.Generator().manual_seed(11)
    cent = X[torch.randperm(X.shape[0], generator=g2)[:K]].clone().to(DEV)
    Xg = X.to(DEV)
    for it in range(25):
        d2 = torch.cdist(Xg, cent)
        lab = d2.argmin(1)
        for k in range(K):
            sel = lab == k
            if int(sel.sum()) > 0:
                cent[k] = Xg[sel].mean(0)
    print("kmeans done; cluster sizes:",
          torch.bincount(lab, minlength=K).tolist(), flush=True)

    def topics_for(rows):
        FBx = bow_proj(rows).to(DEV)
        return torch.cdist(FBx.reshape(-1, 128), cent).argmin(1).view(rows.shape[0], T).cpu()

    TOPF = topics_for(FITR)
    gshuf = torch.Generator().manual_seed(23)
    TOPF_SH = TOPF.flatten()[torch.randperm(TOPF.numel(), generator=gshuf)].view(TOPF.shape)

    toksF = FITR[:, :-1]
    mask_fit = torch.ones_like(toksF, dtype=torch.bool); mask_fit[:, :WIN] = False

    def fit_tables(topics):
        tsum = torch.zeros(50257, D); tcnt = torch.zeros(50257)
        flat_t = toksF[mask_fit]; flat_o = OUTS[mask_fit]
        tsum.index_add_(0, flat_t, flat_o); tcnt.index_add_(0, flat_t, torch.ones(flat_t.shape[0]))
        gmean = flat_o.mean(0)
        tok_table = torch.where(tcnt.unsqueeze(1) > 0, tsum / tcnt.clamp_min(1).unsqueeze(1),
                                gmean.unsqueeze(0))
        resid = flat_o - tok_table[flat_t]
        flat_top = topics[mask_fit]
        dsum = torch.zeros(K, D); dcnt = torch.zeros(K)
        dsum.index_add_(0, flat_top, resid); dcnt.index_add_(0, flat_top, torch.ones(flat_top.shape[0]))
        topic_delta = dsum / dcnt.clamp_min(1).unsqueeze(1)
        fi = fmap[flat_t]
        selF = fi >= 0
        pair = fi[selF] * K + flat_top[selF]
        r2 = resid[selF] - topic_delta[flat_top[selF]]
        psum = torch.zeros(NFREQ * K, D); pcnt = torch.zeros(NFREQ * K)
        psum.index_add_(0, pair, r2); pcnt.index_add_(0, pair, torch.ones(pair.shape[0]))
        pres = torch.where(pcnt.unsqueeze(1) >= 5, psum / pcnt.clamp_min(1).unsqueeze(1),
                           torch.zeros(1, D))
        return tok_table, topic_delta, pres, gmean

    tok_table, topic_delta, pres, gmean = fit_tables(TOPF)
    _, topic_delta_sh, pres_sh, _ = fit_tables(TOPF_SH)
    print("tables fit", flush=True)

    TOPE = topics_for(EVR)
    TOPE_SH = TOPE.flatten()[torch.randperm(TOPE.numel(), generator=gshuf)].view(TOPE.shape)

    hook = H[4].mlp.register_forward_hook(mlp4_hook)

    def ce_run(mode, topics=None, tdelta=None, presid=None):
        s_ = 0.0; n_ = 0
        for i in range(0, NEV, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            if mode is None:
                STAND['mode'] = None
            else:
                tk = idx.cpu()
                st = tok_table[tk].clone() if mode != 'mean' else gmean.expand(tk.shape[0], T, D).clone()
                if mode in ('toktopic',):
                    tp = topics[i:i + 8]
                    st += tdelta[tp]
                    fi = fmap[tk]
                    sel = fi >= 0
                    pairix = (fi.clamp_min(0) * K + tp)
                    add = presid[pairix]
                    st[sel] += add[sel]
                STAND['mode'] = 'on'
                STAND['tensor'] = st.to(DEV)
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            m_ = torch.ones_like(tg, dtype=torch.bool); m_[:, :WIN] = False
            s_ += float(ce[m_].sum()); n_ += int(m_.sum())
        STAND['mode'] = None
        return s_ / max(n_, 1)

    res = {}
    res['live'] = ce_run(None)
    res['mean'] = ce_run('mean')
    res['tok'] = ce_run('tok')
    res['toktopic'] = ce_run('toktopic', TOPE, topic_delta, pres)
    res['tokshuf'] = ce_run('toktopic', TOPE_SH, topic_delta_sh, pres_sh)
    hook.remove()
    for k2, v in res.items():
        print(f"{k2}: {v:.4f}", flush=True)

    stake = res['mean'] - res['live']
    gain = res['tok'] - res['toktopic']
    gain_null = res['tok'] - res['tokshuf']
    recov = (res['mean'] - res['toktopic']) / max(stake, 1e-6)
    pa = gain >= 0.02
    pb = gain >= 3.0 * max(gain_null, 1e-4)
    pc = recov >= 0.40
    out = {'ce': {k2: round(v, 4) for k2, v in res.items()},
           'stake': round(stake, 4), 'topic_gain': round(gain, 4),
           'null_gain': round(gain_null, 4), 'toktopic_recovery': round(recov, 4),
           'params': {'tok_table': '50257xD', 'topic_delta': f'{K}xD',
                      'pair_resid': f'{NFREQ}x{K}xD (count>=5 cells only)'},
           'pred_a_topic_real': bool(pa), 'pred_b_semantic': bool(pb),
           'pred_c_recovery_40': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"stake {stake:.4f} | topic gain {gain:.4f} vs null {gain_null:.4f} | rec {recov:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
