"""E8 SMALL-SCALE GAP-FILLING (Logan 2026-08-05; fresh single-epoch batch-16
protocol throughout, paired vs E0a/E0b and where stated vs the E7m1 combo).

PRIORITY 0 (probes only, no training): wiring-readability (weight-vs-causal
Spearman) + token-determined probes on the EXISTING E7prox and E7m1
checkpoints -- the lasso's justification is readable wiring (E0b reference at
AdamW gc 1e-4: Spearman 0.778 all / 0.578 effectual; E5 frontier at gc
0 / 1e-5 / 3e-5 / 1e-4: 0.07 / 0.42 / 0.62 / 0.78). Question: does
proximal-Muon preserve readability? Also records the E5slots token-determined
profile into qk_e5.json if absent.

Training arms, in priority order:
 1. E8prox3e5   slots base under proximal-Muon at coefficient 3e-5 (the
                frontier point that matters at scale) + wiring probe.
 2. E8combo3e5  per-slot norm + proximal-Muon at 3e-5 + wiring probe.
                (1+2 map the readability-vs-CE frontier under the winning
                optimizer; also paired vs E7m1 for the coefficient effect.)
 3. E8tokw      E4 typed token slot retried with a THREE-slot (33-dim) token
                line -- same design, wider bottleneck; token-determined
                comparison vs E0b (and vs the 11-dim E4).
 4. E8win6      the N=6 lookback window re-priced fresh (all section-111
                evidence is 6-epoch): windowed-vanilla (no slots, no lasso)
                and window+slots+lasso arms, BOTH with the nonzero write init
                (the windowed readout is blind to the embedding; zero-init
                writes start it at the uniform predictor). Wiring probe on the
                slotted arm. NOTE (deviation): implemented with the family
                V8Route harness's window visibility (C.window_vis), whose sum
                assembly is the same rule qk_window_train's WindowMini uses;
                the identity controls verify the reduction to vanilla-A.
 5. E8anneal    anneal-to-certified-zeros from the E7m1 combo checkpoint
                (median-group-norm threshold, 1000-step fine-tune with zeros
                frozen UNDER proximal-Muon) + wiring probe after.
 6. E8v14b      V14b (attention-only token line on the N=6 window) re-priced
                fresh at width 264: the embedding never enters any entry sum
                and is added ONLY into every block's attention read
                (h_att = rms_norm(x_entry + e_norm)); readout reads the
                last-6 writes only. Token-determined probe (does the
                mid-stack MLP relay dissolve?).

Identity positive controls before each new arm; loss curves into qk_e8.json;
idempotent; non-blocking guards (qk_e_common). Results -> qk_e8.json."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, R, V8T, C, W, DEPTH, F, torch
import qk_e1_slotnorm_run as E1R     # noqa: F401  (E1Route via E7R factory)
import qk_e3_anneal_run as E3R       # group_norms / masks / edges helpers
import qk_e4_tokenslot_run as E4R    # E4Route base for the wide token line
import qk_e7_evenout_run as E7R      # make_e7m1 factory + muon_lr

JP = E.jpath('qk_e8.json')
PROX3 = 3e-5
E5_FRONTIER_NOTE = ('E5 wiring frontier (AdamW, fresh): Spearman-all 0.07 at '
                    'gc 0 / 0.42 at 1e-5 / 0.62 at 3e-5 / 0.78 at 1e-4')


def pair_extra(stem, jp, key, others):
    if E.SMOKE:
        return
    for ctl, label in others:
        f_arm, f_ctl = f'{stem}_heldloss.npy', f'{ctl}_heldloss.npy'
        if os.path.exists(f'{E.QK}/{f_ctl}') and os.path.exists(f'{E.QK}/{f_arm}'):
            E.merge(jp, f'{key}_minus_{label}_fresh',
                    E.paired(f_arm, f_ctl, len(Q.HELD), label))


def muon_prox_trainer(prox_coeff):
    def tf(lr, gc, steps, **kw):
        assert gc == 0.0
        return E.train_muon(lr, 0.0, steps, lr_adamw=E.get_lr(),
                            prox_coeff=prox_coeff, **kw)
    return tf


# ---------------- P0: probes on existing checkpoints ----------------
def p0_probes():
    if E.SMOKE:
        return
    E.merge(JP, 'references', {
        'e0b_wiring': 'Spearman all 0.778 / effectual 0.578 (AdamW gc 1e-4)',
        'e5_frontier': E5_FRONTIER_NOTE})
    if os.path.exists(E.ckpath('qk_e7_prox')):
        E.probe_arm('qk_e7_prox', E.make_e0b, JP,
                    'light_probe_E7prox', tok_key='tok_probe_E7prox')
    if os.path.exists(E.ckpath('qk_e7_m1')):
        E.probe_arm('qk_e7_m1', E7R.make_e7m1, JP,
                    'light_probe_E7m1', tok_key='tok_probe_E7m1')
    # E5slots token-determined profile -> qk_e5.json (its factory is make_e0b;
    # the lasso existed only in the loss)
    e5p = E.jpath('qk_e5.json')
    if os.path.exists(E.ckpath('qk_e5_slots264')) \
            and 'tok_probe_E5slots' not in E.loadj(e5p):
        m, _ = E.load_arm('qk_e5_slots264', E.make_e0b)
        E.merge(e5p, 'tok_probe_E5slots', E.tok_probe(m))
        del m
        torch.cuda.empty_cache()


# ---------------- arm 3: three-slot (33-dim) token line ----------------
TOK_SLOTS = 3


def tokwide_slot_of(k):
    return min(TOK_SLOTS + k, 2 * DEPTH - 1)


class E8TokWide(E4R.E4Route):
    """E4Route with the token line widened to TOK_SLOTS slots (33 dims);
    modules 21/22/23 share the last slot with module 20 (264 dims do not
    split into 24 + 3 equal slots; the shared writes are blocks 10-11's,
    which feed mostly the readout)."""

    def __init__(self, variant, depth, tokline=True):
        V8T.V8Route.__init__(self, variant, depth)
        self.tokline = tokline
        if tokline:
            Dm = self.wte.weight.shape[1]
            S = Dm // (2 * depth)
            wm = torch.zeros(2 * depth, Dm)
            for k in range(2 * depth):
                s = tokwide_slot_of(k)
                wm[k, S * s:S * (s + 1)] = 1.0
            with torch.no_grad():
                self.wmask.copy_(wm)

    def embed_stream(self, idx):
        e_raw = self.wte(idx)
        Dm = e_raw.shape[-1]
        if not self.tokline:
            return F.rms_norm(e_raw, (Dm,))
        Sw = TOK_SLOTS * (Dm // (2 * self.depth))
        e = torch.zeros_like(e_raw)
        e[..., :Sw] = F.rms_norm(e_raw[..., :Sw], (Sw,))
        return e


def make_e8tokw(tokline=True):
    C.register('E8tokw')
    torch.manual_seed(Q.SEED)
    return E8TokWide('E8tokw', DEPTH, tokline=tokline).to(E.DEV)


@torch.no_grad()
def tokw_controls():
    idx = Q.HELD[:2, :Q.T]
    base = C.make_variant('E8tokwctl').eval().float()
    moff = make_e8tokw(tokline=False).eval().float()
    d = (moff(idx) - base(idx)).abs().max().item()
    print(f"control E8tokw(tokline off)==V8Route: max |logit diff| {d:.2e}",
          flush=True)
    assert d < 1e-4
    del moff
    m = make_e8tokw().eval().float()
    Sw = TOK_SLOTS * (Q.D // (2 * DEPTH))
    assert float(m.wmask[:, :Sw].abs().sum()) == 0.0
    col = {'entry': [], 'entry_norm': [], 'attn_write': [], 'mlp_write': []}
    m(idx, collect=col)
    tok = F.rms_norm(m.wte(idx)[..., :Sw].float(), (Sw,))
    dmax = max(float((en.float()[..., :Sw] - tok).abs().max())
               for en in col['entry'])
    print(f"control E8tokw entries carry the 33-dim token code: max diff "
          f"{dmax:.2e} over {len(col['entry'])} entries", flush=True)
    assert dmax < 1e-4
    del m, base, col
    torch.cuda.empty_cache()


# ---------------- arm 4: N=6 window, fresh ----------------
def win_vis():
    return [C.window_vis(li, N=6) for li in range(DEPTH + 1)]


def make_e8winv():
    R.KERNEL['E8winv'] = 'sum'
    R.WRITE_INIT.add('E8winv')               # blind readout: nonzero writes
    torch.manual_seed(Q.SEED)
    m = V8T.V8Route('E8winv', DEPTH).to(E.DEV)
    m.vis = win_vis()
    return m


def make_e8wins():
    return C.make_variant('E8wins', lambda li: C.window_vis(li, N=6))


@torch.no_grad()
def win_controls():
    v = win_vis()
    assert v[0] == [0] and 0 not in v[6] and v[12] == list(range(13, 25))
    print("window vis table: block0 emb only, emb gone from block 6 on, "
          "readout reads streams 13..24 (blocks 6-11)", flush=True)
    idx = Q.HELD[:2, :Q.T]
    Q.NL = DEPTH
    ma = Q.make_model('A').eval().float()
    ref = ma(idx)
    del ma
    torch.cuda.empty_cache()
    full = [list(range(1 + 2 * min(li, DEPTH))) for li in range(DEPTH + 1)]
    for mk, name in ((make_e8winv, 'E8winv'), (make_e8wins, 'E8wins')):
        m = mk().eval().float()
        m.vis = [list(fv) for fv in full]
        m.wmask.fill_(1.0)
        for blk in m.h:
            blk.c_proj.weight.zero_()
            blk.Down.weight.zero_()
        d = (m(idx) - ref).abs().max().item()
        print(f"control {name}(full vis, identity proj, zero writes)==A264: "
              f"max |logit diff| {d:.2e}", flush=True)
        assert d < 1e-3
        del m
        torch.cuda.empty_cache()
    del ref


# ---------------- arm 6: V14b attention-only token line, fresh ----------------
class V14bRoute(V8T.V8Route):
    """N=6 window where the normed embedding never enters any entry sum; it is
    added ONLY into every block's attention read: h_att = rms_norm(x + e).
    as_n6=True (control) drops the attention line so that, with the standard
    window visibility installed, the forward reduces to the plain V8Route
    window path."""
    as_n6 = False

    def forward(self, idx, collect=None, sub_entry=None, entry_override=None,
                mlp_sub=None, coef_out=None, attn_sub=None):
        B, Tq = idx.shape
        Dm = self.wte.weight.shape[1]
        NHm, HDm = Q.NH, Q.HD
        e = F.rms_norm(self.wte(idx), (Dm,))
        streams = [e]                        # read only where vis says (as_n6)
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]

        def entry(li):
            if entry_override is not None and li in entry_override:
                return entry_override[li]
            if not self.vis[li]:
                return torch.zeros(B, Tq, Dm, device=e.device, dtype=e.dtype)
            sub = sub_entry.get(li) if sub_entry is not None else None
            return self.assemble(li, streams, sub, coef_out)

        for l, blk in enumerate(self.h):
            x = entry(l)
            if collect is not None:
                collect['entry_norm'].append(
                    x.detach().float().norm(dim=-1).mean().item())
                if 'entry' in collect:
                    collect['entry'].append(x.detach())
            hn = F.rms_norm(x if self.as_n6 else x + e, (Dm,))

            def qk(lin):
                z = lin(hn).view(B, Tq, NHm, HDm)
                return Q.apply_rot(F.rms_norm(z, (HDm,)), cos, sin)

            q, k = qk(blk.c_q), qk(blk.c_k)
            q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
            v = blk.c_v(hn).view(B, Tq, NHm, HDm)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HDm
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HDm
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, Dm)
            aw = blk.c_proj(y)
            if self.proj:
                aw = aw * self.wmask[2 * l].to(aw.dtype)
            if attn_sub is not None and l in attn_sub:
                aw = attn_sub[l]
            x = x + aw
            if mlp_sub is not None and l in mlp_sub:
                mw = mlp_sub[l]
            else:
                xn = F.rms_norm(x, (Dm,))        # mlp: entry + attn write only
                mw = blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
                if self.proj:
                    mw = mw * self.wmask[2 * l + 1].to(mw.dtype)
            if collect is not None:
                collect['attn_write'].append(aw.detach())
                collect['mlp_write'].append(mw.detach())
            streams.append(aw)
            streams.append(mw)
        x = entry(self.depth)
        if collect is not None and 'entry' in collect:
            collect['entry'].append(x.detach())
        x = F.rms_norm(x, (Dm,))
        logits = x @ self.wte.weight.t()
        return 30 * torch.tanh(logits / 30)


def make_e8v14b():
    R.KERNEL['E8v14b'] = 'sum'
    R.WRITE_INIT.add('E8v14b')
    torch.manual_seed(Q.SEED)
    m = V14bRoute('E8v14b', DEPTH).to(E.DEV)
    m.vis = [[i for i in C.window_vis(li, N=6) if i != 0]
             for li in range(DEPTH + 1)]     # emb in NO entry sum (vis[0]=[])
    return m


@torch.no_grad()
def v14b_controls():
    m = make_e8v14b().eval().float()
    assert m.vis[0] == [] and all(0 not in v for v in m.vis)
    m.as_n6 = True
    m.vis = win_vis()                        # reroute emb per the N6 rule
    ref = make_e8winv().eval().float()
    idx = Q.HELD[:2, :Q.T]
    d = (m(idx) - ref(idx)).abs().max().item()
    print(f"control E8v14b(as_n6)==window-vanilla at init: max |logit diff| "
          f"{d:.2e}", flush=True)
    assert d < 1e-4
    del m, ref
    torch.cuda.empty_cache()


# ---------------- arm 5: anneal from the E7m1 combo under prox-Muon ----------------
def anneal_e7m1(mlr):
    out = E.loadj(JP)
    if 'E8anneal' in out and (E.SMOKE or os.path.exists(E.ckpath('qk_e8_anneal'))):
        print("E8anneal: already done -- skip", flush=True)
        return
    if E.SMOKE and not os.path.exists(E.ckpath('qk_e7_m1')):
        m = E7R.make_e7m1()
        torch.save({'state_dict': m.state_dict(), 'config': {}, 'log': {}},
                   E.ckpath('qk_e7_m1'))
        del m
    model, _ = E.load_arm('qk_e7_m1', E7R.make_e7m1)
    g = E3R.group_norms(model)
    thresh = float(g.flatten().median())
    zmask = E3R.build_masks(g, thresh)
    frac = float(zmask.float().mean())
    n100 = 2 if E.SMOKE else 100
    ce_before, _ = E.eval_ce(model, Q.HELD, n_seq=n100)
    E3R.apply_masks(model, zmask)
    ce_zeroed, _ = E.eval_ce(model, Q.HELD, n_seq=n100)
    print(f"E8anneal: {int(zmask.sum())}/{zmask.numel()} groups zeroed "
          f"(frac {frac:.4f}); held{n100} fresh CE {ce_before:.4f} -> "
          f"{ce_zeroed:.4f}", flush=True)
    # fine-tune under proximal-Muon with zeros frozen
    steps = 3 if E.SMOKE else 1000
    lr_m, lr_a = 0.1 * mlr, 0.1 * E.get_lr()
    warm = 1 if E.SMOKE else 50
    mu, dec, nod = E.muon_params_split(model)
    opt_m = E.Muon(mu, lr=lr_m)
    opt_a = torch.optim.AdamW([{'params': dec, 'weight_decay': Q.WD},
                               {'params': nod, 'weight_decay': 0.0}],
                              lr=lr_a, betas=(0.9, 0.95))
    order = Q.epoch_order(2)
    ftlog = {'lr_muon': lr_m, 'lr_adamw': lr_a, 'steps': steps,
             'prox_coeff': 1e-4, 'train_loss': [], 'held_ce': [], 'spikes': 0}
    model.train()
    run, t0 = None, time.time()
    for step in range(steps):
        f = min(1.0, (step + 1) / warm)
        for gpg in opt_m.param_groups:
            gpg['lr'] = lr_m * f
        for gpg in opt_a.param_groups:
            gpg['lr'] = lr_a * f
        seqs = Q.TRAIN[order[step * Q.BATCH:(step + 1) * Q.BATCH]]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(seqs[:, :Q.T])
        ce = F.cross_entropy(logits.float().reshape(-1, Q.V),
                             seqs[:, 1:Q.T + 1].reshape(-1))
        opt_m.zero_grad(set_to_none=True)
        opt_a.zero_grad(set_to_none=True)
        ce.backward()
        E3R.mask_grads(model, zmask)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Q.GRAD_CLIP)
        opt_m.step()
        opt_a.step()
        E.prox_group_lasso(model, lr_m * f * 1e-4)
        E3R.apply_masks(model, zmask)        # frozen zeros (prox keeps 0 at 0)
        l = ce.item()
        run = l if run is None else 0.98 * run + 0.02 * l
        if l > run + 1.0:
            ftlog['spikes'] += 1
        if step % (1 if E.SMOKE else 100) == 0:
            ftlog['train_loss'].append([step, round(l, 4), round(run, 4)])
            print(f"  E8anneal ft step {step}/{steps} ce {l:.4f} "
                  f"(ema {run:.4f}) {time.time() - t0:.0f}s", flush=True)
        if not E.SMOKE and step > 0 and step % 250 == 0:
            hce, _ = Q.eval_held(model, n_seq=100)
            ftlog['held_ce'].append([step, round(hce, 4)])
    model.eval()
    ce_ft, ptf = E.eval_ce(model, Q.HELD, per_token=True)
    leak = float(E3R.group_norms(model)[zmask].abs().max())
    assert leak == 0.0
    rec = {'base': 'qk_e7_m1 (per-slot norm + prox-Muon combo)',
           'n_groups': int(zmask.numel()), 'n_zeroed': int(zmask.sum()),
           'zero_fraction': round(frac, 4), 'threshold': thresh,
           f'held{n100}_fresh_ce_before_zero': round(ce_before, 5),
           f'held{n100}_fresh_ce_after_zero': round(ce_zeroed, 5),
           'held_fresh_ce_after_finetune_bf16': round(ce_ft, 5),
           'finetune': ftlog, 'max_zeroed_group_norm_after_ft': leak}
    E.merge(JP, 'E8anneal', rec)
    edges = E3R.surviving_edges(model, zmask)
    E.merge(JP, 'E8anneal_certified_edges', {
        'n_edges_surviving': len(edges), 'edges': edges[:80]})
    if not E.SMOKE:
        np.save(f'{E.QK}/qk_e8_anneal_heldloss.npy', ptf)
        torch.save({'state_dict': model.state_dict(), 'zmask': zmask,
                    'threshold': thresh, 'config': {}, 'log': ftlog},
                   E.ckpath('qk_e8_anneal'))
    del model, opt_m, opt_a
    torch.cuda.empty_cache()


if __name__ == '__main__':
    E.setup()
    mlr = E7R.muon_lr()
    p0_probes()

    # ---- 1+2: the frontier point 3e-5 under proximal-Muon ----
    for stem, key, factory in (('qk_e8_prox3e5', 'E8prox3e5', E.make_e0b),
                               ('qk_e8_combo3e5', 'E8combo3e5', E7R.make_e7m1)):
        E.train_arm(stem, JP, key, factory, 0.0, lr=mlr,
                    trainer=muon_prox_trainer(PROX3),
                    extra={'optimizer': 'muon', 'prox_coeff': PROX3,
                           'prox_rule': E7R.PROX_RULE,
                           'note': E5_FRONTIER_NOTE})
        E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
        E.paired_fresh(stem, JP, key)
        pair_extra(stem, JP, key, (('qk_e7_m1', 'e7m1'),))
        E.probe_arm(stem, factory, JP, f'light_probe_{key}')

    # ---- 3: wide token line ----
    tokw_controls()
    E.train_arm('qk_e8_tokw', JP, 'E8tokw', make_e8tokw, E.GC,
                extra={'token_slots': TOK_SLOTS, 'token_dims': 33})
    E.oldheld_record('qk_e8_tokw', make_e8tokw, JP, 'E8tokw_oldheld')
    E.paired_fresh('qk_e8_tokw', JP, 'E8tokw')
    E.probe_arm('qk_e8_tokw', make_e8tokw, JP, 'light_probe_E8tokw',
                tok_key='tok_probe_E8tokw', slot_of=tokwide_slot_of)
    if not E.SMOKE:
        e0 = E.loadj(E.jpath('qk_e0.json'))
        e4 = E.loadj(E.jpath('qk_e4.json'))
        e8 = E.loadj(JP)
        if 'tok_probe_E8tokw' in e8:
            E.merge(JP, 'token_determined_comparison', {
                'E0b': (e0.get('tok_probe_E0b') or {}).get('token_determined_mlp'),
                'E4_11dim': (e4.get('tok_probe_E4') or {}).get('token_determined_mlp'),
                'E8tokw_33dim': e8['tok_probe_E8tokw'].get('token_determined_mlp')})

    # ---- 4: window re-priced fresh ----
    win_controls()
    for stem, key, factory, gc in (
            ('qk_e8_winv', 'E8winv', make_e8winv, 0.0),
            ('qk_e8_wins', 'E8wins', make_e8wins, E.GC)):
        E.train_arm(stem, JP, key, factory, gc,
                    extra={'visibility': 'N=6 lookback window',
                           'write_init': 'nonzero (blind readout)'})
        E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
        E.paired_fresh(stem, JP, key)
    E.probe_arm('qk_e8_wins', make_e8wins, JP, 'light_probe_E8wins')

    # ---- 5: anneal the combo ----
    anneal_e7m1(mlr)
    if not E.SMOKE and os.path.exists(E.ckpath('qk_e8_anneal')):
        def load_anneal():
            return E7R.make_e7m1()
        E.oldheld_record('qk_e8_anneal', load_anneal, JP, 'E8anneal_oldheld')
        E.paired_fresh('qk_e8_anneal', JP, 'E8anneal')
        pair_extra('qk_e8_anneal', JP, 'E8anneal', (('qk_e7_m1', 'e7m1'),))
        E.probe_arm('qk_e8_anneal', load_anneal, JP, 'light_probe_E8anneal')

    # ---- 6: V14b fresh ----
    v14b_controls()
    E.train_arm('qk_e8_v14b', JP, 'E8v14b', make_e8v14b, 0.0,
                extra={'design': 'N=6 window, embedding only in the attention '
                                 'read (h_att = rms_norm(entry + e)); no '
                                 'entry-sum embedding anywhere'})
    E.oldheld_record('qk_e8_v14b', make_e8v14b, JP, 'E8v14b_oldheld')
    E.paired_fresh('qk_e8_v14b', JP, 'E8v14b')
    if not E.SMOKE and os.path.exists(E.ckpath('qk_e8_v14b')):
        out = E.loadj(JP)
        if 'tok_probe_E8v14b' not in out:
            m, _ = E.load_arm('qk_e8_v14b', make_e8v14b)
            E.merge(JP, 'tok_probe_E8v14b', E.tok_probe(m))
            del m
            torch.cuda.empty_cache()
    print('e8 gaps run done', flush=True)
