"""E7 EVENING-OUT ARMS (fresh batch-16 single-epoch protocol, full 8250 steps,
all paired vs the fresh E0a/E0b controls). Four arms targeting the specific
cost mechanisms the fresh runs exposed:

(a) E7prox -- the fix for the Muon x group-lasso interaction (+0.076): E0b
    architecture under Muon with the lasso REMOVED from the loss and applied
    as a DECOUPLED PROXIMAL soft-threshold after each optimizer step. Exact
    rule (documented in the JSON): after every Muon/AdamW step, each read-
    matrix slot column group g <- g * max(0, 1 - tau/||g||_F), where
    tau = (scheduled muon lr at that step, warmup+cosine included) x 1e-4 --
    the exact proximal operator of tau * sum ||g||_F for a Euclidean step; the
    per-matrix aspect-ratio scale is NOT folded into tau; the readout is
    excluded (tied embedding) exactly as in loss-lasso training. Positive
    control: with threshold 0 the prox path must reproduce a lasso-free Muon
    slots run bit-for-bit.
(b) E7lr -- per-arm lr sweeps for V11 and V13r1 ({0.002, 0.004, 0.008} x 400
    steps), then full runs at their own winners; tests whether the family lr
    (tuned on E0b) understates them. If an arm's winner equals the family lr
    the existing qk_er run IS that full run and is referenced, not retrained.
(c) E7decid -- V11 with the decoder lasso penalizing (DECODER MINUS IDENTITY)
    instead of the decoder itself: stops taxing the pass-through (trained V11
    decoder norms sit at median 3.8 vs ~16.25 for identity), directly testing
    the shrinkage-tax hypothesis. Positive controls: the identity-relative
    penalty is ~0 at init (vs ~79.6 for the plain V11 penalty) and the forward
    equals plain V11 at init. Note: AdamW weight decay 0.1 still pulls the
    decoders toward zero (kept verbatim from V11 for comparability).
(d) E7m1 -- current best-guess recipe combination: the E1 per-slot-RMSNorm
    base under Muon with the proximal lasso from (a).

Extra pairings beyond E0a/E0b: E7prox vs the loss-lasso Muon arm (E0b_muon)
and vs AdamW E0b; E7m1 vs E7prox (per-slot norm's marginal effect) and vs the
AdamW E1 arm. Results -> qk_e7.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_e_common as E
from qk_e_common import Q, V8T, C, W, DEPTH, torch
import qk_v13_common as X
import qk_e1_slotnorm_run as E1R

JP = E.jpath('qk_e7.json')
PROX_COEFF = 1e-4
PROX_RULE = ('after every optimizer step, each read-matrix slot column group '
             'g <- g * max(0, 1 - tau/||g||_F) with tau = (scheduled muon lr, '
             'warmup+cosine included) * 1e-4; exact proximal operator of '
             'tau * sum_groups ||g||_F; aspect-ratio scale not in tau; '
             'readout excluded as in loss-lasso training')


def muon_lr():
    d = E.loadj(E.jpath('qk_e0m.json'))
    if 'muon_lrsweep' in d:
        return d['muon_lrsweep']['chosen']
    print("WARNING: muon sweep result missing -- fallback 0.02", flush=True)
    return 0.02


def muon_prox_trainer(lr, gc, steps, **kw):
    assert gc == 0.0, "prox arms must not carry the loss lasso"
    return E.train_muon(lr, 0.0, steps, lr_adamw=E.get_lr(),
                        prox_coeff=PROX_COEFF, **kw)


# ---------------- (c) V11 with identity-relative decoder lasso ----------------
class V11IdLasso(W.V11Route):
    """V11Route whose decoder lasso penalizes ||A_k - Id_k||_F, Id_k = the
    slot-identity init matrix (identity on module k's slot columns)."""

    def __init__(self, variant, depth, dec_lasso=True):
        super().__init__(variant, depth, dec_lasso=dec_lasso)
        Dm = self.wte.weight.shape[1]
        S = W.SUB
        idm = torch.zeros(2 * depth, Dm, Dm)
        for k in range(2 * depth):
            for cc in range(S):
                idm[k, S * k + cc, S * k + cc] = 1.0
        self.register_buffer('dec_idmat', idm)

    def dec_penalty(self):
        tot = None
        for k, lin in enumerate(self.dec):
            g = ((lin.weight - self.dec_idmat[k]).pow(2).sum() + 1e-12).sqrt()
            tot = g if tot is None else tot + g
        return tot


def make_e7decid():
    return W.make_v11('E7decid', dec_lasso=True, cls=V11IdLasso)


# ---------------- (d) per-slot-norm base for the Muon+prox combination ----------------
def make_e7m1(groups=None):
    C.register('E7m1')
    torch.manual_seed(Q.SEED)
    m = E1R.E1Route('E7m1', DEPTH).to(E.DEV)
    m.norm_groups = E.NGROUP if groups is None else groups
    return m


@torch.no_grad()
def _statedict_maxdiff(m0, m1):
    return max(float((p0.float() - p1.float()).abs().max())
               for (n0, p0), (n1, p1)
               in zip(m0.state_dict().items(), m1.state_dict().items()))


def prox_sanity_50(mlr):
    """Known-answer test for the NONZERO-threshold proximal path (2026-08-04
    request). Measured on 2026-08-04 while debugging the deadlocked run: at 50
    steps prox-Muon and PLAIN Muon differ by 0.0001 nats (9.1487 vs 9.1486)
    while AdamW+loss-lasso sits at 7.7880 -- so the AdamW trajectory is the
    WRONG 50-step yardstick (pure Muon warmup dynamics, which invert by step
    400: the sweeps gave Muon 6.20 vs AdamW 6.24). The control therefore
    asserts against the matched-optimizer reference: prox-Muon must track
    PLAIN lasso-free Muon within 0.05 nats at 50 steps (tau ~ 2e-6 is a tiny
    shrink), must zero essentially no read groups, and the AdamW gap is
    reported informationally."""
    steps = 50
    logp, mp = E.train_muon(mlr, 0.0, steps, log_every=10, save=False,
                            factory=E.make_e0b, lr_adamw=E.get_lr(),
                            prox_coeff=PROX_COEFF, return_model=True)
    gn = E.read_group_norms(mp)
    frac0 = float((gn < 1e-8).float().mean())
    gmin = float(gn.min())
    del mp
    torch.cuda.empty_cache()
    logm = E.train_muon(mlr, 0.0, steps, log_every=10, save=False,
                        factory=E.make_e0b, lr_adamw=E.get_lr(),
                        prox_coeff=None)
    loga = V8T.train_v8(E.get_lr(), E.GC, steps, log_every=10, save=False,
                        factory=E.make_e0b)
    cep, cem, cea = (logp['final_held_ce'], logm['final_held_ce'],
                     loga['final_held_ce'])
    print(f"control prox 50-step: prox-muon {cep:.4f} vs plain muon "
          f"{cem:.4f} (|diff| {abs(cep - cem):.4f}; must be < 0.05) vs "
          f"adamw+loss-lasso {cea:.4f} (info only: muon warmup gap "
          f"{cep - cea:+.4f}); read groups zeroed {frac0:.4f}, min group "
          f"norm {gmin:.4f}", flush=True)
    assert abs(cep - cem) < 0.05, "prox path diverges from plain Muon"
    assert frac0 < 0.01, "prox is zeroing read groups wholesale"


def controls():
    mlr = muon_lr()
    if not E.SMOKE:
        prox_sanity_50(mlr)
    # (a) prox with threshold 0 == plain lasso-free Muon run (3 steps)
    log0, m_prox = E.train_muon(mlr, 0.0, 3, log_every=1, save=False,
                                factory=E.make_e0b, lr_adamw=E.get_lr(),
                                prox_coeff=0.0, return_model=True)
    log1, m_plain = E.train_muon(mlr, 0.0, 3, log_every=1, save=False,
                                 factory=E.make_e0b, lr_adamw=E.get_lr(),
                                 prox_coeff=None, return_model=True)
    d = _statedict_maxdiff(m_prox, m_plain)
    print(f"control E7prox(tau=0)==plain lasso-free Muon after 3 steps: "
          f"max |weight diff| {d:.2e}", flush=True)
    assert d < 1e-6
    del m_prox, m_plain
    torch.cuda.empty_cache()
    # (c) identity-relative penalty ~0 at init; forward == plain V11 at init
    with torch.no_grad():
        mv = W.make_v11('E7decidctl', dec_lasso=True,
                        cls=V11IdLasso).eval().float()
        p0 = float(mv.dec_penalty())
        plain = float(W.V11Route.dec_penalty(mv))
        print(f"control E7decid penalty at init {p0:.2e} "
              f"(plain V11 penalty would be {plain:.2f})", flush=True)
        assert p0 < 1e-3 and plain > 1.0
        idx = Q.HELD[:2, :Q.T]
        ref = W.make_v11('V11', dec_lasso=True,
                         cls=W.V11Route).eval().float()
        dd = (mv(idx) - ref(idx)).abs().max().item()
        print(f"control E7decid(init)==V11(init): max |logit diff| {dd:.2e}",
              flush=True)
        assert dd < 1e-4
        del mv, ref
        torch.cuda.empty_cache()
    # (d) E7m1 with one norm group == plain V8Route at init
    with torch.no_grad():
        base = C.make_variant('E7m1ctl').eval().float()
        m1 = make_e7m1(groups=1).eval().float()
        idx = Q.HELD[:2, :Q.T]
        dd = (m1(idx) - base(idx)).abs().max().item()
        print(f"control E7m1(1 group)==V8Route at init: max |logit diff| "
              f"{dd:.2e}", flush=True)
        assert dd < 1e-4
        del base, m1
        torch.cuda.empty_cache()


if __name__ == '__main__':
    E.setup()
    controls()
    mlr = muon_lr()

    # ---- (a) E7prox ----
    E.train_arm('qk_e7_prox', JP, 'E7prox', E.make_e0b, 0.0, lr=mlr,
                trainer=muon_prox_trainer,
                extra={'optimizer': 'muon', 'prox_coeff': PROX_COEFF,
                       'prox_rule': PROX_RULE,
                       'lr_adamw_for_embedding': E.get_lr()})
    E.oldheld_record('qk_e7_prox', E.make_e0b, JP, 'E7prox_oldheld')
    E.paired_fresh('qk_e7_prox', JP, 'E7prox')
    if not E.SMOKE:
        for ctl, label in (('qk_e0b_muon264', 'e0b_muon'),):
            f_arm, f_ctl = 'qk_e7_prox_heldloss.npy', f'{ctl}_heldloss.npy'
            if os.path.exists(f'{E.QK}/{f_ctl}') \
                    and os.path.exists(f'{E.QK}/{f_arm}'):
                E.merge(JP, f'E7prox_minus_{label}_fresh',
                        E.paired(f_arm, f_ctl, len(Q.HELD), label))

    # ---- (b) E7lr: per-arm sweeps, full runs at own winners ----
    for name, factory, er_stem in (
            ('v11', lambda: W.make_v11('V11', dec_lasso=True,
                                       cls=W.V11Route), 'qk_er_v11'),
            ('v13r1', lambda: X.make_v13('V13r1', 1), 'qk_er_v13r1')):
        chosen = E.lr_sweep(JP, f'{name}_lrsweep', (0.002, 0.004, 0.008),
                            lambda lr, steps, f=factory: V8T.train_v8(
                                lr, E.GC, steps, log_every=100, save=False,
                                factory=f))
        key = f'E7lr_{name}'
        if abs(chosen - E.get_lr()) < 1e-12 and \
                os.path.exists(E.ckpath(er_stem)):
            E.merge(JP, key, {'chosen_lr': chosen,
                              'note': f'winner equals the family lr; the full '
                                      f'run at this lr is {er_stem} '
                                      f'(referenced, not retrained)'})
        else:
            stem = f'qk_e7_{name}lr'
            E.train_arm(stem, JP, key, factory, E.GC, lr=chosen,
                        extra={'own_lr_winner': chosen,
                               'family_lr': E.get_lr()})
            E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
            E.paired_fresh(stem, JP, key)

    # ---- (c) E7decid ----
    E.train_arm('qk_e7_decid', JP, 'E7decid', make_e7decid, E.GC,
                extra={'decoder_penalty': 'group lasso on (decoder - slot '
                                          'identity); read lasso unchanged; '
                                          'weight decay 0.1 still on decoders '
                                          '(V11 convention kept)'})
    E.oldheld_record('qk_e7_decid', make_e7decid, JP, 'E7decid_oldheld')
    E.paired_fresh('qk_e7_decid', JP, 'E7decid')
    if not E.SMOKE and os.path.exists(E.ckpath('qk_e7_decid')):
        m, _ = E.load_arm('qk_e7_decid', make_e7decid)
        dn = [round(float(lin.weight.detach().float().norm()), 3)
              for lin in m.dec]
        E.merge(JP, 'E7decid_decoder_norms', {
            'per_module': dn,
            'median': round(float(torch.tensor(dn).median()), 3),
            'identity_norm': round(float(m.dec_idmat[0].norm()), 3)})
        del m
        torch.cuda.empty_cache()

    # ---- (d) E7m1 ----
    E.train_arm('qk_e7_m1', JP, 'E7m1', make_e7m1, 0.0, lr=mlr,
                trainer=muon_prox_trainer,
                extra={'recipe': 'per-slot RMSNorm (E1) + Muon + proximal '
                                 'lasso (a)', 'optimizer': 'muon',
                       'prox_coeff': PROX_COEFF, 'prox_rule': PROX_RULE})
    E.oldheld_record('qk_e7_m1', make_e7m1, JP, 'E7m1_oldheld')
    E.paired_fresh('qk_e7_m1', JP, 'E7m1')
    if not E.SMOKE:
        for ctl, label in (('qk_e7_prox', 'e7prox'),
                           ('qk_e1_slotnorm', 'e1_adamw')):
            f_arm, f_ctl = 'qk_e7_m1_heldloss.npy', f'{ctl}_heldloss.npy'
            if os.path.exists(f'{E.QK}/{f_ctl}') \
                    and os.path.exists(f'{E.QK}/{f_arm}'):
                E.merge(JP, f'E7m1_minus_{label}_fresh',
                        E.paired(f_arm, f_ctl, len(Q.HELD), label))
    print('e7 evenout run done', flush=True)
