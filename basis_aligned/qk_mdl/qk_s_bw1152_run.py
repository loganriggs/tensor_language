"""THE BRANCH POINT: bandwidth reinvestment + the readability dial at w1152.

Local's frontier-best w264 arm is E19a (qk_e19.json): the E15c bandwidth-
reinvestment architecture -- TRUE-SMALL decoders (c_proj Dc->s, Down
hidden->s, scattered into the module's own slot) with the ~27% of body
params that the recipe's hard write masks waste spent instead on SLOT WIDTH
-- trained with the in-loss group-lasso raised to 1e-4. At w264 it reads
4.9742 fresh held at covariance-composed wiring Spearman 0.8259, i.e. it
beats the readable recipe (E9a 5.0547 @ 0.8575) by -0.0804 nats at a
readability tie within 0.03.

This runner ports that construction to w1152 and asks the question the
program's decision tree hangs on (BRAINSTORM_STATE.md "Scale in-flight"):

  HOLDS (CE better than the recipe at readability-tie-or-better)
      -> bandwidth reinvestment is THE retrain recipe core.
  FLIPS (like all four w264 structural wins did at w1152: shared values
      +0.0711, param-matched sv +0.0550, shrinking channel +0.0633,
      funnel-sv +0.0715, funnel +0.1097)
      -> w264 loses structural standing entirely and the program pivots
      its weight to post-training (crystallization / distillation), which
      does not depend on width transfer.

CONSTRUCTION (exactly make_e15c's, with the slot dim SOLVED from live param
counts rather than transcribed). Compute width stays Dc = 1152 (18 heads x
64) and the bilinear hidden width stays 4*Dc = 4608; the residual stream
widens to Ws = 24*s where s is the largest slot dim whose body param count
fits vanilla-1152's body. Per slot dim the body costs
    12 * [24*(5*Dc + 2*hidden) + Dc + hidden + 1]  =  12 * 365,185
so s = round(286,668,288 / 4,382,220) = 65 and the stream widens 1152 ->
1560 (+35.4% message bandwidth) at essentially unchanged body params. The
tied embedding widens with the stream (VRAM cost, excluded from body
accounting exactly as at w264).

ARMS (argv[1]); both are the same architecture, only the dial differs, and
each pairs against the w1152 recipe arm trained at ITS OWN coefficient so
the comparison isolates the architecture and never the dial:

  bw1e4  group-lasso 1e-4 -- the literal port of E19a's dial.
         Matched control: combo1e4loss (4.22360 scale / 4.27241 f34k).
  bw3e5  group-lasso 3e-5 -- the DIAL-RESCALED port, and simultaneously the
         direct port of E19a's 3e-5 parent E15c (4.9038 @ w264).
         Matched control: combo3e5loss (4.10596 scale / 4.15500 f34k),
         which is also the readable recipe at scale.

  Why both: this program has already MEASURED that the readability point
  tracks coefficient ~ 1/width (qk_s_w1152_gate.json 'sparsity_analysis';
  MAILBOX 2026-08-05 15:45 UTC) -- gc3e5 at w1152 matches gc1e4 at w264 in
  relative read-mass shrinkage (/5.8 vs /8.1), in wiring Spearman (0.76 vs
  0.78), and in the coefficient ratio itself (1e-4 * 264/1152 = 2.3e-5).
  So the literal 1e-4 port bites about TWICE as hard at w1152 as E19a's
  dial did at w264, and a FLIP measured only at 1e-4 would be confounded
  with over-penalization. bw3e5 is the readability-equivalent transfer.

Everything else is the scale session's standard: corpus_fresh shards
00..06 rows [0:298496], single pass, epoch_order(0) -- the identical data
order every other w1152 arm used -- effective batch 32 with micro
accumulation chosen by preflight, Muon 0.02 on the 2D hidden matrices with
the tied embedding and sub-2D params on AdamW at the gate's swept lr,
cosine schedule, 30*tanh(logits/30), bf16 autocast train and eval, held =
shard06's last 1500 rows plus fresh34k rows [33000:34500].

WRITE INIT (documented deviation from the w264 parent): E15cRoute hardcodes
its decoder init std at 0.02/sqrt(2*depth), which was the w264 convention.
Every w1152 arm in this session instead uses the width-rescaled
R.WRITE_INIT_STD = 0.02/sqrt(2*depth)/sqrt(width/384). This runner uses the
SCALE convention so the arm is comparable to combo3e5loss/combo1e4loss, and
control (c) proves the re-draw reproduces the class's own init bit-exactly
when handed the class's own std -- i.e. the only change is the scale factor.

POSITIVE CONTROLS, all before training:
  (a) IDENTITY REDUCTION: the bandwidth architecture at s = 48 (so
      Ws = 24*48 = 1152 = the compute width) with its small decoders
      loaded from the slot rows of an E1-recipe model's masked full
      decoders must reproduce that recipe model's forward at init. tf32
      is disabled around BOTH forwards -- an asymmetric comparison
      measures tf32 rounding, not the architecture (that mistake cost
      this program the original E15 launch: 6.07e-4 vs a 1e-4 threshold).
  (b) PENALTY vs NAIVE: the seg_ind-vectorized custom_group_penalty
      against an explicit per-group Frobenius loop, rel < 1e-6, plus the
      V8T dispatch identity (the trainer calls V8T.group_penalty, so the
      dispatch is what actually has to be right).
  (c) WRITE-INIT re-draw fidelity (above), bit-exact.
  (d) PARAM ACCOUNTING: body params within 1% of vanilla-1152's
      286,668,288, and nominal == effective by construction (true-small
      decoders waste no rows), printed and stored.

Outputs qk_s_w1152_bw1e4.{json,pt} / qk_s_w1152_bw3e5.{json,pt} plus
_heldloss.npy / _f34kloss.npy. Idempotent: re-running skips completed
controls and returns immediately if the checkpoint and 'run' key exist.
"""
import os
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import math

import qk_tokenline_train as Q

Q.gpu_guard = lambda *a, **k: None

import qk_s_gate_run as G                  # WIDTH 1152, data + eval + json io
import qk_w1152_train as W2                # patch_width
import qk_deeproute_train as R             # R.WRITE_INIT_STD (scale convention)
import qk_v8_train as V8T                  # group_penalty (dispatch target)
import qk_e_common as E
from qk_e_common import DEPTH, torch
import qk_e1_slotnorm_run as E1R           # make_e1 = the recipe factory
import qk_e15_reinvest_run as E15R         # E15cRoute + copy_from_ref (+dispatch)
import qk_s_muon_run as M

# MEMORY (measured on this box, 2026-08-06): the stock preflight ladder
# accepted micro 16 at a measured peak of 27,360 MiB against the session's
# 29,000 MiB budget, then the real training loop OOMed at step 0 -- the
# steady-state loop peaks roughly 2.5 GiB above what the 2-step preflight
# sees (Muon orthogonalization temporaries + grad clipping on top of the
# fp32 logits, which alone are micro*512*50257*4 bytes). This arm is also
# genuinely heavier than the recipe arms (24,773 MiB): its stream is 1560
# wide, so every read matrix and the tied embedding grow by 35%. Budget
# lowered so the ladder drops to micro 8; effective batch stays 32 by
# accumulation, whose numerical equivalence this session already verified
# (rel 3.1e-5).
G.MEM_BUDGET_MIB = 26000

ARM = sys.argv[1] if len(sys.argv) > 1 else 'bw1e4'
CFG = {'bw1e4': dict(stem='qk_s_w1152_bw1e4', coeff=1e-4,
                     control='qk_s_w1152_combo1e4loss',
                     control_label='combo1e4loss', seed_offset=0),
       'bw3e5': dict(stem='qk_s_w1152_bw3e5', coeff=3e-5,
                     control='qk_s_w1152_combo3e5loss',
                     control_label='combo3e5loss', seed_offset=0),
       # SEED REPLICATE of the winning arm (reviewer-2 R4: nothing enters the
       # retrain recommendation single-seed). Init seed +1, data order and
       # everything else identical -- so the bw3e5-vs-bw3e5_s1 spread is the
       # init lottery on THIS architecture, to be read against the measured
       # w1152 vanilla seed noise floor of 0.0127.
       'bw3e5_s1': dict(stem='qk_s_w1152_bw3e5_s1', coeff=3e-5,
                        control='qk_s_w1152_combo3e5loss',
                        control_label='combo3e5loss', seed_offset=1)}[ARM]
STEM = CFG['stem']
COEFF = CFG['coeff']
SEED_OFFSET = CFG['seed_offset']
JP = os.path.join(G.OUT_DIR, f'{STEM}.json')

VANILLA_BODY_W1152 = 286_668_288           # asserted against a live count


# ---------------- construction ----------------
def solve_slot():
    """Slot dim s at fixed compute/hidden width such that the body param
    count matches vanilla-1152's. Same algebra as E15R.solve_slot_c, with
    the target verified against a live vanilla build rather than trusted."""
    D, hidden = Q.D, 4 * Q.D
    per_s = 24 * (5 * D + 2 * hidden) + D + hidden + 1
    s = max(1, int(round(VANILLA_BODY_W1152 / (DEPTH * per_s))))
    return s, per_s


def write_std():
    """The scale session's width-rescaled decoder init std."""
    return R.WRITE_INIT_STD


def make_bw(s=None, std=None):
    """E15cRoute at the patched scale width, decoders re-drawn at `std`.

    The re-draw mirrors the class's own init loop exactly -- same generator
    seed 888, same parameter order -- so passing the class's own std
    reproduces its init bit-exactly (control c)."""
    hidden = 4 * Q.D
    if s is None:
        s = SLOT
    torch.manual_seed(Q.SEED)
    m = E15R.E15cRoute(f'BW1152_{ARM}', DEPTH, s, Q.D, Q.NH, Q.HD,
                       hidden).to(E.DEV)
    if std is not None:
        gw = torch.Generator().manual_seed(888)
        with torch.no_grad():
            for blk in m.h:
                for p in (blk.c_proj.weight, blk.Down.weight):
                    p.copy_(torch.randn(p.shape, generator=gw) * std)
    return m


def factory():
    return make_bw(std=write_std())


def body_params(m):
    tot = sum(p.numel() for p in m.parameters())
    return tot - m.wte.weight.numel()


# ---------------- positive controls ----------------
def controls():
    out = G.loadj(JP)
    if out.get('controls', {}).get('all_passed'):
        print("controls: already passed -- skip", flush=True)
        return out['controls']
    rec = {}
    S = G.WIDTH // 24                      # 48: the recipe's slot dim
    idx = Q.HELD[:2, :Q.T]

    # (c) write-init re-draw fidelity: class default std reproduced exactly
    cls_std = 0.02 / math.sqrt(2 * DEPTH)
    m_plain = make_bw(s=S, std=None)
    m_redraw = make_bw(s=S, std=cls_std)
    d_init = max(
        (a - b).abs().max().item()
        for blk_a, blk_b in zip(m_plain.h, m_redraw.h)
        for a, b in ((blk_a.c_proj.weight, blk_b.c_proj.weight),
                     (blk_a.Down.weight, blk_b.Down.weight)))
    print(f"control (c) write-init re-draw at the class std: max |diff| "
          f"{d_init:.2e} (expect exactly 0.0)", flush=True)
    assert d_init == 0.0, d_init
    rec['write_init_redraw_max_diff'] = d_init
    rec['write_init_std_class'] = cls_std
    rec['write_init_std_scale'] = write_std()
    del m_plain, m_redraw
    torch.cuda.empty_cache()

    # (a) identity reduction at s = 48 to the recipe model
    ref = E1R.make_e1().eval().float()
    assert ref.h[0].Left.weight.shape[0] == 4 * Q.D, ref.h[0].Left.weight.shape
    ref_body = body_params(ref)
    print(f"recipe (E1) body params {ref_body} "
          f"({'MATCH' if ref_body == VANILLA_BODY_W1152 else 'MISMATCH'} "
          f"vanilla-1152 {VANILLA_BODY_W1152})", flush=True)
    assert ref_body == VANILLA_BODY_W1152, ref_body
    mc = make_bw(s=S, std=None).eval().float()
    E15R.copy_from_ref(mc, ref)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            out_ref = ref(idx)
            d_id = (mc(idx) - out_ref).abs().max().item()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    print(f"control (a) bandwidth(s={S}, weights copied from the recipe) == "
          f"recipe at init: max |logit diff| {d_id:.2e} (threshold 1e-4; "
          f"algebraic identity, GEMM reduction order may differ)", flush=True)
    assert d_id < 1e-4, d_id
    rec['identity_reduction_s48_max_logit_diff'] = d_id
    rec['identity_reduction_threshold'] = 1e-4
    del ref, out_ref
    torch.cuda.empty_cache()

    # (b) penalty fast vs naive, on the s=48 reduction, then dispatch identity
    with torch.no_grad():
        p_fast = float(mc.custom_group_penalty())
        p_naive = 0.0
        for blk in mc.h:
            for nm in E.READ_NAMES:
                W = getattr(blk, nm).weight
                for k in range(2 * DEPTH):
                    p_naive += float(
                        W[:, S * k:S * (k + 1)].pow(2).sum()) ** 0.5
        rel = abs(p_fast - p_naive) / p_naive
        p_disp = float(V8T.group_penalty(mc))
    print(f"control (b) penalty fast {p_fast:.4f} vs naive {p_naive:.4f} "
          f"rel {rel:.2e}; V8T dispatch {p_disp:.4f}", flush=True)
    assert rel < 1e-6, rel
    assert abs(p_disp - p_fast) < 1e-6 * max(1.0, abs(p_fast)), (p_disp, p_fast)
    rec['penalty_fast'] = p_fast
    rec['penalty_naive'] = p_naive
    rec['penalty_rel_diff'] = rel
    rec['penalty_dispatch_matches'] = True
    del mc
    torch.cuda.empty_cache()

    # (d) param accounting at the real slot dim
    m = make_bw(std=write_std()).eval()
    body = body_params(m)
    emb = m.wte.weight.numel()
    print(f"control (d) bandwidth arm: slot {SLOT}, stream {m.Ws} "
          f"(compute {Q.D}, hidden {m.hidden}), body {body} "
          f"({body / 1e6:.2f}M) vs vanilla-1152 "
          f"{VANILLA_BODY_W1152 / 1e6:.2f}M -> "
          f"{100 * (body / VANILLA_BODY_W1152 - 1):+.2f}%; embedding {emb} "
          f"({emb / 1e6:.1f}M, excluded from body)", flush=True)
    assert abs(body / VANILLA_BODY_W1152 - 1) < 0.01, body
    rec.update({'slot': SLOT, 'stream_width': m.Ws, 'compute_width': Q.D,
                'hidden': m.hidden, 'body_params': body,
                'body_params_effective': body,
                'body_vs_vanilla_frac': body / VANILLA_BODY_W1152,
                'vanilla_body': VANILLA_BODY_W1152, 'embedding_params': emb,
                'bandwidth_gain_vs_recipe': m.Ws / G.WIDTH,
                'note': 'true-small decoders: nominal body == effective body '
                        'by construction (no masked-away rows)'})
    del m
    torch.cuda.empty_cache()

    rec['all_passed'] = True
    out['controls'] = rec
    G.savej(JP, out)
    return rec


# ---------------- registered predictions ----------------
def register_predictions():
    out = G.loadj(JP)
    if 'registered_predictions' in out:
        return
    out['registered_predictions'] = {
        'registered': 'before training, this file written first',
        'w264_basis': {
            'E19a_bandwidth_1e4': 4.9742, 'E19a_cov_spearman': 0.8259,
            'E15c_bandwidth_3e5': 4.9038, 'E15c_cov_spearman': 0.6728,
            'recipe_E9a': 5.0547, 'recipe_cov_spearman': 0.8575,
            'E19a_minus_recipe': -0.0804},
        'w1152_controls': {
            'combo3e5loss_scale': 4.105955, 'combo3e5loss_f34k': 4.155002,
            'combo1e4loss_scale': 4.223603, 'combo1e4loss_f34k': 4.272409,
            'muonvanilla_scale': 3.964506, 'muonvanilla_f34k': 4.011635},
        'primary': (
            'HOLDS if this arm beats its OWN-COEFFICIENT recipe control on '
            'scale held CE (bw1e4 < 4.1060 for combo3e5loss is the strong '
            'form; bw1e4 < 4.2236 vs combo1e4loss is the coefficient-matched '
            'form). FLIPS if it lands above its matched control by more than '
            '0.02, which is beyond the measured w1152 seed noise floor '
            '(0.0127) plus the paired seq-clustered SE (~0.0012).'),
        'point_estimate_if_w264_transfers': (
            'full transfer of E19a-minus-recipe (-0.0804) would put bw1e4 at '
            'about 4.026 scale held; partial transfer anywhere in '
            '(4.026, 4.106) still HOLDS the strong form'),
        'readability': (
            'covariance-composed wiring Spearman >= the recipe control on the '
            'SAME probe is the readability-tie condition; the retrain rule '
            '(BRAINSTORM_STATE) requires CE better at readability-tie-or-'
            'better, at both widths, over 2-3 seeds'),
        'dial_rescaling_caveat': (
            'the measured rule is that the readability point tracks '
            'coefficient ~ 1/width (gc3e5@1152 == gc1e4@264 in relative read-'
            'mass shrinkage 5.8 vs 8.1 and Spearman 0.76 vs 0.78), so the '
            'literal 1e-4 port bites about twice as hard at w1152 as E19a '
            'did at w264 -- bw3e5 is the readability-equivalent transfer and '
            'a FLIP seen ONLY at 1e-4 must not be read as an architecture '
            'verdict'),
        'prior': (
            'all four w264 structural wins flipped sign at w1152 (shared '
            'values +0.0711, param-matched sv +0.0550, shrinking channel '
            '+0.0633, funnel-sv +0.0715, funnel +0.1097), so the prior '
            'favours a FLIP; the mechanism that would make this one differ '
            'is that it adds message bandwidth rather than constraining it, '
            'and the census showed saturation EASES at 48-dim slots')}
    G.savej(JP, out)
    print("registered predictions written to the JSON before training",
          flush=True)


# ---------------- pairing ----------------
def pair_results():
    import numpy as np
    out = G.loadj(JP)
    pairs = {}
    for ctl, label in ((CFG['control'], CFG['control_label']),
                       ('qk_s_w1152_combo3e5loss', 'combo3e5loss'),
                       ('qk_s_w1152_combo1e4loss', 'combo1e4loss'),
                       ('qk_s_w1152_muonvanilla', 'muonvanilla')):
        for which, n in (('heldloss', G.HELD_N), ('f34kloss', G.HELD_N)):
            fa = os.path.join(G.OUT_DIR, f'{STEM}_{which}.npy')
            fb = os.path.join(G.OUT_DIR, f'{ctl}_{which}.npy')
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            la, lb = np.load(fa), np.load(fb)
            if la.shape != lb.shape:
                continue
            dd = la - lb
            ds = dd.reshape(n, -1).mean(1)
            pairs[f'{which}_minus_{label}'] = {
                'arm_ce': float(la.mean()), 'control_ce': float(lb.mean()),
                'delta': float(dd.mean()),
                'se_token': float(dd.std(ddof=1) / math.sqrt(len(dd))),
                'se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds))),
                'n_seq': int(n)}
    out['paired'] = pairs
    G.savej(JP, out)
    for k, v in pairs.items():
        print(f"  {k}: {v['delta']:+.5f} +/- {v['se_seq']:.5f} (seq SE)",
              flush=True)
    return pairs


# ---------------- wire into the scale trainer ----------------
SLOT, PER_S = None, None


def main():
    global SLOT, PER_S
    W2.patch_width(G.WIDTH)
    if SEED_OFFSET:
        # every factory reseeds from Q.SEED, so bumping it here is the whole
        # of the replicate; the data order (epoch_order(0), driven by
        # Q.DATA_SEED) is deliberately left untouched
        Q.SEED = Q.SEED + SEED_OFFSET
        print(f"SEED REPLICATE: init seed -> {Q.SEED} "
              f"(data order unchanged, epoch_order(0))", flush=True)
    SLOT, PER_S = solve_slot()
    print(f"solved slot dim {SLOT} at compute width {Q.D}, hidden {4 * Q.D}: "
          f"body per slot dim {PER_S} x depth {DEPTH} = {DEPTH * PER_S}; "
          f"stream widens {G.WIDTH} -> {24 * SLOT}", flush=True)

    # the scale trainer reads these module globals
    M.ARM = ARM
    M.CFG = dict(stem=STEM, coeff=COEFF, prox=None, sweep=False)
    M.COEFF = COEFF
    M.PROX = None
    M.STEM = STEM
    M.JP = JP
    M.factory = factory

    # NOTE: deliberately NOT calling G.setup_data() here -- M.main() calls it,
    # and calling it twice transiently doubles the 1.22 GiB GPU-resident train
    # tensor right before preflight. The controls only need a 2-sequence
    # forward, for which the imported Q.HELD (the documented cooc substitute
    # on this box) is sufficient; nothing in them depends on the held corpus.
    controls()
    register_predictions()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    out = G.loadj(JP)
    out['arm_description'] = {
        'name': f'bandwidth_reinvestment_{ARM}',
        'architecture': 'E15c/E19a bandwidth reinvestment at w1152: '
                        'true-small decoders + per-slot RMSNorm + Muon, '
                        f'{24 * SLOT}-dim stream of 24 x {SLOT}-dim slots, '
                        f'compute width {Q.D}, hidden {4 * Q.D}',
        'group_coeff': COEFF, 'matched_control': CFG['control_label'],
        'write_init_std': write_std(),
        'write_init_note': 'scale convention (width-rescaled), NOT the w264 '
                           'parent hardcoded 0.02/sqrt(24); control (c) shows '
                           'the re-draw is otherwise bit-exact',
        'seed': Q.SEED, 'seed_offset': SEED_OFFSET,
        'data_seed': Q.DATA_SEED, 'data_order': 'epoch_order(0)'}
    G.savej(JP, out)

    M.main()
    pair_results()


if __name__ == '__main__':
    main()
