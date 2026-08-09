"""THE ATTENTION x FEED-FORWARD FACTORIAL.

The conventional baseline (tfb_std*) showed that a softmax+GELU transformer
crosses the induction floor one octave of width earlier than our foldable
family at every depth >= 2.  It cannot say WHY, because it changes two things
at once.  This file changes them one at a time.

    attention                          feed-forward
    ---------                          ------------
    bilin      (q1.k1/hd)*(q2.k2/hd)   bilin   Down(Left x * Right x) + b
               masked to 0, NO norm.           ungated, exactly foldable
               OURS, exactly foldable
    bilinnorm  the same product, then  gelu    Down(GELU(Up x)) + b
               divided by the row L1.          exact erf GELU
               DIAGNOSTIC ONLY -- see below
    softmax    softmax(q.k/sqrt hd),
               one branch.  CONVENTIONAL

(bilin, bilin) IS the family's vanilla model and (softmax, gelu) IS the
conventional baseline, so two of the six cells are already on disk.  The value
is in the two hybrids -- and the code earns the right to be read for them by
reproducing BOTH known corners from a state-dict transplant (gates G1, G2).

bilinnorm IS NOT A PROPOSED ARCHITECTURE.  Its denominator depends on every
visible key, so it does not fold to a fixed token-pair table and can never be
reported as a foldable win.  It is here to ask one question: is the softmax
effect about COMPETITION between keys, or about the EXPONENTIAL?  L1 rather
than plain row-sum normalisation because the two-branch product is signed and
a row sum can pass through zero; the L1 denominator is bounded away from it
and keeps the signs, so the arm stays a normalisation test and does not
quietly become a positivity test as well.

PARAMETER MATCHING.  Body cost per block is
    bilin/bilinnorm attention  6 W^2      softmax attention  4 W^2
    bilinear feed-forward      3 H W + W  GELU feed-forward  2 H W + W
so H is set per arm to hold the body at the family's 18 W^2 + W.  Softmax
frees 2 W^2, which does not divide by 3, so two arms land a few hundred
parameters off exactly-matched; the residual is recorded per arm rather than
rounded into an integer expansion factor and forgotten.

Usage
  python tf_factorial.py controls
  python tf_factorial.py cell --attn softmax --mlp bilin --depth 2 --width 128
  python tf_factorial.py report
"""
import argparse
import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import tf_model as M
import tf_train as TR
import tf_baseline_std as B
import tf_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = B.DEV
HEAD_DIM = B.HEAD_DIM
ATTNS = ('bilin', 'bilinnorm', 'softmax')
MLPS = ('bilin', 'gelu')
L1_EPS = 1e-6


# ---------------------------------------------------------------- config
@dataclass
class FacConfig:
    attn: str = 'softmax'
    mlp: str = 'bilin'
    depth: int = 2
    width: int = 128
    vocab: int = 8192
    tok: str = 'bpe'
    seed: int = 0
    head_dim: int = HEAD_DIM
    T: int = 512
    qk_norm: bool = True
    hidden: int = 0                 # 0 = derive by parameter matching
    suffix: str = ''
    variant: str = 'factorial'      # so downstream tooling can branch
    group_coeff: float = 0.0

    @property
    def n_heads(self):
        return self.width // self.head_dim

    def __post_init__(self):
        assert self.attn in ATTNS, self.attn
        assert self.mlp in MLPS, self.mlp
        assert self.width % self.head_dim == 0
        if not self.hidden:
            self.hidden = matched_hidden(self.attn, self.mlp, self.width)

    def stem(self):
        v = f'b{self.vocab}' if self.tok == 'bpe' else f'v{self.vocab}'
        return (f'tff_{self.attn}_{self.mlp}_d{self.depth}_w{self.width}'
                f'_{v}_s{self.seed}{self.suffix}')


def attn_coef(attn):
    """Body cost of the attention half, in units of W^2 per block."""
    return 4 if attn == 'softmax' else 6


def mlp_coef(mlp):
    """Body cost of the feed-forward half, in units of H*W per block."""
    return 2 if mlp == 'gelu' else 3


def matched_hidden(attn, mlp, width):
    """H such that the body equals the family's 18 W^2 + W per block, rounded
    to the nearest integer.  The rounding residual is reported, never hidden."""
    return max(1, round((18 - attn_coef(attn)) * width / mlp_coef(mlp)))


def fac_body_params(cfg):
    """Closed form, checked against the live count in control G3."""
    W, H = cfg.width, cfg.hidden
    return cfg.depth * (attn_coef(cfg.attn) * W * W
                        + mlp_coef(cfg.mlp) * H * W + W)


# ---------------------------------------------------------------- model
class FacBlock(nn.Module):
    """Parameter NAMES deliberately match whichever reference this arm reduces
    to, so both gates are a plain state-dict transplant with no name map."""

    def __init__(self, cfg):
        super().__init__()
        W, H = cfg.width, cfg.hidden
        self.c_q = nn.Linear(W, W, bias=False)
        self.c_k = nn.Linear(W, W, bias=False)
        if cfg.attn != 'softmax':
            self.c_q2 = nn.Linear(W, W, bias=False)
            self.c_k2 = nn.Linear(W, W, bias=False)
        self.c_v = nn.Linear(W, W, bias=False)
        if cfg.mlp == 'gelu':
            self.fc_up = nn.Linear(W, H, bias=False)
            self.fc_down = nn.Linear(H, W, bias=True)
        else:
            self.Left = nn.Linear(W, H, bias=False)
            self.Right = nn.Linear(W, H, bias=False)
            self.Down = nn.Linear(H, W, bias=False)
            self.Down_bias = nn.Parameter(torch.zeros(W))
        self.c_proj = nn.Linear(W, W, bias=False)


class FacTransformer(nn.Module):
    def __init__(self, cfg: FacConfig):
        super().__init__()
        self.cfg = cfg
        W = cfg.width
        self.W = W
        torch.manual_seed(cfg.seed)
        self.wte = nn.Embedding(cfg.vocab, W)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.h = nn.ModuleList([FacBlock(cfg) for _ in range(cfg.depth)])
        with torch.no_grad():
            for blk in self.h:
                blk.c_proj.weight.zero_()
                if cfg.mlp == 'gelu':
                    blk.fc_down.weight.zero_()
                    blk.fc_down.bias.zero_()
                else:
                    blk.Down.weight.zero_()
                    blk.Down_bias.zero_()
        cos, sin = M.rope_tables_exact(cfg.T, cfg.head_dim, 'cpu')
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.register_buffer('mask', torch.tril(
            torch.ones(cfg.T, cfg.T, dtype=torch.bool)))
        self.muon_exclude = ()

    def pattern(self, blk, hn, B_, Tq, cos, sin, mask):
        cfg = self.cfg
        H, hd = cfg.n_heads, cfg.head_dim

        def qk(lin):
            z = lin(hn).view(B_, Tq, H, hd)
            if cfg.qk_norm:
                z = F.rms_norm(z, (hd,))
            return M.apply_rot(z, cos, sin)

        q, k = qk(blk.c_q), qk(blk.c_k)
        if cfg.attn == 'softmax':
            att = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(hd)
            return att.masked_fill(~mask, float('-inf')).softmax(-1)
        q2, k2 = qk(blk.c_q2), qk(blk.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / hd
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if cfg.attn == 'bilinnorm':
            pat = pat / (pat.abs().sum(-1, keepdim=True) + L1_EPS)
        return pat

    def forward(self, idx, pat_out=None):
        cfg = self.cfg
        W = self.W
        B_, Tq = idx.shape
        cos = self.cos[None, :Tq, None, :]
        sin = self.sin[None, :Tq, None, :]
        mask = self.mask[:Tq, :Tq]
        x = F.rms_norm(self.wte(idx), (W,))
        for blk in self.h:
            hn = F.rms_norm(x, (W,))
            pat = self.pattern(blk, hn, B_, Tq, cos, sin, mask)
            if pat_out is not None:
                pat_out.append(pat.detach())
            v = blk.c_v(hn).view(B_, Tq, cfg.n_heads, cfg.head_dim)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B_, Tq, W)
            x = x + blk.c_proj(y)
            xn = F.rms_norm(x, (W,))
            if cfg.mlp == 'gelu':
                x = x + blk.fc_down(F.gelu(blk.fc_up(xn)))
            else:
                x = x + blk.Down(blk.Left(xn) * blk.Right(xn)) + blk.Down_bias
        x = F.rms_norm(x, (W,))
        return 30 * torch.tanh((x @ self.wte.weight.t()) / 30)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def body_params(self):
        return self.n_params() - self.wte.weight.numel()


def make_fac(cfg, device=DEV):
    return FacTransformer(cfg).to(device)


def load_fac(stem, device=DEV):
    ck = torch.load(f'{HERE}/{stem}.pt', map_location=device,
                    weights_only=False)
    cfg = FacConfig(**ck['cfg'])
    m = FacTransformer(cfg).to(device)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m, cfg, ck.get('log', {})


# ---------------------------------------------------------------- gates
@torch.no_grad()
def _transplant_and_compare(mine, ref, idx):
    """Copy ref's parameters into mine BY NAME, refusing to skip any, then
    compare forwards.  A name that exists on one side only is a hard failure --
    that is the whole point of the gate."""
    sd_mine = dict(mine.named_parameters())
    sd_ref = dict(ref.named_parameters())
    if set(sd_mine) != set(sd_ref):
        return {'pass': False, 'max_abs_diff': None,
                'only_in_factorial': sorted(set(sd_mine) - set(sd_ref)),
                'only_in_reference': sorted(set(sd_ref) - set(sd_mine))}
    for n, p in sd_mine.items():
        if p.shape != sd_ref[n].shape:
            return {'pass': False, 'max_abs_diff': None,
                    'shape_mismatch': [n, list(p.shape),
                                       list(sd_ref[n].shape)]}
        p.copy_(sd_ref[n])
    mine.eval()
    ref.eval()
    a = mine(idx).float()
    b = ref(idx).float()
    d = (a - b).abs().max().item()
    return {'pass': bool(d < 1e-5), 'max_abs_diff': d,
            'n_params_matched': len(sd_mine)}


def controls(depth=2, width=128, vocab=512, seed=0):
    """G1-G4.  Run on a small vocabulary so the gates are seconds, not minutes;
    nothing they test depends on vocabulary size."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = DEV
    out = {'note': 'gates that must pass before any factorial arm is read',
           'tf32_disabled': True, 'device': dev,
           'grid': {'depth': depth, 'width': width, 'vocab': vocab}}
    g = torch.Generator(device='cpu').manual_seed(7)
    idx = torch.randint(0, vocab, (2, 64), generator=g).to(dev)

    # ---- G1: (bilin, bilin) IS the family's vanilla model
    fcfg = FacConfig(attn='bilin', mlp='bilin', depth=depth, width=width,
                     vocab=vocab, seed=seed, T=512)
    vcfg = M.TFConfig(depth=depth, width=width, vocab=vocab, tok='bpe',
                      seed=seed, variant='vanilla', T=512)
    r = _transplant_and_compare(make_fac(fcfg, dev),
                                M.TinyBilin(vcfg).to(dev), idx)
    r['claim'] = ('the factorial at (bilin, bilin) reproduces '
                  'tf_model.TinyBilin variant=vanilla')
    r['hidden'] = fcfg.hidden
    r['family_hidden'] = 4 * width
    r['hidden_agrees'] = bool(fcfg.hidden == 4 * width)
    out['G1_family_corner'] = r

    # ---- G2: (softmax, gelu) IS the conventional baseline at exp 7
    fcfg2 = FacConfig(attn='softmax', mlp='gelu', depth=depth, width=width,
                      vocab=vocab, seed=seed, T=512)
    scfg = B.StdConfig(depth=depth, width=width, vocab=vocab, tok='bpe',
                       seed=seed, exp=7, T=512)
    r2 = _transplant_and_compare(make_fac(fcfg2, dev),
                                 B.StdTransformer(scfg).to(dev), idx)
    r2['claim'] = ('the factorial at (softmax, gelu) reproduces '
                   'tf_baseline_std.StdTransformer at the parameter-matched '
                   'expansion')
    r2['hidden'] = fcfg2.hidden
    r2['std_matched_hidden'] = 7 * width
    r2['hidden_agrees'] = bool(fcfg2.hidden == 7 * width)
    out['G2_conventional_corner'] = r2

    # ---- G3: closed-form body counts equal live counts, all six arms
    rows, ok = [], True
    fam_body = M.vanilla_body_params(
        M.TFConfig(depth=depth, width=width, vocab=vocab, tok='bpe',
                   seed=seed, variant='vanilla', T=512))
    for a in ATTNS:
        for m_ in MLPS:
            c = FacConfig(attn=a, mlp=m_, depth=depth, width=width,
                          vocab=vocab, seed=seed, T=512)
            live = make_fac(c, 'cpu').body_params()
            cf = fac_body_params(c)
            ok = ok and live == cf
            rows.append({'attn': a, 'mlp': m_, 'hidden': c.hidden,
                         'closed_form_body': cf, 'live_body': live,
                         'agrees': bool(live == cf),
                         'family_body': fam_body,
                         'body_over_family': round(live / fam_body, 6),
                         'residual_params_vs_family': live - fam_body})
    out['G3_parameter_accounting'] = {'pass': bool(ok), 'arms': rows}

    # ---- G4: bilinnorm IS bilin, then divided by the row L1.
    # Two models with the SAME weights, one per attention level, compared on
    # the FIRST block's pattern.  First block only, deliberately: from block 1
    # on the two models have different residual streams (that is the point of
    # the arm), so a later-block comparison would not be an identity test.
    # Comparing against the raw pattern this way is also what makes the test
    # non-vacuous -- `pat_out` hands back the pattern AFTER normalisation, so
    # measuring its L1 against a recomputed denominator would divide twice.
    cn = FacConfig(attn='bilinnorm', mlp='bilin', depth=depth, width=width,
                   vocab=vocab, seed=seed, T=512)
    cr = FacConfig(attn='bilin', mlp='bilin', depth=depth, width=width,
                   vocab=vocab, seed=seed, T=512)
    mn, mr = make_fac(cn, dev), make_fac(cr, dev)
    with torch.no_grad():
        sd = dict(mr.named_parameters())
        for n, p in mn.named_parameters():
            p.copy_(sd[n])
        for m_ in (mn, mr):                     # undo the zero-init so the
            for blk in m_.h:                    # stream is nonzero downstream
                blk.c_proj.weight.normal_(0, 0.02)
                blk.Down.weight.normal_(0, 0.02)
        sd = dict(mr.named_parameters())
        for n, p in mn.named_parameters():
            p.copy_(sd[n])
        pn, pr = [], []
        mn(idx, pat_out=pn)
        mr(idx, pat_out=pr)
    Tq = idx.shape[1]
    tri = torch.tril(torch.ones(Tq, Tq, dtype=torch.bool, device=dev))
    raw = pr[0]
    mass = raw.abs().sum(-1, keepdim=True)
    expect = raw / (mass + L1_EPS)
    g4_id = float((pn[0] - expect).abs().max())
    g4_leak = float(pn[0].masked_select(~tri).abs().max())
    g4_min_mass = float(mass.min())
    g4_l1 = float((pn[0].abs().sum(-1) - 1.0).abs().max())
    out['G4_row_normalisation'] = {
        'pass': bool(g4_id < 1e-6 and g4_leak == 0.0),
        'max_abs_deviation_from_raw_over_L1': g4_id,
        'identity_tested': ('block 0 of the bilinnorm arm == block 0 of the '
                            'bilin arm at identical weights, divided by its '
                            'row L1 plus 1e-6'),
        'max_abs_value_on_a_masked_key': g4_leak,
        'smallest_raw_row_mass_seen': g4_min_mass,
        'resulting_max_abs_row_L1_minus_one': g4_l1,
        'why_that_is_not_zero': ('the 1e-6 denominator floor, which bites only '
                                 'on rows whose raw mass is comparable to it -- '
                                 'query position 0, one visible key, on an '
                                 'untrained model. It is a property of the '
                                 'definition, not slack in the gate, which is '
                                 'why the gate tests the identity above and '
                                 'not this number.')}
    # the OTHER half of the definitional claim: the two arms must NOT agree
    # once normalisation actually does something, or the arm is a no-op.
    out['G4_row_normalisation']['arms_actually_differ'] = bool(
        float((pn[0] - pr[0]).abs().max()) > 1e-3)
    out['all_pass'] = bool(out['G1_family_corner']['pass']
                           and out['G2_conventional_corner']['pass']
                           and out['G3_parameter_accounting']['pass']
                           and out['G4_row_normalisation']['pass'])
    json.dump(out, open(f'{HERE}/tf_factorial_controls.json', 'w'), indent=2)
    print(json.dumps(out, indent=2), flush=True)
    return out


# ---------------------------------------------------------------- cell
def run_fac_cell(attn='softmax', mlp='bilin', depth=2, width=128, seed=0,
                 vocab=8192, tok='bpe', lr=0.02, suffix='', qk_norm=True):
    cfg = FacConfig(attn=attn, mlp=mlp, depth=depth, width=width, vocab=vocab,
                    tok=tok, seed=seed, T=TR.T, suffix=suffix, qk_norm=qk_norm)
    stem = cfg.stem()
    jp = f'{HERE}/{stem}.json'
    out = json.load(open(jp)) if os.path.exists(jp) else {}
    if out.get('run', {}).get('final_held_ce') is not None \
            and os.path.exists(f'{HERE}/{stem}.pt'):
        print(f'{stem}: already trained -- skip', flush=True)
        return out
    gp = f'{HERE}/tf_factorial_controls.json'
    assert os.path.exists(gp) and json.load(open(gp))['all_pass'], \
        'run `python tf_factorial.py controls` first -- gates G1-G4 must pass'
    corpus = TR.Corpus(vocab, tok)
    fam = M.TFConfig(depth=depth, width=width, vocab=vocab, tok=tok, seed=seed,
                     variant='vanilla', T=TR.T)
    fam_body = M.vanilla_body_params(fam)
    out['registered_predictions_file'] = 'tf_factorial_predictions.json'
    out['gates_file'] = 'tf_factorial_controls.json'
    out['config'] = dict(vars(cfg))
    out['config']['n_heads'] = cfg.n_heads
    out['factor_levels'] = {'attention': attn, 'feed_forward': mlp}
    out['is_a_known_corner'] = (
        'family vanilla' if (attn, mlp) == ('bilin', 'bilin') else
        'conventional baseline' if (attn, mlp) == ('softmax', 'gelu') else
        None)
    out['foldable'] = {
        'attention': attn == 'bilin',
        'feed_forward': mlp == 'bilin',
        'note': ('bilinnorm is DIAGNOSTIC ONLY -- its row denominator depends '
                 'on every visible key, so it does not fold to a fixed '
                 'token-pair table and must never be reported as a foldable '
                 'win')}
    out['params'] = {
        'closed_form_body': fac_body_params(cfg),
        'family_body_same_cell': fam_body,
        'body_over_family': round(fac_body_params(cfg) / fam_body, 6),
        'residual_params_vs_family': fac_body_params(cfg) - fam_body,
        'hidden': cfg.hidden,
        'embedding': vocab * width}
    out['corpus'] = {
        'vocab': vocab, 'tokenizer': tok, 'seq_len': TR.T + 1,
        'train_split_rows': corpus.ntr,
        'data_order_id': f'epoch_order(0)@seed{TR.DATA_SEED}',
        'single_epoch_arithmetic': f'{TR.STEPS} steps x batch {TR.BATCH}',
        'held': f'held split rows [0:{TR.HELD_EVAL_N}]'}
    json.dump(out, open(jp, 'w'), indent=2)
    model = make_fac(cfg, DEV)
    log, model = B.train_std(cfg, corpus, lr_muon=lr, stem=stem, model=model)
    out['run'] = log
    bp = f'{HERE}/tf_baselines_' + \
         (f'b{vocab}' if tok == 'bpe' else f'v{vocab}') + '.json'
    if os.path.exists(bp) and not log['diverged']:
        b = json.load(open(bp))
        out['vs_baselines'] = {
            'unigram_floor': b['unigram_floor_held_ce'],
            'bigram': b['bigram_held_ce'],
            'model_bits_per_byte': round(
                tf_corpus.bits_per_byte(log['final_held_ce'], vocab, tok), 5)}
    fp = f'{HERE}/tf_vanilla_d{depth}_w{width}_b{vocab}_s{seed}.json'
    if os.path.exists(fp) and not log['diverged']:
        fce = json.load(open(fp)).get('run', {}).get('final_held_ce')
        if fce is not None:
            out['vs_family_same_cell'] = {
                'family_held_ce': fce,
                'this_arm_held_ce': log['final_held_ce'],
                'nats_below_family': round(fce - log['final_held_ce'], 5),
                'sign': 'positive = this arm beats the fully foldable family'}
    json.dump(out, open(jp, 'w'), indent=2)
    del model
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['controls', 'cell'])
    ap.add_argument('--attn', default='softmax', choices=ATTNS)
    ap.add_argument('--mlp', default='bilin', choices=MLPS)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--vocab', type=int, default=8192)
    ap.add_argument('--tok', default='bpe')
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--no-qk-norm', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'controls':
        controls()
    else:
        run_fac_cell(attn=a.attn, mlp=a.mlp, depth=a.depth, width=a.width,
                     seed=a.seed, vocab=a.vocab, tok=a.tok, lr=a.lr,
                     suffix=a.suffix, qk_norm=not a.no_qk_norm)
