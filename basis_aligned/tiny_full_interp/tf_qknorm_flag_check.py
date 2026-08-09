"""TASK 1 GATE: prove that adding TFConfig.qk_norm / tf_train --no-qk-norm left
the DEFAULT path bit-identical, and that the new qk_norm=False path runs.

Three checks, all reported as numbers (a gate that only prints "OK" is not a
gate):

  A. BITWISE.  The pre-edit tf_model.py (kept as a .bak by the patch script) is
     imported alongside the post-edit one and BOTH build TinyBilin at the same
     seed.  Requirement: every parameter identical bit-for-bit AND the forward
     logits identical bit-for-bit (max|diff| == 0.0 exactly, not "small") on
     fixed tokens, at fp32 and fp64, inside exact_math().  This is strictly
     stronger than a tolerance and it is the check that a conditional-on-a-
     default-True flag actually owes.
  B. TRAINED-CHECKPOINT REPRODUCTION.  Load tf_vanilla_d2_w128_b8192_s0 and
     re-run tf_train.eval_held on the SAME held rows [0:1500] at context 512
     with the same bf16 autocast the number was produced under; require
     |reproduced - run.final_held_ce| < 1e-4.  (Split / context / token count
     are recorded next to the number -- README standing failure mode "two
     evaluations of the same quantity, quoted without labels".)
  C. qk_norm=False SMOKE.  A tiny CPU model (vocab 64, width 32, depth 2) takes
     a few optimizer steps on random tokens; require finite loss throughout and
     that the per-head RMSNorm really is gone (the cap-off forward must DIFFER
     from the cap-on forward at the same weights -- otherwise the flag is a
     no-op and check A would pass vacuously).

Usage:  python tf_qknorm_flag_check.py [--bak /path/to/tf_model.py.bak]
"""
import argparse
import importlib.machinery
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BAK = ('/tmp/claude-0/-workspace-tensor-language/'
               'a6c5fb86-7bce-48e6-bb32-8679e85cbf66/scratchpad/tf_model.py.bak')


def load_module(path, name):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def preedit_source(bak_path):
    """The pre-edit tf_model.py.  Preferred source is git HEAD (auditable by
    anyone: `git show HEAD:...tf_model.py`); the patch script's .bak is only a
    fallback for when the edit has already been committed."""
    import subprocess
    r = subprocess.run(['git', '-C', HERE, 'show',
                        'HEAD:basis_aligned/tiny_full_interp/tf_model.py'],
                       capture_output=True, text=True)
    if r.returncode == 0 and 'class TinyBilin' in r.stdout:
        p = os.path.join(os.path.dirname(bak_path) if bak_path else '/tmp',
                         'tf_model_preedit_from_git.py')
        open(p, 'w').write(r.stdout)
        return p, 'git HEAD'
    return bak_path, 'patch-script .bak'


# ------------------------------------------------------------------ check A
def check_bitwise(bak_path, depth=2, width=128, vocab=512, seed=0, device='cpu'):
    import tf_model as NEW
    bak_path, prov = preedit_source(bak_path)
    out = {'preedit_source': bak_path, 'preedit_provenance': prov,
           'available': bool(bak_path) and os.path.exists(bak_path)}
    if not out['available']:
        out['pass'] = None
        out['note'] = 'pre-edit tf_model.py not available; check A skipped'
        return out
    OLD = load_module(bak_path, 'tf_model_preedit')
    out['old_has_qk_norm_field'] = 'qk_norm' in OLD.TFConfig.__dataclass_fields__
    out['new_has_qk_norm_field'] = 'qk_norm' in NEW.TFConfig.__dataclass_fields__
    out['new_default_qk_norm'] = NEW.TFConfig().qk_norm

    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, vocab, (3, 48), generator=g).to(device)
    res = {}
    with NEW.exact_math():
        for dt_name, dt in (('fp32', torch.float32), ('fp64', torch.float64)):
            mo = OLD.TinyBilin(OLD.TFConfig(depth=depth, width=width,
                                            vocab=vocab, tok='bpe', seed=seed,
                                            variant='vanilla', T=512)).to(device)
            mn = NEW.TinyBilin(NEW.TFConfig(depth=depth, width=width,
                                            vocab=vocab, tok='bpe', seed=seed,
                                            variant='vanilla', T=512)).to(device)
            so, sn = mo.state_dict(), mn.state_dict()
            assert set(so) == set(sn), (sorted(set(so) ^ set(sn)))
            pmax = max(float((so[k].to(torch.float64)
                              - sn[k].to(torch.float64)).abs().max())
                       for k in so if so[k].is_floating_point())
            mo = (mo.double() if dt is torch.float64 else mo).eval()
            mn = (mn.double() if dt is torch.float64 else mn).eval()
            with torch.no_grad():
                a, b = mo(idx), mn(idx)
            res[f'{dt_name}_param_max_abs_diff'] = pmax
            res[f'{dt_name}_forward_max_abs_diff'] = float((a - b).abs().max())
            res[f'{dt_name}_forward_bitwise_equal'] = bool(torch.equal(a, b))
            # the FOLD path is edited too -- gate it identically
            with torch.no_grad():
                fa, fb = mo.fold_forward(idx), mn.fold_forward(idx)
            res[f'{dt_name}_fold_forward_max_abs_diff'] = float(
                (fa - fb).abs().max())
            res[f'{dt_name}_fold_forward_bitwise_equal'] = bool(
                torch.equal(fa, fb))
            del mo, mn
    out.update(res)
    out['pass'] = bool(all(v for k, v in res.items()
                           if k.endswith('bitwise_equal')))
    return out


# ------------------------------------------------------------------ check B
def check_checkpoint(stem='tf_vanilla_d2_w128_b8192_s0', tol=1e-4):
    """Re-run eval_held on a trained checkpoint and compare to the stored
    run.final_held_ce.  Uses a held-only corpus shim so the 240k-row train
    split is never loaded onto the (possibly shared) GPU."""
    import numpy as np
    import tf_corpus
    import tf_model as M
    import tf_train as TR
    out = {'stem': stem, 'tol': tol}
    jp = f'{HERE}/{stem}.json'
    ck = torch.load(f'{HERE}/{stem}.pt', map_location='cpu',
                    weights_only=False)
    stored = json.load(open(jp))['run']['final_held_ce']
    cfg = M.TFConfig(**{k: v for k, v in ck['cfg'].items()
                        if k in M.TFConfig.__dataclass_fields__})
    model = M.make_model(cfg, TR.DEV)
    model.load_state_dict(ck['state_dict'])
    model.eval()

    class HeldOnly:
        pass
    c = HeldOnly()
    c.V, c.tok = cfg.vocab, cfg.tok
    held = tf_corpus.load_split(cfg.vocab, 'held', TR.HELD_EVAL_N, tok=cfg.tok)
    c.held = torch.from_numpy(held).to(TR.DEV)
    ce, _ = TR.eval_held(model, c)
    out.update({
        'stored_run_final_held_ce': stored,
        'reproduced_held_ce': round(float(ce), 6),
        'abs_diff': abs(float(ce) - stored),
        'device': TR.DEV,
        'eval_labels': {
            'split': 'held', 'rows': f'[0:{TR.HELD_EVAL_N}]',
            'context_len': TR.T,
            'target_tokens': int(c.held.shape[0] * TR.T),
            'autocast': f'bfloat16 on {TR.DEV}' if TR.DEV == 'cuda' else 'none',
            'note': 'same function, split, rows, context and autocast the '
                    'stored number was produced under'},
        'cfg_qk_norm_after_load': cfg.qk_norm})
    out['pass'] = bool(out['abs_diff'] < tol)
    del model
    if TR.DEV == 'cuda':
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ check C
def check_noqknorm_smoke(steps=8, vocab=64, width=32, depth=2, T=32):
    import tf_model as M
    out = {'vocab': vocab, 'width': width, 'depth': depth, 'steps': steps,
           'device': 'cpu'}
    cfg_off = M.TFConfig(depth=depth, width=width, vocab=vocab, tok='bpe',
                         seed=0, variant='vanilla', T=T, qk_norm=False)
    cfg_on = M.TFConfig(depth=depth, width=width, vocab=vocab, tok='bpe',
                        seed=0, variant='vanilla', T=T, qk_norm=True)
    m_off = M.TinyBilin(cfg_off).to('cpu')
    # TinyBilin ZERO-INITIALISES c_proj and Down, so at init nothing downstream
    # of the attention pattern moves and a cap-on/cap-off comparison would be
    # 0.0 for a reason that has nothing to do with the flag (this bit the first
    # version of this check).  Randomise the write matrices first.
    gp = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for blk in m_off.h:
            for p in (blk.c_proj.weight, blk.Down.weight):
                p.copy_(torch.randn(p.shape, generator=gp) * 0.05)
            blk.Down_bias.copy_(
                torch.randn(blk.Down_bias.shape, generator=gp) * 0.01)
    m_on = M.TinyBilin(cfg_on).to('cpu')
    m_on.load_state_dict(m_off.state_dict())
    out['note_writes_randomised'] = (
        'c_proj / Down / Down_bias randomised before the comparison; they are '
        'zero at init, which would make any cap-on/cap-off diff vacuously 0')
    g = torch.Generator().manual_seed(5)
    idx = torch.randint(0, vocab, (4, T), generator=g)
    with torch.no_grad():
        d = float((m_off(idx) - m_on(idx)).abs().max())
    out['cap_off_vs_cap_on_max_logit_diff'] = d
    out['flag_is_not_a_noop'] = bool(d > 1e-3)

    opt = torch.optim.AdamW(m_off.parameters(), lr=1e-3)
    losses = []
    m_off.train()
    for s in range(steps):
        b = torch.randint(0, vocab, (4, T + 1), generator=g)
        logits = m_off(b[:, :T])
        loss = F.cross_entropy(logits.reshape(-1, vocab), b[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m_off.parameters(), 1.0))
        opt.step()
        losses.append([s, round(float(loss), 5), round(gn, 5)])
    out['loss_curve_step_loss_gradnorm'] = losses
    out['all_finite'] = bool(all(torch.isfinite(torch.tensor(l[1]))
                                 and torch.isfinite(torch.tensor(l[2]))
                                 for l in losses))
    # and the fold machinery must follow the flag: with qk_norm False the
    # folded Q factors must equal the RAW projection, not the normalised one.
    with torch.no_grad():
        f = m_off.fold_layer0_qk(materialize=False, dtype=torch.float64)
        Ehn = m_off.token_input_table(torch.float64)
        raw = (Ehn @ m_off.h[0].c_q.weight.double().t()).view(
            vocab, cfg_off.n_heads, cfg_off.head_dim).permute(1, 0, 2)
        out['fold_Q1_equals_raw_projection_when_cap_off'] = float(
            (f['Q1'] - raw).abs().max())
    out['pass'] = bool(out['all_finite'] and out['flag_is_not_a_noop']
                       and out['fold_Q1_equals_raw_projection_when_cap_off']
                       < 1e-12)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bak', default=DEFAULT_BAK)
    ap.add_argument('--out', default=f'{HERE}/tf_qknorm_flag_check.json')
    a = ap.parse_args()
    rep = {'task': 'TASK 1 -- --no-qk-norm flag for tf_train.py/tf_model.py',
           'claim': 'the default (qk_norm=True) path is unchanged; the new '
                    'qk_norm=False path runs and is not a no-op'}
    rep['A_bitwise_vs_preedit_module'] = check_bitwise(a.bak)
    print('A done', flush=True)
    rep['B_trained_checkpoint_reproduction'] = check_checkpoint()
    print('B done', flush=True)
    rep['C_qk_norm_false_smoke'] = check_noqknorm_smoke()
    print('C done', flush=True)
    rep['all_pass'] = bool(
        (rep['A_bitwise_vs_preedit_module']['pass'] is not False)
        and rep['B_trained_checkpoint_reproduction']['pass']
        and rep['C_qk_norm_false_smoke']['pass'])
    json.dump(rep, open(a.out, 'w'), indent=2)
    print(json.dumps(rep, indent=2))
