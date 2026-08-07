"""E20 follow-up (pre-chain job for the E22 build): the TWO-PERIOD-CODES
question from qk_e20_code_dictionaries.json -- slot 15 (mlp7's write) has two
distinct heavily-used period codes, 69 and 193. Is this CONTEXTUAL SPLITTING
of one token class (the enumerable-superposition case Logan flagged: two
codes = two enumerable contexts of '.') or NOISE DUPLICATION (two codes for
the same distribution, an artifact of VQ dynamics)?

Method: re-derive the FULL code-assignment sets on the fixed audit slice
(fresh34k[33000:33200], the same 200 sequences the dictionaries used) by
running the trained E20a checkpoint with the code-collection hooks (the
saved dictionaries only keep top-10 contexts per code; the assignments
themselves are recomputed deterministically from the checkpoint). Then for
each code: firing-token distribution, PRECEDING-token distribution,
position-in-sequence distribution, per-sequence (document) concentration,
PMI with the slot's step-2 codes and with mlp0's (slot 1) step-1 codes at
the same token, codebook cosine between the two code vectors, and
label-permutation z-scores on the total-variation distances between the two
codes' context distributions.

VERDICT RULE (fixed before computing): CONTEXTUAL SPLITTING if any context
distribution (preceding token / position / firing token) separates the two
codes with TV distance > 0.10 at permutation z > 4; NOISE DUPLICATION if all
TVs < 0.10 (or z < 2) AND the code vectors are near-parallel (cos > 0.9);
MIXED otherwise. Result -> qk_e22.json under 'period_codes_followup'.
CPU-feasible (small model, 200 seqs); uses the GPU when >= 2500 MiB free."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import subprocess
import time


def _gpu_free_mib():
    try:
        return int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits']).decode().split('\n')[0])
    except Exception:
        return -1


USE_CUDA = _gpu_free_mib() >= 2500

import torch                                     # noqa: E402

if not USE_CUDA:
    def _cpu_dev(a):
        if isinstance(a, str) and a.startswith('cuda'):
            return 'cpu'
        if isinstance(a, torch.device) and a.type == 'cuda':
            return torch.device('cpu')
        return a
    _t_to = torch.Tensor.to

    def _tto(self, *args, **kw):
        args = tuple(_cpu_dev(a) for a in args)
        if 'device' in kw:
            kw['device'] = _cpu_dev(kw['device'])
        return _t_to(self, *args, **kw)
    torch.Tensor.to = _tto
    _m_to = torch.nn.Module.to

    def _mto(self, *args, **kw):
        args = tuple(_cpu_dev(a) for a in args)
        if 'device' in kw:
            kw['device'] = _cpu_dev(kw['device'])
        return _m_to(self, *args, **kw)
    torch.nn.Module.to = _mto

import numpy as np                               # noqa: E402

import qk_tokenline_train as _Q0                 # noqa: E402
_Q0.gpu_guard = lambda *a, **k: None

import qk_e_common as E                          # noqa: E402
from qk_e_common import Q, DEPTH                 # noqa: E402
import qk_e17_composed_wiring as E17             # noqa: E402  (width patch)
import qk_e20_codebook_run as E20R               # noqa: E402

DEV = 'cuda' if USE_CUDA else 'cpu'
E.DEV = DEV
JP = E.jpath('qk_e22.json')
SLOT = 15                                        # mlp7's write
CODES = (69, 193)
ROWS = (33000, 33200)                            # THE fixed audit slice
XSLOT = 1                                        # mlp0 (cross-slot PMI)
N_PERM = 1000


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def dist(vals, n):
    c = np.bincount(vals, minlength=n).astype(np.float64)
    return c / max(c.sum(), 1)


def perm_tv_z(vals_a, vals_b, n, rng):
    """Label-permutation z-score for the TV distance between the two codes'
    distributions over `vals` (categorical with n bins)."""
    real = tv(dist(vals_a, n), dist(vals_b, n))
    pool = np.concatenate([vals_a, vals_b])
    na = len(vals_a)
    nulls = np.empty(N_PERM)
    for i in range(N_PERM):
        rng.shuffle(pool)
        nulls[i] = tv(dist(pool[:na], n), dist(pool[na:], n))
    mu, sd = float(nulls.mean()), float(nulls.std())
    return real, (real - mu) / max(sd, 1e-12), mu, sd


def main():
    t0 = time.time()
    done = E.loadj(JP).get('period_codes_followup')
    if done and 'verdict' in done:
        print('period_codes_followup: already done -- skip', flush=True)
        return
    print(f"period-codes follow-up on {DEV}", flush=True)
    m, ck = E.load_arm('qk_e20_a', lambda: E20R.make_e20(s=15))
    m.eval().float()
    held = np.load(E.HELD_PATH)[ROWS[0]:ROWS[1]].astype(np.int64)
    rows = torch.from_numpy(held).to(DEV)
    codes_all, scales_all = {}, {}
    with torch.no_grad():
        for i in range(0, len(rows), 4):
            b = rows[i:i + 4]
            col = {'codes': {}, 'scales': {}}
            m(b[:, :Q.T], collect=col)
            for k in (SLOT, XSLOT):
                codes_all.setdefault(k, []).append(
                    torch.cat(col['codes'][k], 0))
                scales_all.setdefault(k, []).append(
                    torch.cat(col['scales'][k], 0))
    cos_69_193 = float(torch.dot(m.qz_codebook[SLOT][CODES[0]],
                                 m.qz_codebook[SLOT][CODES[1]]))
    del m
    if USE_CUDA:
        torch.cuda.empty_cache()
    codes = torch.cat(codes_all[SLOT], 0).long().numpy()     # (Ntok, 2)
    scales = torch.cat(scales_all[SLOT], 0).numpy()
    xcodes = torch.cat(codes_all[XSLOT], 0).long().numpy()
    ntok = codes.shape[0]
    T = Q.T
    flat_tok = held[:, :T].reshape(-1)
    flat_prev = np.roll(held[:, :T], 1, axis=1)
    flat_prev[:, 0] = -1
    flat_prev = flat_prev.reshape(-1)
    pos = np.arange(ntok) % T
    seq = np.arange(ntok) // T

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('gpt2')

        def dec(i):
            return tok.decode([int(i)]) if i >= 0 else '<BOS>'
    except Exception:
        def dec(i):
            return str(int(i))

    sels = {c: np.nonzero(codes[:, 0] == c)[0] for c in CODES}
    per_code = {}
    for c in CODES:
        s = sels[c]
        curc = np.bincount(flat_tok[s], minlength=Q.V)
        prevc = np.bincount(flat_prev[s] + 1, minlength=Q.V + 1)
        top_cur = np.argsort(curc)[::-1][:8]
        top_prev = np.argsort(prevc)[::-1][:15]
        c2 = np.bincount(codes[s, 1], minlength=E20R.QZ_N)
        top_c2 = np.argsort(c2)[::-1][:8]
        seqc = np.bincount(seq[s], minlength=len(held))
        top_seq = np.argsort(seqc)[::-1][:5]
        per_code[str(c)] = {
            'n_firings_step1': int(len(s)),
            'mean_scale1': round(float(scales[s, 0].mean()), 3),
            'firing_token_top': [
                {'tok': repr(dec(t)), 'count': int(curc[t]),
                 'frac': round(float(curc[t] / len(s)), 3)}
                for t in top_cur if curc[t] > 0],
            'preceding_token_top': [
                {'tok': repr(dec(t - 1)), 'count': int(prevc[t]),
                 'frac': round(float(prevc[t] / len(s)), 3)}
                for t in top_prev if prevc[t] > 0],
            'position_octile_fracs': [
                round(float(((pos[s] // (T // 8)) == o).mean()), 3)
                for o in range(8)],
            'mean_position': round(float(pos[s].mean()), 1),
            'n_distinct_seqs': int((seqc > 0).sum()),
            'top5_seqs_by_count': [[int(ROWS[0] + q), int(seqc[q])]
                                   for q in top_seq],
            'step2_code_top': [
                {'code2': int(j), 'count': int(c2[j])}
                for j in top_c2 if c2[j] > 0]}

    # PMI of each code with slot-15 step-2 codes and with mlp0 step-1 codes
    def pmi_rows(sel, partner_codes, n_codes, min_count=15):
        p_c = len(sel) / ntok
        pc = np.bincount(partner_codes, minlength=n_codes) / ntok
        joint = np.bincount(partner_codes[sel], minlength=n_codes) / ntok
        out = []
        for j in np.nonzero(joint * ntok >= min_count)[0]:
            out.append({'partner': int(j),
                        'count': int(round(joint[j] * ntok)),
                        'pmi_bits': round(math.log2(
                            joint[j] / (p_c * pc[j])), 3)})
        out.sort(key=lambda r: -r['pmi_bits'])
        return out[:10]

    for c in CODES:
        per_code[str(c)]['pmi_slot15_step2'] = pmi_rows(
            sels[c], codes[:, 1], E20R.QZ_N)
        per_code[str(c)]['pmi_mlp0_step1'] = pmi_rows(
            sels[c], xcodes[:, 0], E20R.QZ_N)

    # do 69 and 193 ever co-occur as the (code1, code2) pair on one token?
    co = int(((codes[:, 0] == CODES[0]) & (codes[:, 1] == CODES[1])).sum()
             + ((codes[:, 0] == CODES[1]) & (codes[:, 1] == CODES[0])).sum())

    # permutation tests on the context distributions
    rng = np.random.RandomState(0)
    a, b = sels[CODES[0]], sels[CODES[1]]
    tv_prev, z_prev, mu_p, sd_p = perm_tv_z(flat_prev[a] + 1,
                                            flat_prev[b] + 1, Q.V + 1, rng)
    tv_cur, z_cur, _, _ = perm_tv_z(flat_tok[a], flat_tok[b], Q.V, rng)
    tv_pos, z_pos, _, _ = perm_tv_z(pos[a] // (T // 8), pos[b] // (T // 8),
                                    8, rng)
    ca = np.bincount(seq[a], minlength=len(held)).astype(np.float64)
    cb = np.bincount(seq[b], minlength=len(held)).astype(np.float64)
    seq_corr = float(np.corrcoef(ca, cb)[0, 1])
    tv_seq, z_seq, _, _ = perm_tv_z(seq[a], seq[b], len(held), rng)

    tests = {
        'preceding_token': {'tv': round(tv_prev, 4), 'z': round(z_prev, 1),
                            'null_mean': round(mu_p, 4),
                            'null_sd': round(sd_p, 5)},
        'firing_token': {'tv': round(tv_cur, 4), 'z': round(z_cur, 1)},
        'position_octile': {'tv': round(tv_pos, 4), 'z': round(z_pos, 1)},
        'sequence_identity': {'tv': round(tv_seq, 4), 'z': round(z_seq, 1),
                              'per_seq_count_corr': round(seq_corr, 3)}}

    sep = [(nm, d) for nm, d in tests.items()
           if nm != 'sequence_identity'
           and d['tv'] > 0.10 and d['z'] > 4]
    dup = all(d['tv'] < 0.10 or d['z'] < 2 for nm, d in tests.items()
              if nm != 'sequence_identity')
    if sep:
        verdict = ('CONTEXTUAL SPLITTING (enumerable superposition): the '
                   'two period codes occupy systematically different '
                   'contexts -- separating distributions: '
                   + ', '.join(f"{nm} (TV {d['tv']}, z {d['z']})"
                               for nm, d in sep)
                   + f'; codebook cos {round(cos_69_193, 3)}')
    elif dup and cos_69_193 > 0.9:
        verdict = ('NOISE DUPLICATION: no context distribution separates '
                   'the codes (all TV < 0.10 or z < 2) and the code '
                   f'vectors are near-parallel (cos {round(cos_69_193, 3)})')
    else:
        verdict = (f'MIXED: no single distribution passes the TV > 0.10 & '
                   f'z > 4 bar, but the codes are not clean duplicates '
                   f'either (cos {round(cos_69_193, 3)}; see tests)')

    rec = {
        'question': 'slot 15 (mlp7) period codes 69 vs 193: contextual '
                    'splitting of one token class (enumerable '
                    'superposition) or noise duplication?',
        'audit_slice': f'fresh34k[{ROWS[0]}:{ROWS[1]}] (the fixed audit '
                       'slice; assignments recomputed from qk_e20_a.pt)',
        'verdict_rule': 'pre-fixed: SPLITTING if any context distribution '
                        '(preceding token / firing token / position) has '
                        'TV > 0.10 at permutation z > 4; DUPLICATION if '
                        'all TV < 0.10 (or z < 2) and codebook cos > 0.9; '
                        'MIXED otherwise',
        'codebook_cos_69_193': round(cos_69_193, 4),
        'pair_cooccurrence_69_193_same_token': co,
        'separation_tests_label_permutation': tests,
        'per_code': per_code,
        'device': DEV,
        'runtime_s': round(time.time() - t0, 1),
        'verdict': verdict}
    E.merge(JP, 'period_codes_followup', rec)
    print(json.dumps({'period_codes_followup_tests': tests,
                      'verdict': verdict}, indent=2), flush=True)


if __name__ == '__main__':
    main()
