"""RUNG 297: shifted-corpus OOD for literal mixed104 online-c_v0.

The source is WikiText-2 raw TEST (Salesforce/wikitext), tokenized with GPT-2
and concatenated with explicit paragraph separators. It is a different corpus,
not a later window of Pile-10k. Rows are deterministic non-overlapping
257-token chunks after skipping the first 1024 tokens. No WikiText token,
label, or statistic is used to fit any program tensor.

REGISTERED PREDICTIONS (compiled CE minus native CE on the SAME positions):
  (a) SHIFTED MEAN: mean damage <=0.012.
  (b) SHIFTED TAIL: 95th percentile of the 120 row-mean damages <=0.020.
  (c) SANITY/IDENTITY: native WikiText CE in [2.0,8.0]; exact mixed104 index
      set and width; active set excludes a0/a1v/tailE; 120 rows scored.
NULL: mean damage >=0.030 or row-damage p95 >=0.060, showing the Pile result
does not transport. PRICE unchanged: 539,595,062 standalone scalars.
Self-reviewed; bqrunner only.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
OUT = ROOT / 'mixed104_online_cv0_ood_results.json'

if os.environ.get('BQLIB_DRYRUN') == '1':
    needed = [ROOT / 'circuits/BATTERY.json', ROOT / 'census_state_diverse.pt',
              ROOT / 'mixed104_online_cv0_results.json',
              ROOT / 'mixed104_exact_bill_results.json',
              ROOT / 'ops/cevdump_ct96.py']
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        print(f'DRYRUN FAIL: missing {missing}')
        raise SystemExit(1)
    winner = json.load(open(ROOT / 'mixed104_online_cv0_results.json'))
    assert winner['pred_a_literal_candidate'] and winner['pred_b_table_equivalence']
    print('DRYRUN OK: rung 297 WikiText-2 shifted OOD')
    raise SystemExit(0)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ops'))
sys.path.insert(0, '/workspace/rspd')

import torch
from datasets import load_dataset
import tiktoken
import census_lib as CN
import cevdump_ct96 as C


def wikitext_rows(n=120, width=257, skip=1024):
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(row['text'] for row in ds if row['text'].strip())
    toks = tiktoken.get_encoding('gpt2').encode_ordinary(text)
    stop = skip + n * width
    if len(toks) < stop:
        raise RuntimeError(f'WikiText token stream too short: {len(toks)} < {stop}')
    rows = torch.tensor(toks[skip:stop], dtype=torch.long).reshape(n, width)
    return rows, str(ds._fingerprint), len(toks)


@torch.no_grad()
def main():
    started = time.time()
    rows_ood, fingerprint, token_count = wikitext_rows()
    CN.use_state('census_state_diverse.pt')
    rows = CN.rows().cpu()
    C.CROWS = rows
    C.CBASE = CN.base_ce().float().cpu()
    C.NFLAT = CN.nflat()
    C.ANCH = json.load(open(ROOT / 'frontier_tail_traj_results.json'))
    C.SEL.update({
        'mode': 'norm', 'K': 4608, 'K69': 4608, 'K69MAP': {},
        'skipset': tuple(range(10,18)), 'motif_off': (), 'clsdmg': True,
        'ext_rows': rows, 'cp_swap': 4608, 'qk_r': 96,
        'qk_rmap': {}, 'qk_extra_tail': 8, 'qk_tail': True,
        'drop_tailE': True, 'drop_a1v': True, 'drop_a0': True,
        'extra_eval_rows': rows_ood, 'extra_eval_name': 'wikitext-2-raw-v1-test',
    })
    print('ARM: literal mixed104 online-c_v0 + frozen WikiText-2 OOD', flush=True)
    C.main()

    wanted = tuple(list(range(96)) + list(range(120,128)))
    index_sets = C.SEL.get('_QK_INDEX_SETS', {})
    qk = C.SEL.get('_QKR', {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    active = tuple(C.SEL.get('_ORDER2', ()))
    ev = C.SEL['extra_eval']
    by_row = torch.tensor(ev['damage_by_row'])
    p95 = float(torch.quantile(by_row, 0.95))
    mean = float(ev['damage_mean'])
    pred_a = mean <= 0.012
    pred_b = p95 <= 0.020
    pred_c = (2.0 <= ev['native_ce'] <= 8.0 and ev['n_rows'] == 120 and
              set(index_sets) == set(range(2,18)) and
              all(value == wanted for value in index_sets.values()) and
              widths == {104} and
              not any(name in active for name in ('a0','a1v','tailE')))
    null = mean >= 0.030 or p95 >= 0.060
    bill = json.load(open(ROOT / 'mixed104_exact_bill_results.json'))
    result = {
        'convention': 'compiled CE minus native CE on identical WikiText positions',
        'dataset': 'Salesforce/wikitext:wikitext-2-raw-v1:test',
        'dataset_fingerprint': fingerprint,
        'tokenizer': 'tiktoken gpt2',
        'source_token_count': token_count,
        'row_construction': {'skip_tokens':1024,'n_rows':120,'tokens_per_row':257},
        'native_ce': round(ev['native_ce'], 8),
        'compiled_ce': round(ev['compiled_ce'], 8),
        'damage_mean': round(mean, 8),
        'damage_mean_abs_position': round(ev['damage_mean_abs_position'], 8),
        'damage_row_p50': round(float(torch.quantile(by_row,0.50)), 8),
        'damage_row_p95': round(p95, 8),
        'damage_row_min': round(float(by_row.min()), 8),
        'damage_row_max': round(float(by_row.max()), 8),
        'damage_by_row': [round(float(v),8) for v in by_row],
        'qk_singular_indices': list(wanted),
        'physical_qk_factor_widths': sorted(widths),
        'active_replacements': list(active),
        'literal_standalone_scalars': bill['storage_minimal_semantic_candidate']['scalars'],
        'pred_a_shifted_mean': bool(pred_a),
        'pred_b_shifted_tail': bool(pred_b),
        'pred_c_sanity_and_identity': bool(pred_c),
        'null_triggered': bool(null),
        'decision_level': 'shifted-corpus predictive gate; broader signed battery remains',
        'runtime_s': round(time.time()-started,1),
    }
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='damage_by_row'},indent=2),flush=True)
    print(f'wrote {OUT}',flush=True)


if __name__ == '__main__':
    main()
