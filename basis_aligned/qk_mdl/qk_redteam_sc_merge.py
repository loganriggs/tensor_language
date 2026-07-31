"""Merge the per-stage red-team result files into the single deliverable qk_redteam_sc.json."""
import json, os
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
MAIN = f'{QK}/qk_redteam_sc.json'
PARTS = ['qk_redteam_sc_refit.json', 'qk_redteam_sc_co.json', 'qk_redteam_sc_cp.json',
         'qk_redteam_sc_nw.json', 'qk_redteam_sc_g2.json']
res = json.load(open(MAIN))
for p in PARTS:
    fp = f'{QK}/{p}'
    if not os.path.exists(fp):
        print(f"  (missing {p})"); continue
    r = json.load(open(fp))
    for sec, val in r.items():
        if sec == 'meta': continue
        if sec not in res: res[sec] = {}
        if isinstance(val, dict):
            for k, v in val.items():
                if sec == 'gauge' and k in res['gauge'] and not k.startswith(('long_', 'restart')):
                    continue
                res[sec][k] = v
        else:
            res[sec] = val
res['meta']['stage_files'] = PARTS
json.dump(res, open(MAIN, 'w'), indent=1)
print("merged sections:", {k: (len(v) if isinstance(v, dict) else 1) for k, v in res.items()})
