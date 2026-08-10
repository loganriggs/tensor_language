"""D-to-minus-I sparsity sweep (Logan's F9 feedback). Registered:
predictions/part2_Dsweep_prediction.md. Reuses part2.py's sgd_train verbatim."""
import json, numpy as np, part2 as P

rng = np.random.default_rng(P.DATA_SEED if hasattr(P, 'DATA_SEED') else 0)
Z, y = P.make_facts() if hasattr(P, 'make_facts') else (None, None)
if Z is None:
    raise SystemExit('adjust: no make_facts')
out = {}
for l1 in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
    for seed in range(5):
        (L, R, D), _, acc, _ = P.sgd_train(Z, y, 40, seed, l1=l1)
        cn = np.linalg.norm(D, axis=0)
        live = cn > 1e-3 * cn.max()
        dom = np.abs(D[:, live]).max(0) / cn[live]
        neg = D[np.abs(D[:, live]).argmax(0), np.where(live)[0]] < 0
        out[f'{l1}_{seed}'] = {'l1': l1, 'seed': seed, 'acc': round(acc, 4),
            'live_units': int(live.sum()), 'dominance_mean': round(float(dom.mean()), 4),
            'dominance_median': round(float(np.median(dom)), 4),
            'neg_dominant_frac': round(float(neg.mean()), 4)}
        print(out[f'{l1}_{seed}'], flush=True)
json.dump(out, open('part2_dsweep.json', 'w'), indent=1)
print('done')
