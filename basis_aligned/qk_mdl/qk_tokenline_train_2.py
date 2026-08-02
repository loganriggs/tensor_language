"""TOKEN-LINE EXPERIMENT, fix/extension run: the 15-epoch budget OVERFITS (variant A's
held CE bottoms ~5.66 near step 4000 = ~6 epochs, then climbs to 6.89). Protocol kept:
same frozen lr (from the variant-A sweep), same seed/init/data order — but a SHORT
budget of 6 epochs (4122 steps), landing near the held-CE optimum. Models saved with
suffix 'S': qk_tokenline_{A,B,C,D}S.pt, per-token held losses qk_tokenline_heldloss_
{V}S.npy, paired CE differences -> qk_tokenline_ce_short.json. The 15-epoch runs stay
as the primary fixed-budget comparison; this checks that conclusions are not an
artifact of the overfit regime."""
import json, math, os
import numpy as np
import qk_tokenline_train as Q

QK = Q.QK
EPOCHS_S = 6
STEPS = EPOCHS_S * Q.STEPS_PER_EPOCH

if __name__ == '__main__':
    LR = json.load(open(f'{QK}/qk_tokenline_lrsweep.json'))['chosen']
    print(f"short-budget run: {STEPS} steps ({EPOCHS_S} epochs), lr {LR}", flush=True)
    for variant in ['A', 'B', 'C', 'D']:
        if os.path.exists(f'{QK}/qk_tokenline_{variant}S.pt'):
            print(f"variant {variant}S already trained -- skip", flush=True)
            continue
        print(f"==== training variant {variant}S ====", flush=True)
        Q.train_variant(variant, LR, STEPS, suffix='S')

    losses = {v: np.load(f'{QK}/qk_tokenline_heldloss_{v}S.npy') for v in 'ABCD'}
    out = {v: {'held_ce': float(losses[v].mean())} for v in 'ABCD'}
    for v in 'BCD':
        d = losses[v] - losses['A']
        out[v]['minus_A'] = float(d.mean())
        out[v]['minus_A_se'] = float(d.std(ddof=1) / math.sqrt(len(d)))
        ds = d.reshape(len(Q.HELD), Q.T).mean(1)
        out[v]['minus_A_se_seq'] = float(ds.std(ddof=1) / math.sqrt(len(ds)))
    json.dump(out, open(f'{QK}/qk_tokenline_ce_short.json', 'w'), indent=2)
    print(json.dumps(out, indent=2), flush=True)
