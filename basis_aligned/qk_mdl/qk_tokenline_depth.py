"""TOKEN-LINE DEPTH TEST (follow-up to §106): is the embedding re-injection line's value
a DEPTH phenomenon? §106 found at depth 4 the line buys nothing on held CE (B-A =
+0.0058 +/- 0.0028 seq-clustered), yet bilin18 (18 blocks) gets 2.26 nats from its line.
Hypothesis: the token washes out of a deep accumulating stream but survives a shallow one,
so the line should help increasingly with depth.

DISCRIMINATING TEST: train variant A (vanilla residual, a=1 b=0) and variant B (clamped
line: a=1, learned per-block b, rms-normed embedding) at depths 2, 8, 12. Depth 4 reuses
the §106 6-epoch models qk_tokenline_{A,B}S.pt and qk_tokenline_ce_short.json numbers.
Everything else VERBATIM from qk_tokenline_train.py: width 384, 6 heads, head dim 64,
tied embeddings, vocab 50257, FineWeb cooc corpus train [0:5500] held [5500:6000],
T=512, batch 8, lr 0.001 (frozen from the §106 variant-A sweep), 6-epoch budget
(6 x 687 = 4122 steps), same seed/init/data order, AdamW wd 0.1, cosine decay,
warmup 250, grad clip 1.0. Sequential with GPU guard (<4500 MiB free -> sleep 20 s).

Memory check (measured, one fwd+bwd at batch 8): depth 12 peaks 6131 MiB torch-allocated
(depth 8: 5255, depth 4 historical: 4402) -- fits the < 7 GB budget, so batch stays 8 at
ALL depths (no reduction needed).

Saves qk_tldepth_{A,B}{2,8,12}.pt + qk_tldepth_heldloss_{V}{depth}.npy, paired CE table
(all four depths) appended into qk_tokenline_depth.json by the probe script _2.
"""
import json, math, os
import numpy as np
import qk_tokenline_train as Q

QK = Q.QK
DEPTHS = [2, 8, 12]
EPOCHS = 6
STEPS = EPOCHS * Q.STEPS_PER_EPOCH          # 4122, batch 8 at every depth

if __name__ == '__main__':
    LR = json.load(open(f'{QK}/qk_tokenline_lrsweep.json'))['chosen']
    print(f"depth run: depths {DEPTHS}, {STEPS} steps ({EPOCHS} epochs), lr {LR}, "
          f"batch {Q.BATCH} (depth 12 measured 6131 MiB peak -- fits)", flush=True)
    for depth in DEPTHS:
        Q.NL = depth                         # module global read by MiniBilin/make_model
        for variant in ['A', 'B']:
            target = f'{QK}/qk_tldepth_{variant}{depth}.pt'
            if os.path.exists(target):
                print(f"depth {depth} variant {variant} already trained -- skip", flush=True)
                continue
            print(f"==== training depth {depth} variant {variant} ====", flush=True)
            Q.train_variant(variant, LR, STEPS, suffix=f'_d{depth}tmp')
            os.replace(f'{QK}/qk_tokenline_{variant}_d{depth}tmp.pt', target)
            os.replace(f'{QK}/qk_tokenline_heldloss_{variant}_d{depth}tmp.npy',
                       f'{QK}/qk_tldepth_heldloss_{variant}{depth}.npy')

    # ---- paired per-token CE differences, all four depths ----
    out = {}
    for depth in [2, 4, 8, 12]:
        if depth == 4:
            la = np.load(f'{QK}/qk_tokenline_heldloss_AS.npy')
            lb = np.load(f'{QK}/qk_tokenline_heldloss_BS.npy')
        else:
            la = np.load(f'{QK}/qk_tldepth_heldloss_A{depth}.npy')
            lb = np.load(f'{QK}/qk_tldepth_heldloss_B{depth}.npy')
        d = lb - la
        ds = d.reshape(len(Q.HELD), Q.T).mean(1)
        out[str(depth)] = {
            'held_ce_A': float(la.mean()), 'held_ce_B': float(lb.mean()),
            'B_minus_A': float(d.mean()),
            'B_minus_A_se_token': float(d.std(ddof=1) / math.sqrt(len(d))),
            'B_minus_A_se_seq': float(ds.std(ddof=1) / math.sqrt(len(ds))),
        }
    json.dump(out, open(f'{QK}/qk_tldepth_ce.json', 'w'), indent=2)
    print(json.dumps(out, indent=2), flush=True)
