# The attention x feed-forward factorial

Every arm holds the body parameter count at the family's 18 W^2 + W per block. `bilinnorm` is DIAGNOSTIC ONLY -- its row denominator depends on every visible key, so it does not fold to a fixed token-pair table and is never a foldable win.

## depth 2 width 128 seed 0

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.65117 | -0.0138 | 0.0081 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | 4.54320 | +0.1558 | 0.0189 | published conventional baseline, parameter-matched arm |

## depth 2 width 128 seed 1

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.65005 | -0.0022 | 0.0147 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | -- | -- | -- | published conventional baseline, parameter-matched arm |

## depth 2 width 128 seed 2

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.63768 | +0.0059 | 0.0127 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | -- | -- | -- | published conventional baseline, parameter-matched arm |

## depth 3 width 64 seed 0

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.94174 | +0.0077 | 0.0115 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | 4.87913 | +0.1128 | 0.0073 | published conventional baseline, parameter-matched arm |

## depth 3 width 64 seed 1

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.94260 | +0.0033 | 0.0053 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | -- | -- | -- | published conventional baseline, parameter-matched arm |

## depth 3 width 64 seed 2

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.94062 | -0.0005 | 0.0116 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | -- | -- | -- | published conventional baseline, parameter-matched arm |

## depth 1 width 128 seed 0

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.81940 | -0.0276 | 0.0087 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | 4.74877 | -0.0321 | 0.0066 | published conventional baseline, parameter-matched arm |

## depth 1 width 128 seed 1

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.82524 | -0.0274 | 0.0064 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | -- | -- | -- | published conventional baseline, parameter-matched arm |

## depth 1 width 128 seed 2

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.82302 | -0.0242 | 0.0125 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | -- | -- | -- | published conventional baseline, parameter-matched arm |

## depth 2 width 256 seed 0

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.32661 | +0.0841 | 0.0088 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | 4.20009 | +0.4534 | 0.0320 | published conventional baseline, parameter-matched arm |

## depth 2 width 256 seed 1

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.32406 | +0.0965 | 0.0174 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | -- | -- | -- | published conventional baseline, parameter-matched arm |

## depth 2 width 256 seed 2

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.32546 | +0.1007 | 0.0103 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | -- | -- | -- | published conventional baseline, parameter-matched arm |

## depth 3 width 128 seed 0

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.52845 | +0.0974 | 0.0254 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | 4.42622 | +0.4662 | 0.0341 | published conventional baseline, parameter-matched arm |

## depth 3 width 128 seed 1

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.52707 | +0.1233 | 0.0149 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | -- | -- | -- | published conventional baseline, parameter-matched arm |

## depth 3 width 128 seed 2

| attention | feed-forward | held CE | induction | probe floor | source |
|---|---|---|---|---|---|
| two-branch product, unnormalised (ours, foldable) | ungated bilinear product (ours, foldable) | 4.52730 | +0.1048 | 0.0188 | published family model |
| two-branch product, unnormalised (ours, foldable) | GELU gate (conventional) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| the same product / its row L1 (diagnostic only) | GELU gate (conventional) | -- | -- | -- | new |
| softmax, one branch (conventional) | ungated bilinear product (ours, foldable) | -- | -- | -- | new |
| softmax, one branch (conventional) | GELU gate (conventional) | -- | -- | -- | published conventional baseline, parameter-matched arm |

## Predictions, scored

```
{
  "scored_at": "depth 2 width 128 seed 0",
  "available": true,
  "F5_negative_control": {
    "depth_1_scores": {
      "bilin+bilin": -0.027565214369032275,
      "softmax+gelu": -0.03214177025688976
    },
    "predicted": "no arm inducts at depth 1",
    "holds": true,
    "note": "a depth-1 arm above floor means the probe leaks, not that the arm is clever"
  }
}
```
