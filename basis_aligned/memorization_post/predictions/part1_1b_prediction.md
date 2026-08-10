# Pre-registered prediction for Part 1, section 1b (trained H=8 model)

REGISTRATION: this file must be git-committed BEFORE the 1b training stage runs
(`part1.py` enforces this with a git gate). The git commit time of this file is
the registration time.

Setup being predicted: bilinear layer y = D((Lx)*(Rx)), d=3 features
[furry, happy, whiskers], C=3 classes (Dog=furry&happy, Cat=furry&whiskers,
Catfish=whiskers&happy), H=8 (overcomplete), softmax CE, full-batch AdamW,
weight decay in {0, 1e-3, 1e-2}, 5 seeds, trained on the 3 single-class keys
plus the all-zeros "none" input with uniform target.

## Prediction (from the handoff, quoted)

For each class c with feature pair (a,b) and absent feature m, the trained
interaction matrix B_c shows:

1. POSITIVE off-diagonal on the class's feature pair: B_c[a,b] > 0.
2. PLUS negative structure involving the class's absent feature m — one or both of:
   (i) negative off-diagonals coupling m to a and/or b: B_c[a,m] < 0, B_c[b,m] < 0;
   (ii) negative DIAGONAL on m: B_c[m,m] < 0 — i.e. linear "not-whiskers"-style
   logic carried by the diagonal (x_m^2 = x_m on booleans).

To report: which channel SGD actually used ((i) off-diagonal, (ii) diagonal/linear,
or both), and whether the weight-decay setting {0, 1e-3, 1e-2} changes the channel.
