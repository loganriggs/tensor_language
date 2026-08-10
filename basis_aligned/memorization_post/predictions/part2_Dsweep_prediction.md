# D-to-minus-I sparsity sweep — registered prediction (commit time = registration time)

Logan's feedback on F9 panel iii: in his local runs D DOES go to -I under a sparsity
constraint — as one of many solutions, not always. Our single setting (L1 1e-3 on all
weights) found dominance 0.39-0.49 and mixed signs; that refutes D≈-I only at weak sparsity.

Sweep: L1 in {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}, H=40, 5 seeds each, same sgd_train as F9.
Metrics per run, gauge-aware: per-live-unit dominance max_c|D[c,h]| / ||D[:,h]||_2 (a unit is
live if ||D[:,h]|| > 1e-3 * max column norm), fraction of live units whose dominant entry is
negative, live-unit count, memorization accuracy.

Prediction (before running): dominance rises with L1 and negative-dominant fraction rises
toward ~1 at L1 >= 3e-2 in MOST but not all seeds (one-of-many-solutions: expect at least one
seed at high L1 stuck away from the -I family or dead); memorization holds at 100% through
1e-2 and may break at 1e-1. Confidence 0.6.
