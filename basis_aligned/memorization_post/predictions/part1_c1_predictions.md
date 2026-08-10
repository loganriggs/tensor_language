# C1 pre-registered collateral predictions (Part 1)

Registration: the git commit time of this file (and the .json next to it) is the
registration time; the measurement stage is git-gated on it.

KKT-edit class c to uniform: minimal-norm rank-1 update to D along the stored key h_c=(Lz_c)*(Rz_c) so that logits(z_c) become equal (uniform softmax). Weights + the key only; no corpus.

Predicted collateral = |cosine(h_c, h_v)| — the (c,v) off-diagonal of the stored-key Gram matrix (F5). Computed from model weights BEFORE any edit is applied or measured.

Measured collateral = victim margin drop = margin_v(before) - margin_v(after) on the victim's key z_v, margin = true logit minus max other logit.

Prediction: predicted_collateral rank-correlates strongly with measured margin drop across all (edit, victim) pairs, seeds, and both regimes; undercomplete points have large Gram off-diagonals and large measured collateral, overcomplete points small on both.

| regime | seed | edit | victim | predicted |
|---|---|---|---|---|
| overcomplete | 0 | Dog | Cat | 0.0145 |
| overcomplete | 0 | Dog | Catfish | 0.1716 |
| overcomplete | 0 | Cat | Dog | 0.0145 |
| overcomplete | 0 | Cat | Catfish | 0.0794 |
| overcomplete | 0 | Catfish | Dog | 0.1716 |
| overcomplete | 0 | Catfish | Cat | 0.0794 |
| overcomplete | 1 | Dog | Cat | 0.1087 |
| overcomplete | 1 | Dog | Catfish | 0.1810 |
| overcomplete | 1 | Cat | Dog | 0.1087 |
| overcomplete | 1 | Cat | Catfish | 0.0077 |
| overcomplete | 1 | Catfish | Dog | 0.1810 |
| overcomplete | 1 | Catfish | Cat | 0.0077 |
| overcomplete | 2 | Dog | Cat | 0.2951 |
| overcomplete | 2 | Dog | Catfish | 0.4085 |
| overcomplete | 2 | Cat | Dog | 0.2951 |
| overcomplete | 2 | Cat | Catfish | 0.0564 |
| overcomplete | 2 | Catfish | Dog | 0.4085 |
| overcomplete | 2 | Catfish | Cat | 0.0564 |
| overcomplete | 3 | Dog | Cat | 0.1487 |
| overcomplete | 3 | Dog | Catfish | 0.1911 |
| overcomplete | 3 | Cat | Dog | 0.1487 |
| overcomplete | 3 | Cat | Catfish | 0.1772 |
| overcomplete | 3 | Catfish | Dog | 0.1911 |
| overcomplete | 3 | Catfish | Cat | 0.1772 |
| overcomplete | 4 | Dog | Cat | 0.2276 |
| overcomplete | 4 | Dog | Catfish | 0.1356 |
| overcomplete | 4 | Cat | Dog | 0.2276 |
| overcomplete | 4 | Cat | Catfish | 0.4913 |
| overcomplete | 4 | Catfish | Dog | 0.1356 |
| overcomplete | 4 | Catfish | Cat | 0.4913 |
| undercomplete | 0 | Dog | Cat | 0.3114 |
| undercomplete | 0 | Dog | Catfish | 0.0269 |
| undercomplete | 0 | Cat | Dog | 0.3114 |
| undercomplete | 0 | Cat | Catfish | 0.9583 |
| undercomplete | 0 | Catfish | Dog | 0.0269 |
| undercomplete | 0 | Catfish | Cat | 0.9583 |
| undercomplete | 1 | Dog | Cat | 0.6039 |
| undercomplete | 1 | Dog | Catfish | 0.4076 |
| undercomplete | 1 | Cat | Dog | 0.6039 |
| undercomplete | 1 | Cat | Catfish | 0.4817 |
| undercomplete | 1 | Catfish | Dog | 0.4076 |
| undercomplete | 1 | Catfish | Cat | 0.4817 |
| undercomplete | 2 | Dog | Cat | 0.2951 |
| undercomplete | 2 | Dog | Catfish | 0.4523 |
| undercomplete | 2 | Cat | Dog | 0.2951 |
| undercomplete | 2 | Cat | Catfish | 0.7187 |
| undercomplete | 2 | Catfish | Dog | 0.4523 |
| undercomplete | 2 | Catfish | Cat | 0.7187 |
| undercomplete | 3 | Dog | Cat | 0.8608 |
| undercomplete | 3 | Dog | Catfish | 0.5866 |
| undercomplete | 3 | Cat | Dog | 0.8608 |
| undercomplete | 3 | Cat | Catfish | 0.0928 |
| undercomplete | 3 | Catfish | Dog | 0.5866 |
| undercomplete | 3 | Catfish | Cat | 0.0928 |
| undercomplete | 4 | Dog | Cat | 0.7503 |
| undercomplete | 4 | Dog | Catfish | 0.1738 |
| undercomplete | 4 | Cat | Dog | 0.7503 |
| undercomplete | 4 | Cat | Catfish | 0.5206 |
| undercomplete | 4 | Catfish | Dog | 0.1738 |
| undercomplete | 4 | Catfish | Cat | 0.5206 |
