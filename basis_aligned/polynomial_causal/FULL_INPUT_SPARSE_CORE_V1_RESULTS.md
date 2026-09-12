# Full-input sparse interaction result —12 September05:35 UTC

Folding through MLP16 concentrates edges more than the isolated last layer,
but neither fixed spectral frame is strongly sparse. Full input coverage does
not rescue the earlier best256edge result. All outputs remain in the objective.

| Centered coefficient capture | Last bilinear layer | MLP16 producer pullback |
|---|---:|---:|
| Best256edges | 1.4019% | 2.7123% |
| Best1024edges | 2.3889% | 4.4391% |
| Best4096edges | 4.5614% | 8.1839% |
| Edges needed for90% | 486,547 | 382,163 |
| Active readers among4096edges | 412 | 589 |

The relative improvement at4096edges is79.4%, exceeding the25% prediction;
the absolute50%capture prediction fails. Instrument identities hold to4.4e-15.
Managed V2 completed in11.53seconds. V1 never executed a native calculation:
preflight rejected keyword-style prediction keys; V2 only spells those keys
explicitly. [Raw result](FULL_INPUT_SPARSE_CORE_V2_RESULT.json),
[preregistration](FULL_INPUT_SPARSE_CORE_V1_PREREGISTRATION.md),
[CPU control](FULL_INPUT_SPARSE_CORE_V1_CONTROL.json).

The full-frame best256 result differs negligibly from the restricted128frame:
1.4019% versus1.4011% for the isolated layer,2.7123% versus2.7112% for the
producer pullback. Thus truncating inputs was a plausible explanation checked
and found insufficient for this particular sparse-edge failure. The complete
128input subcore holds6.406% and10.707%, respectively; that is not the same
quantity as energy in its best256edges.

Mathematically, each token's symmetric interaction matrix is transformed into
the complete orthogonal input-marginal eigenbasis. Diagonal entries and
sqrt(2)-scaled off-diagonal entries form orthonormal coordinates. Their squared
norms across the full centered unembedding give exact edge energies. Selecting
the largest is globally optimal **only within this fixed basis**. The producer
case changes the metric through the exact MLP16 quadratic Gram; it still uses
paired coefficients, not the fully symmetric quartic or FineWeb distribution.

Do not infer an arithmetic lower bound from the large90%edge count. The native
layer already has an exact4608product representation using two nonorthogonal,
potentially overcomplete reader banks. It costs more per product than a shared
orthogonal frame, but proves that these edge counts are representation-dependent.
No causal circuit or native extraction success follows from this screen.

Next action executed: [streamed frame optimizer](streamed_sparse_frame_v1.py)
reuses exact gradients and Armijo while streaming complete support selection.
CPU dense-selection error is0, orthogonality error5.83e-16, capture increases
0.40360→0.60474 over20updates with no decrease. This validates selection/ascent,
not global recovery. [Next preregistration](STREAMED_SPARSE_FRAME_V1_PREREGISTRATION.md)
tests the fixed-frame restriction with full inputs. Native wrapper remains to
be completed; no optimizer GPU run is represented as queued or converged.
