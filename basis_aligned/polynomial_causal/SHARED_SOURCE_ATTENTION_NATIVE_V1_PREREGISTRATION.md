# Native shared-source quadratic channels

Four fixed varimax forms0,1,4,2, selected previously by total loading energy.
Each is an exact normalized native output combination, not a separately fitted
attention function. Keep the selection, all nine heads and full2304-dimensional
source tuple fixed. No labels, sequences or model body forwards.

Use O17 and W_h=[(1-lambda17)V17_h,lambda17 V0_h] with checkpoint lambda.
The implemented low-rank contraction returns the full lifted tensor energy
and its separately symmetric/antisymmetric head/source parts. The latter
vanishes on one-source rank1 inputs but can contribute for multiple sources.
Do not delete it or equate coefficient energy with naturally occurring effects.

Controls: one common signed source-coordinate permutation preserves both norms;
independent signed permutations per head preserve every head Gram and total
energy while scrambling shared-source alignment. Seed33417, source permutations
and signs drawn on CPU, same controls for all four forms. This is a coordinate
null, not an alternate biological/statistical model of token inputs.

- A: existing dense controls pass; original Q norm1, symmetric+antisymmetric
  energy closure, nonnegativity, common-permutation component invariance and
  independent-permutation total-energy invariance errors<=1e-10.
- B: A plus antisymmetric fraction>=0.10 in every selected form. Tests whether
  unrestricted coefficient geometry counts substantial directions that vanish
  on one-source inputs. This is not sufficient to prefer a quotient model for
  full multi-source attention.
- C: A plus absolute difference between mean native and independently permuted
  antisymmetric fractions>=0.05. Otherwise large learned source-alignment gain
  is not demonstrated in this four-form screen. Report per-form differences.

FP64, managed GPU,900-second alarm; exact calculations with no optimization.
Persist only JSON; all native U/L/R/D/O/V, routing, histories, residual and norm
interfaces remain charged. The four forms cover a selected part of the prior
rank128 output projection, not a census of the full native tensor.

If B holds but C fails, the reduction may be generic shared-source algebra
rather than specially learned alignment. Next seek a compact cross-source
contrast description without throwing away its live antisymmetric channel.
If B fails, do not call all shared-source constraints negligible: scope is four
forms and this formal coefficient norm. Every negative gets the invariant/null
comparison recorded before structural interpretation.
