# Subject-number L11H3 three-port behavioral test V1

## Purpose

Test the missing behavioral properties of the frozen OOD three-port graph
`upstream_0_7 + mlp_8 + mlp_10` at the input to layer 11.

## Interventions

For each corrected-authority prompt, capture the exact base and embedding-number
removed decompositions immediately before L11 attention. A removal hybrid starts
from the native base state and swaps only the three frozen ports to their values
under embedding-number removal. A rescue hybrid starts from the fully removed
state and restores those same three ports. All other residual components, the
first-value bus, and the layer-recurrence anchor remain native to the receiving
run. The hybrid then runs through L11–L17 and the native vocabulary readout.

Run each of the three port swaps separately. Their independently measured
correct-vs-wrong logit-margin damages are summed with no fitted coefficient and
compared with the joint three-port damage. Run 16 equal-L2 random edits at the
same L11 subject site, orthogonalized per row to the target joint edit. Report
`can`-versus-`will` collateral.

The four frozen 32-row panels remain discovery, context OOD, lexical OOD, and
joint OOD. No selection or fitting occurs in this experiment.

## Registered gates

- Instrument: finite; native suffix replay at most `1e-5`; component closure at
  most `1e-10`; closed-port correction relative L2 at most `1e-5`; random edit
  norm mismatch at most `1e-5`; exact prices and checkpoint.
- Capability: native agreement accuracy at least `0.75` in every panel.
- Removal/OOD: joint damage RMS at least `0.10` and positive fraction at least
  `0.60` in every panel.
- Selectivity: overall target damage is at least twice the median equal-L2
  random-control damage, and `can`/`will` collateral RMS is at most half the
  target agreement damage RMS.
- Composition: the sum of the three individual damages has relative L2 at most
  `0.25`, cosine at least `0.90`, and positive aligned recovery in every panel.
- Rescue: three-port rescue of the fully removed run has cosine at least `0.50`
  with full embedding-removal damage and positive aligned recovery overall.

Passing licenses a three-native-port extracted behavioral graph on this
authority. It remains an intermediate-state extraction: the three port values
are supplied by native upstream computation, rather than predicted from tokens
by a standalone formula.
