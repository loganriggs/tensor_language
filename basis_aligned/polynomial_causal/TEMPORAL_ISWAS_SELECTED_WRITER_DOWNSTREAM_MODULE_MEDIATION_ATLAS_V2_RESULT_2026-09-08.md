# Selected is-was writer downstream full-module mediation atlas — 2026-09-08

## Verdict

The 16-module causal atlas is valid but returns `distributed_or_unstable_bypass`: no single
non-L11 downstream attention or MLP write reaches the frozen `.20` original-FIT recovery gate.
The result preserves the bypass and licenses a greedy combination rather than a lower singleton
threshold.

Result artifact:
`basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_selected_writer_downstream_module_mediation_atlas_v2_result.json`

SHA-256: `88537c2561098239472072b277ab29bef868cc1d648d672024f306fed814990b`.

## Strongest complete-module mediators

| module | FIT recovery / cosine / direction | HOLDOUT recovery / cosine / direction |
|---|---:|---:|
| A11 | `.1771 / .9820 / 1.0` | `.1670 / .9898 / 1.0` |
| M11 | `.1716 / .9347 / 1.0` | `.1683 / .9394 / 1.0` |
| M12 | `.0935 / .8322 / .844` | `.1050 / .8738 / .813` |
| M15 | `.0732 / .9663 / .969` | `.0833 / .9688 / .969` |
| M13 | `.0689 / .9134 / .969` | `.0785 / .9431 / .969` |
| M16 | `.0540 / .8346 / .938` | `.0628 / .9104 / .984` |

The A11 positive control passes and matches the known small aligned L11 branch. Writer replay is
exact, every one of the 16 attention/MLP outputs has the expected `128×27×1152` capture, and all
patched temporal-command effects are exactly zero. Predictions A/B pass. C fails because no
non-A11 module reaches `.20`; D/E are false because the frozen selector is empty, not because a
selected module failed validation or selectivity.

## Distributive successor

Original FIT singleton recovery fixes the full cumulative order before combination outcome:

`A11, M11, M12, M15, M13, M16, M10, A15, M14, A10, A13, A17, A16, A14, A12, M17`.

The additive projection predicts that P4 first crosses `.50` (`.5154` FIT; `.5237` HOLDOUT), but
module clamps at different depths need not add linearly. The successor will execute every nested
prefix, let original FIT select the smallest qualified one, and validate its frozen identity on
HOLDOUT. A pass establishes a distributed module set and licenses head splitting within selected
attention modules; a failure establishes that singleton-ranked module composition cannot close
the bypass at this grain.
