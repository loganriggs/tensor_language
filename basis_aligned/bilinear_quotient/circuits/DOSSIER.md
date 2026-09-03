# Circuit dossier — bilin18

Assembled from the frozen census (source note: 2026-08-30 by Claude, circuit task (Logan)) plus current version-2 records. **62 census response regions and 5 task-defined behavior circuits/shared subroutines**. Each census region was localised by two independent causal interventions over the 256,000-position census grid.

`concentration` = mean|dCE| on the circuit's members / mean|dCE| off its slice, when the named component is ablated. **mean** replaces the component output with its grid mean; **interchange** replaces it with its output at a random other position (seed 20260830).

Sources: `circuits/BATTERY.json` (localisation), `circuits/DAS.json` (learned subspace, where run), and each circuit's own file (story, examples, certification). Nothing here is recomputed.


## Behavior circuits, shared subroutines, and counterfactual identification

These version-2 records are task-defined behaviors or cross-module subroutines, not assumed aliases of census leaves. Their events include failed/null/invalid evidence so the same causal question is not silently repeated.

| circuit | kind | status | declared variable | families | negative events | next missing evidence |
|---|---|---|---|---:|---:|---|
| `subroutine.induction.equality_score` | shared_subroutine | site_live | `cross_head_equality_score` | 5 | 5 | materialize the text-edit and matched-natural answer-changing families plus the payload-preserving invariance family; then measure complete-state query/key/MLP7 ceilings with identical patch semantics before fitting a shared subspace |
| `task.bracket.pending_opener` | behavior_circuit | specified | `pending_opener_state` | 5 | 4 | run FIT/SELECT-only balanced four-closer capability and complete-state site ceilings; then compare ordinary and endpoint-readout-deflated contrastive DAS without opening FINAL_TEST/OOD |
| `task.increment.state` | behavior_circuit | proposed | `increment_state` | 4 | 0 | freeze cross-format rows; require number-word transfer and nonincrement numeric controls |
| `task.induction.selector_payload` | behavior_circuit | proposed | `induction_selector_payload` | 5 | 1 | freeze two-valid-source and payload-swap rows; measure selector and value site ceilings |
| `task.successor.pointer` | behavior_circuit | proposed | `successor_pointer_state` | 4 | 2 | expand families and test shared-plus-private projectors against failed cross-family transfer |

### `subroutine.induction.equality_score` — site_live

**Read:** whether the current token matches a token at each earlier position. **Operation:** construct an attention score pattern over matching earlier positions independently of the copied payload. **Write:** a score pattern that can be combined with L8H4's value/output payload and read by MLP9. **Endpoint:** signed causal recovery of L8H4's copy-related effect and donor-answer logit movement.

| family | role | status |
|---|---|---|
| `cross_head_score_swap` | interchange | validated |
| `text_match_pattern_edit_payload_fixed` | interchange | proposed |
| `matched_natural_whole_state_swap` | interchange | proposed |
| `payload_swap_match_preserved` | invariance | proposed |
| `match_break_answer_fixed` | necessity | frozen |

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `equality_score_swap.r459.natural.v1` | complete | composition | **held** | active | `r459_result` |
| `equality_score_swap.r460.code.v1` | complete | ood | **failed** | active | `r460_result` |
| `equality_downstream_gate.r462.null.v1` | complete | composition | **null** | active | `r462_result` |
| `equality_source_correction.r464.v1` | complete | composition | **held** | active | `r464_result` |
| `equality_action_quotient.r498.null.v1` | complete | cross_family_transfer | **null** | active | `r498_result` |
| `equality_mlp9_reader.r500.v1` | complete | composition | **held** | active | `r500_result` |
| `equality_factor_branch_sharing.r531.null.v1` | complete | null_control | **null** | active | `r531_result` |
| `terminal_copy_four_head_removal.collateral_failure.v1` | complete | removal | **failed** | active | `terminal_copy_negative_receipt` |

**Frozen artifacts:** 21. Paths and SHA-256 hashes are in the canonical JSON record.

**Next:** materialize the text-edit and matched-natural answer-changing families plus the payload-preserving invariance family; then measure complete-state query/key/MLP7 ceilings with identical patch semantics before fitting a shared subspace

### `task.bracket.pending_opener` — specified

**Read:** opener, closer, ordering, and recency evidence in the preceding context. **Operation:** maintain which opener type remains pending after completed earlier spans. **Write:** signed evidence for the matching closer token. **Endpoint:** symmetric donor-closer versus base-closer final-logit margin.

| family | role | status |
|---|---|---|
| `direct_four_closer_type_substitution` | interchange | frozen |
| `completed_then_reopened_four_closer_order` | interchange | frozen |
| `pending_type_preserved_surface_paraphrase` | invariance | frozen |
| `pending_type_preserved_distance_shift` | invariance | frozen |
| `pending_type_preserved_nonopener_punctuation` | invariance | frozen |

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `pending_opener_rank4_das.legacy.v1` | complete | das_interchange | **held** | active | `das_result` |
| `pending_opener_capability.r537.preregistered.v1` | preregistered | capability | **inconclusive** | superseded by `pending_opener_capability.r537.complete.v1` | `—` |
| `pending_opener_common_site_ceiling.r537.preregistered.v1` | preregistered | full_swap_ceiling | **inconclusive** | superseded by `pending_opener_common_site_ceiling.r538.invalid_unverified_checkpoint.v1` | `—` |
| `pending_opener_capability.r537.complete.v1` | complete | capability | **held** | active | `r537_capability_result` |
| `pending_opener_common_site_ceiling.r538.invalid_unverified_checkpoint.v1` | invalid | full_swap_ceiling | **invalid** | superseded by `pending_opener_common_site_ceiling.r538.complete.v2` | `r538_site_invalid_unverified_checkpoint_result` |
| `pending_opener_common_site_ceiling.r538.complete.v2` | complete | full_swap_ceiling | **held** | active | `r538_site_result_v2` |
| `pending_opener_control_ceilings.r539.preregistered.v1` | preregistered | null_control | **inconclusive** | superseded by `pending_opener_control_ceilings.r539.complete.v1` | `—` |
| `pending_opener_control_ceilings.r539.complete.v1` | complete | null_control | **held** | active | `r539_control_result` |
| `pending_opener_cross_family_das.r540.preregistered.v1` | preregistered | cross_family_transfer | **inconclusive** | superseded by `pending_opener_cross_family_das.r540.complete.v1` | `—` |
| `pending_opener_cross_family_das.r540.complete.v1` | complete | cross_family_transfer | **null** | active | `r540_das_result` |
| `pending_opener_split_integrity.r542.invalid_statistical_unit.v1` | complete | seed_stability | **invalid** | active | `r542_split_integrity_result` |
| `pending_opener_rows.r543.v1.invalid_unbalanced_delimiter_pairs` | invalid | null_control | **invalid** | active | `r543_unique_rows_receipt` |

**Frozen artifacts:** 43. Paths and SHA-256 hashes are in the canonical JSON record.

**Next:** run FIT/SELECT-only balanced four-closer capability and complete-state site ceilings; then compare ordinary and endpoint-readout-deflated contrastive DAS without opening FINAL_TEST/OOD

### `task.increment.state` — proposed

**Read:** recent numeric state and list relation. **Operation:** apply an increment relation to the numeric state. **Write:** evidence for the next numeric token. **Endpoint:** signed shifted-next-number minus base-next-number logit margin.

| family | role | status |
|---|---|---|
| `coherent_constant_shift` | interchange | proposed |
| `cross_format_operation_swap` | interchange | proposed |
| `incoherent_one_number_edit` | necessity | proposed |
| `operation_preserved_surface_edit` | invariance | proposed |

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `increment_postattn_rank4_das.legacy.v1` | complete | das_interchange | **held** | active | `postattn_result` |

**Frozen artifacts:** 5. Paths and SHA-256 hashes are in the canonical JSON record.

**Next:** freeze cross-format rows; require number-word transfer and nonincrement numeric controls

### `task.induction.selector_payload` — proposed

**Read:** token equality and source position. **Operation:** select a matching earlier source, then transport its following payload. **Write:** selector-dependent payload contribution to the target-token logits. **Endpoint:** signed donor-answer minus base-answer logit margin at the query continuation.

| family | role | status |
|---|---|---|
| `two_valid_sources_selector_swap` | interchange | proposed |
| `payload_swap_match_preserved` | interchange | proposed |
| `natural_pair_interchange` | interchange | proposed |
| `match_break_payload_preserved` | necessity | proposed |
| `copy_relation_preserved_nuisance_change` | invariance | proposed |

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `induction_terminal_collateral_failure.legacy.v1` | complete | removal | **failed** | active | `campaign_report` |

**Frozen artifacts:** 5. Paths and SHA-256 hashes are in the canonical JSON record.

**Next:** freeze two-valid-source and payload-swap rows; measure selector and value site ceilings

### `task.successor.pointer` — proposed

**Read:** the final sequence element and coherence/family context. **Operation:** use an identity pointer to retrieve the next element. **Write:** evidence for the successor token. **Endpoint:** signed donor-successor minus base-successor logit margin.

| family | role | status |
|---|---|---|
| `same_family_last_element_swap` | interchange | proposed |
| `coherent_whole_sequence_shift` | interchange | proposed |
| `internal_pointer_imposition` | interchange | proposed |
| `prefix_change_final_pointer_preserved` | invariance | proposed |

**Append-only evidence ledger:**
| event | stage | test | verdict | lifecycle | result artifact |
|---|---|---|---|---|---|
| `successor_cross_family_transfer.legacy.v1` | complete | cross_family_transfer | **failed** | active | `task_report` |
| `successor_layer8_input_ceiling.legacy.v1` | complete | full_swap_ceiling | **null** | active | `task_report` |

**Frozen artifacts:** 5. Paths and SHA-256 hashes are in the canonical JSON record.

**Next:** expand families and test shared-plus-private projectors against failed cross-family transfer


## Summary table

| # | circuit | best (mean) | conc | best (interchange) | conc | agree | members |
|---|---------|-------------|------|--------------------|------|-------|---------|
| 1 | `r.2.0.0` | a8 | 12.28 | a8 | 9.603 | both | 864 / 5,760 |
| 2 | `r.3.0.0` | a16 | 11.87 | a16 | 8.888 | both | 864 / 5,760 |
| 3 | `r.3.0` | a16 | 9.21 | a16 | 6.999 | both | 5,760 / 38,400 |
| 4 | `r.2.0.2` | a8 | 8.99 | a8 | 7.144 | both | 864 / 5,760 |
| 5 | `r.3.0.2` | a16 | 8.56 | a16 | 6.515 | both | 864 / 5,760 |
| 6 | `r.2.0` | a8 | 8.35 | a8 | 6.437 | both | 5,760 / 38,400 |
| 7 | `r.2.0.1` | a8 | 8.35 | a8 | 6.614 | both | 864 / 5,760 |
| 8 | `r.3.0.1` | a16 | 7.57 | a16 | 5.929 | both | 864 / 5,760 |
| 9 | `r.2.1.1` | a8 | 6.16 | a8 | 5.124 | both | 864 / 5,760 |
| 10 | `r.1.2.0` | m16 | 6.12 | m15 | 4.975 | rows-only | 864 / 5,760 |
| 11 | `r.1.0.0` | a16 | 6.06 | a16 | 5.064 | both | 864 / 5,760 |
| 12 | `r.2.2.1` | a8 | 5.93 | a8 | 5.067 | both | 864 / 5,760 |
| 13 | `r.4.1.1` | a16 | 5.47 | a16 | 4.529 | both | 864 / 5,760 |
| 14 | `r.0.0.0` | a3 | 5.07 | a3 | 4.558 | both | 864 / 5,760 |
| 15 | `r.0.0` | a4 | 4.95 | a3 | 4.459 | rows-only | 5,760 / 38,400 |
| 16 | `r.23.2.3` | a8 | 4.93 | a8 | 3.971 | both | 864 / 5,760 |
| 17 | `r.11.1.2` | a8 | 4.90 | a8 | 4.19 | both | 864 / 5,760 |
| 18 | `r.3.1.1` | a17 | 4.80 | a17 | 4.003 | both | 864 / 5,760 |
| 19 | `r.0.0.1` | a3 | 4.60 | a3 | 4.194 | methods-only | 864 / 5,760 |
| 20 | `r.1.2.1` | m16 | 4.51 | m15 | 4.268 | rows-only | 864 / 5,760 |
| 21 | `r.2.2.2` | a6 | 4.37 | a6 | 3.806 | both | 864 / 5,760 |
| 22 | `r.1.2` | m16 | 4.37 | m15 | 3.839 | neither | 5,760 / 38,400 |
| 23 | `r.11.3.1` | a8 | 4.31 | a8 | 3.628 | both | 864 / 5,760 |
| 24 | `r.4.1.0` | a16 | 4.22 | a16 | 3.618 | both | 864 / 5,760 |
| 25 | `r.1.0` | m14 | 4.22 | m14 | 3.308 | both | 5,760 / 38,400 |
| 26 | `r.2.3` | a8 | 4.18 | a8 | 3.479 | both | 5,760 / 38,400 |
| 27 | `r.1.0.2` | m14 | 4.14 | m14 | 3.191 | methods-only | 864 / 5,760 |
| 28 | `r.2.1` | a8 | 4.13 | a8 | 3.401 | both | 5,760 / 38,400 |
| 29 | `r.6.3.0` | a16 | 4.11 | a14 | 3.559 | neither | 864 / 5,760 |
| 30 | `r.1.3.1` | m13 | 4.10 | m14 | 3.485 | neither | 864 / 5,760 |
| 31 | `r.1.1.2` | m16 | 4.08 | m15 | 3.791 | rows-only | 864 / 5,760 |
| 32 | `r.1.1.1` | m16 | 4.08 | m14 | 3.828 | neither | 864 / 5,760 |
| 33 | `r.2.2` | a8 | 4.05 | a8 | 3.371 | both | 5,760 / 38,400 |
| 34 | `r.1.0.3` | m14 | 4.00 | m14 | 3.227 | both | 864 / 5,760 |
| 35 | `r.2.1.0` | a8 | 3.99 | a8 | 3.385 | both | 864 / 5,760 |
| 36 | `r.11.1.1` | a8 | 3.98 | a8 | 3.488 | both | 864 / 5,760 |
| 37 | `r.6.0.1` | a14 | 3.97 | a14 | 3.93 | methods-only | 864 / 5,760 |
| 38 | `r.1.1.0` | m14 | 3.96 | m14 | 3.536 | methods-only | 864 / 5,760 |
| 39 | `r.0.3.0` | a3 | 3.91 | a3 | 3.484 | methods-only | 864 / 5,760 |
| 40 | `r.6.1.0` | m5 | 3.84 | a14 | 3.062 | rows-only | 864 / 5,760 |
| 41 | `r.1.1` | m13 | 3.81 | m14 | 3.357 | rows-only | 5,760 / 38,400 |
| 42 | `r.1.3` | m13 | 3.77 | m13 | 3.298 | both | 5,760 / 38,400 |
| 43 | `r.6.0.3` | m17 | 3.65 | m16 | 3.101 | rows-only | 864 / 5,760 |
| 44 | `r.6.2.2` | m16 | 3.63 | m16 | 3.258 | both | 864 / 5,760 |
| 45 | `r.3.1.0` | a2 | 3.63 | a16 | 3.225 | neither | 864 / 5,760 |
| 46 | `r.1.0.1` | m14 | 3.60 | a16 | 2.969 | neither | 864 / 5,760 |
| 47 | `r.6.2.1` | a9 | 3.59 | a9 | 2.88 | both | 864 / 5,760 |
| 48 | `r.2.2.0` | a8 | 3.55 | a8 | 3.099 | both | 864 / 5,760 |
| 49 | `r.6.1.1` | a16 | 3.53 | a16 | 3.063 | methods-only | 864 / 5,760 |
| 50 | `r.8.1.0` | a3 | 3.50 | a3 | 3.13 | both | 864 / 5,760 |
| 51 | `r.5.0.1` | a16 | 3.40 | a16 | 3.034 | methods-only | 864 / 5,760 |
| 52 | `r.23.2.1` | a8 | 3.38 | a8 | 2.908 | both | 864 / 5,760 |
| 53 | `r.5.3.1` | a15 | 3.33 | a15 | 2.947 | both | 864 / 5,760 |
| 54 | `r.1.3.0` | m13 | 3.23 | a16 | 2.703 | rows-only | 864 / 5,760 |
| 55 | `r.6.0.2` | a9 | 3.19 | a16 | 2.892 | rows-only | 864 / 5,760 |
| 56 | `r.6.2.0` | a16 | 3.17 | a16 | 2.965 | methods-only | 864 / 5,760 |
| 57 | `r.6.3.1` | a16 | 3.13 | a16 | 2.802 | both | 864 / 5,760 |
| 58 | `r.18.2.0` | a7 | 3.10 | a7 | 2.763 | both | 864 / 5,760 |
| 59 | `r.13.2.1` | a3 | 3.04 | a7 | 2.771 | neither | 864 / 5,760 |
| 60 | `r.7.1.1` | a7 | 2.98 | a7 | 2.679 | methods-only | 864 / 5,760 |
| 61 | `r.6.0.0` | a16 | 2.88 | a16 | 2.633 | both | 864 / 5,760 |
| 62 | `r.6.2.3` | a9 | 2.61 | a17 | 2.661 | neither | 864 / 5,760 |

## Per-circuit detail


### 1. `r.2.0.0` — a8, concentration 12.28

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 12.282 | a14 4.732 | a6 4.145 |
| interchange | a8 9.603 | a6 4.164 | a14 4.06 |

At `a8`: mean|dCE| on members **1.687**, off slice 0.1374, signed dCE on members 0.4605. Second-best component is `a14` at 4.732 — a 2.60x margin.


**DAS (rank 1, held-out):** member dCE 0.2345, concentration 13.045, recovers 0.125 of the full component; overlap with the closed-form direction 0.286.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\n#### Ramsgate\n\n#### Sandwich\n\n#### Dover` → `\n`  (dCE -25.41, base CE 29.76)
- `…life\n\n#### Entertainment\n\n#### Shopping\n\n#### Information` → `\n`  (dCE -27.55, base CE 32.04)
- `…life\n\n7PurlB3\n\n7Shopping` → `\n`  (dCE -21.07, base CE 24.34)


### 2. `r.3.0.0` — a16, concentration 11.87

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 11.866 | a14 6.816 | a15 6.271 |
| interchange | a16 8.888 | a14 5.583 | a15 4.712 |

At `a16`: mean|dCE| on members **0.7896**, off slice 0.0665, signed dCE on members 0.4652. Second-best component is `a14` at 6.816 — a 1.74x margin.


**DAS (rank 1, held-out):** member dCE 0.0824, concentration 10.234, recovers 0.105 of the full component; overlap with the closed-form direction 0.133.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT class_subword', 'NOT class_comma']], 'program_bacc': 0.748, 'program_null': 0.597, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE -20.72, base CE 22.84)
- `…\n\n#### Dover\n\n#### Around Dover\n\n#### Rye` → `\n`  (dCE -21.62, base CE 29.5)
- `…\n#### London Highlights\n\n#### History\n\n#### Sights` → `\n`  (dCE -20.86, base CE 24.86)


### 3. `r.3.0` — a16, concentration 9.21

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 9.211 | a14 6.957 | a15 5.347 |
| interchange | a16 6.999 | a14 5.731 | a15 4.115 |

At `a16`: mean|dCE| on members **0.5423**, off slice 0.0589, signed dCE on members 0.3359. Second-best component is `a14` at 6.957 — a 1.32x margin.


**DAS (rank 1, held-out):** member dCE 0.0539, concentration 6.557, recovers 0.094 of the full component; overlap with the closed-form direction 0.336.


**Story (from the circuit file):** {'mechanism_level': 'computational', 'mechanism': {'code': 'heads (16,8),(16,2): z_h(q)=sum over k in top8 |pat_h(q,:)| of pat_h(q,k)*vm_h(k)', 'replication': {'k1': 0.2089, 'k8': 0.1073, 'deletion': 0.5052, 'shuffled_control': 0.358}, 'notes': '79% of named-head function at k=8; a6/a7 bundles not yet coded'}}


### 4. `r.2.0.2` — a8, concentration 8.99

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 8.987 | a6 4.107 | a14 3.823 |
| interchange | a8 7.144 | a6 3.905 | a14 3.407 |

At `a8`: mean|dCE| on members **1.2345**, off slice 0.1374, signed dCE on members 0.269. Second-best component is `a6` at 4.107 — a 2.19x margin.


**DAS (rank 1, held-out):** member dCE 0.1319, concentration 8.301, recovers 0.094 of the full component; overlap with the closed-form direction 0.223.


**Story (from the circuit file):** {'blind_name': 'input-enrichment for a0 (layer-0 attn) into the a8/a6 double-QK bundle; no surviving behavioral story', 'program': [['NOT class_other', 'NOT prev1_class_other']], 'program_bacc': 0.693, 'program_null': 0.436, 'mechanism_level': 'none', 'mechanism': {'components': ['a8', 'a6'], 'per_component_top_writer': {'a8': {'writer': 'a0', 'ratio': 1.464, 'null_top_ratio': 1.2, 'ENRICHED': True, 'BEATS_NULL': True}, 'a6': {'writer': 'a0', 'ratio': 1.475, 'null_top_ratio': 1.2, 'ENRICHED': True, 'BEATS_NULL': True}}, 'mechanism_line': "a8 and a6 (r.2.0 double-QK bundle) both ENRICHED for writer a0 (layer-0 attn) in their input: ratio 1.464/1.475 vs null 1.20 (BEATS_NULL); a0's absolute share is tiny (~0.0007) vs m0's dominant ~0.42-0.44 (ratio ~0.99, not selective).", 'escalation_status': 'NOT ESCALATED: ENRICHED writer (a0) named but qk_writer_decomp-style writer-pair QK decomposition not run (out of scope for this pass); flagged for ladder follow-up. Note a0 share of input is small in absolute terms (~0.0007) despite ratio enrichment.', 'leaf_mech_file': 'leaf_mech/r.2.0.2.json'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `… Braymer. The New Zealand rabbits were purchased by the Missouri State` → ` Rabbit`  (dCE 0.11, base CE 11.4)
- `…–]Subhazard 48 Punkte49 Punkte50 Punkte` → ` 9`  (dCE -0.51, base CE 5.52)
- `…. But, eventually hiding wasn’t enough and H.` → ` erect`  (dCE -2.83, base CE 5.41)


### 5. `r.3.0.2` — a16, concentration 8.56

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 8.558 | a14 5.762 | a15 5.396 |
| interchange | a16 6.515 | a14 4.816 | a15 4.179 |

At `a16`: mean|dCE| on members **0.5694**, off slice 0.0665, signed dCE on members 0.3292. Second-best component is `a14` at 5.762 — a 1.49x margin.


**DAS (rank 1, held-out):** member dCE 0.0251, concentration 4.628, recovers 0.041 of the full component; overlap with the closed-form direction 0.094.


**Story (from the circuit file):** {'blind_name': 'attn-input-writer decomp: a14 dominant writer into a15/16/17', 'mechanism_level': 'none', 'program': [['class_ind']], 'program_bacc': 0.613, 'program_null': 0.593, 'mechanism': 'a15/a16/a17 attn (r.3.0 PCA bundles, incl. layer-16); top writer a14 enriched 2.85/2.96/2.84x (null ~1.1-1.25) into all three; a16 also feeds a17 (2.58x)', 'behavior': 'no behavioral claim survives base-rate testing (capitalized/space_word/subword/punct/newline all fail; digit nominally passes but n=3, whole population, not credible)'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… services and hosting globally," said Henderson, CTO and president of` → ` academic`  (dCE -0.0, base CE 10.92)
- `…-collected fishery data, including fishery data held by` → ` RF`  (dCE 0.65, base CE 7.61)
- `…rdinator will have overall responsibility for the technical quality of` → ` Merlin`  (dCE -0.24, base CE 8.32)


### 6. `r.2.0` — a8, concentration 8.35

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 8.347 | a6 3.662 | a14 3.612 |
| interchange | a8 6.437 | a6 3.438 | a14 3.187 |

At `a8`: mean|dCE| on members **0.9828**, off slice 0.1177, signed dCE on members 0.3151. Second-best component is `a6` at 3.662 — a 2.28x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\n#### Dover\n\n#### Around Dover\n\n#### Rye` → `\n`  (dCE -27.48, base CE 29.5)
- `…\n\n#### Ramsgate\n\n#### Sandwich\n\n#### Dover` → `\n`  (dCE -27.14, base CE 29.76)
- `…life\n\n7PurlB3\n\n7Shopping` → `\n`  (dCE -19.73, base CE 24.34)


### 7. `r.2.0.1` — a8, concentration 8.35

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 8.346 | a6 5.437 | a14 3.753 |
| interchange | a8 6.614 | a6 4.741 | a14 3.452 |

At `a8`: mean|dCE| on members **1.1464**, off slice 0.1374, signed dCE on members 0.1527. Second-best component is `a6` at 5.437 — a 1.54x margin.


**Story (from the circuit file):** {'blind_name': 'r.2.0 double-QK bundle (a6/a8); input writer composition NOT enriched (ENRICHED_STABLE=False both); no surviving behavioral story', 'program': [['NOT class_other', 'NOT prev1_class_other']], 'program_bacc': 0.682, 'program_null': 0.561, 'mechanism_level': 'none', 'mechanism_line': "a6 and a8 (r.2.0 double-QK bundle) input writer composition does NOT distinguish members under 5-draw bootstrap: top writer both components is a4 (a6: mean 1.134 [1.063-1.212]; a8: mean 1.132 [1.063-1.211]), ENRICHED_STABLE=False for both (single-draw ENRICHED was already False, so no a0-style collapse to retract). Dominant writer by input mass is m0 (~0.43-0.45 share) but its ratio is flat (~0.986-0.987, not selective) -- consistent with the universal identity code, not this leaf's mechanism.", 'behavior_line': 'no behavioral claim survives base-rate testing (12 class x direction pairs tested, alpha=0.10/12=0.0083, none robust)', 'leaf_mech_file': 'leaf_mech/r.2.0.1.json'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n|College Teams||High Schools|\nMarch 4, 2012` → `Tweet`  (dCE -0.49, base CE 12.39)
- `…\n|College Teams||High Schools|\nJanuary 19, 2012` → `1`  (dCE 0.78, base CE 14.49)
- `…\n|College Teams||High Schools|\nMay 24, 2012` → `A`  (dCE -1.67, base CE 10.85)


### 8. `r.3.0.1` — a16, concentration 7.57

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 7.569 | a14 4.89 | a17 4.807 |
| interchange | a16 5.929 | a14 4.242 | a17 4.193 |

At `a16`: mean|dCE| on members **0.5036**, off slice 0.0665, signed dCE on members 0.2977. Second-best component is `a14` at 4.89 — a 1.55x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\n#### Woburn\n\n#### Waddesdon` → `\n`  (dCE -2.51, base CE 16.09)
- `…#### Margate\n\n#### Broadstairs\n\n#### Ramsgate` → `\n`  (dCE -1.42, base CE 19.99)
- `…#### Festivals & Events\n\n#### Sleeping\n\n#### Eating` → `\n`  (dCE -2.28, base CE 22.08)


### 9. `r.2.1.1` — a8, concentration 6.16

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 6.156 | a14 4.029 | a6 3.837 |
| interchange | a8 5.124 | a14 3.664 | a6 3.58 |

At `a8`: mean|dCE| on members **0.916**, off slice 0.1488, signed dCE on members 0.1477. Second-best component is `a14` at 4.029 — a 1.53x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…designed by John Nash). From Pall Mall, walk along Cocks` → `pur`  (dCE -0.48, base CE 2.21)
- `… Frith St, W1; s £216, d/` → `ste`  (dCE -0.9, base CE 15.8)
- `… ; www.shakespearesglobe.com; 21 New` → ` Globe`  (dCE -1.51, base CE 7.83)


### 10. `r.1.2.0` — m16, concentration 6.12

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 5.6996, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 6.119 | m15 4.933 | m13 4.888 |
| interchange | m15 4.975 | m16 4.728 | m13 4.688 |

At `m16`: mean|dCE| on members **1.8821**, off slice 0.3076, signed dCE on members 1.4723. Second-best component is `m15` at 4.933 — a 1.24x margin.


**Story (from the circuit file):** {'blind_name': 'Members are hard-to-predict tokens in venue/address listings; machinery hurts prediction of common words (off, d) but helps rarer proper-noun/genre completions (house, Banqueting, Queen).', 'program': [['class_subword', 'NOT is_newline'], ['mid_word']], 'program_bacc': 0.542, 'program_null': 0.439, 'mechanism_level': 'none', 'redteam_hits': '3/3', 'redteam_verdict': 'HELD'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… Lane, Cowcross St, EC1; s £222,` → ` d`  (dCE -0.82, base CE 8.3)
- `… most popular villages are busy in summer, it's easy to get` → ` off`  (dCE -1.03, base CE 5.37)
- `… Rd Arches)\n\nSeafront club hosting indie,` → ` house`  (dCE 0.84, base CE 4.94)


### 11. `r.1.0.0` — a16, concentration 6.06

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 6.064 | a14 4.6 | m14 4.427 |
| interchange | a16 5.064 | a14 4.322 | m14 3.213 |

At `a16`: mean|dCE| on members **0.4508**, off slice 0.0743, signed dCE on members 0.2994. Second-best component is `a14` at 4.6 — a 1.32x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT prev1_starts_space', 'NOT class_subword', 'NOT prev1_class_ind'], ['circ_r_3_1_0']], 'program_bacc': 0.721, 'program_null': 0.675, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…eth that sit closest to the river.\n\nSouth Bank\n` → `\n`  (dCE 6.27, base CE 22.8)
- `… lanes leading off it.\n\nTrulloITALIAN\n` → `\n`  (dCE 5.43, base CE 21.6)
- `…\n2130 St Mary Axe\n\n22Heron Tower\n` → `\n`  (dCE 6.31, base CE 20.18)


### 12. `r.2.2.1` — a8, concentration 5.93

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 5.933 | a6 4.529 | a14 3.686 |
| interchange | a8 5.067 | a6 4.191 | a14 3.409 |

At `a8`: mean|dCE| on members **0.8839**, off slice 0.149, signed dCE on members 0.1215. Second-best component is `a6` at 4.529 — a 1.31x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… NW8; tours adult/child £18/12; ` → ` t`  (dCE -0.24, base CE 13.86)
- `…1Sights\n\n1Madame TussaudsB` → `1`  (dCE 0.25, base CE 8.17)
- `…rant preserve.\n\n8Information\n\nTourist OfficeT` → `OUR`  (dCE 1.83, base CE 9.1)


### 13. `r.4.1.1` — a16, concentration 5.47

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 5.466 | a14 4.202 | a12 3.986 |
| interchange | a16 4.529 | a14 3.704 | a12 3.332 |

At `a16`: mean|dCE| on members **0.3993**, off slice 0.0731, signed dCE on members 0.2903. Second-best component is `a14` at 4.202 — a 1.30x margin.


**Story (from the circuit file):** {'blind_name': 'No strong single-writer mechanism found (a12/m1 top ratios 1.06/1.01 vs thresholds 1.31/1.30); no behavioral claim survives base-rate testing (12 class/direction pairs).', 'program': [['class_ind']], 'program_bacc': 0.559, 'program_null': 0.512, 'mechanism_level': 'none', 'behavior_line': 'no behavioral claim survives base-rate testing', 'story_test_sweep': {'n_tests': 12, 'alpha': 0.0083, 'kinds_x_directions': 'subword,space_word,digit,punct,capitalized,newline x {help,hurt}', 'best_seed_pass_frac': 0.4, 'best_population_p': 0.1135, 'all_ROBUST_V2': False}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…umi in Japan. Later, yields reduced and traditional whaling w` → `aned`  (dCE 2.01, base CE 2.81)
- `…s challenge, however, will be convincing its own investors that the` → ` African`  (dCE 1.36, base CE 6.15)
- `… addition, Nagle said, the district must find two classrooms for` → ` Leh`  (dCE 0.95, base CE 5.76)


### 14. `r.0.0.0` — a3, concentration 5.07

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a3 5.072 | m5 4.882 | a4 4.763 |
| interchange | a3 4.558 | a2 4.027 | m5 3.996 |

At `a3`: mean|dCE| on members **1.51**, off slice 0.2977, signed dCE on members 0.1555. Second-best component is `m5` at 4.882 — a 1.04x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['circ_r_1_1_2', 'NOT is_punct'], ['NOT class_other', 'NOT class_ind', 'NOT class_digit']], 'program_bacc': 0.744, 'program_null': 0.516, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…3\n\n34RevengeF4\n\n3Entertainment` → `\n`  (dCE -10.6, base CE 29.94)
- `…\n\n#### Arundel\n\n#### Chichester` → `\n`  (dCE -12.46, base CE 29.36)
- `… Chichester\n\n#### Farnham\n\n#### Hindhead` → `\n`  (dCE -15.01, base CE 27.0)


### 15. `r.0.0` — a4, concentration 4.95

5,760 member positions in a slice of 38,400 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 5.0709, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a4 4.953 | m6 4.791 | m5 4.779 |
| interchange | a3 4.459 | a2 4.02 | m5 4.011 |

At `a4`: mean|dCE| on members **1.7276**, off slice 0.3488, signed dCE on members 0.141. Second-best component is `m6` at 4.791 — a 1.03x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n44Pitt Cue CoB4\n\nPortraitE6` → `\n`  (dCE -29.26, base CE 30.56)
- `…Ye Olde Cheshire CheeseB3\n\n3Entertainment` → `\n`  (dCE -15.69, base CE 23.3)
- `…3\n\n34RevengeF4\n\n3Entertainment` → `\n`  (dCE -15.03, base CE 29.94)


### 16. `r.23.2.3` — a8, concentration 4.93

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.931 | a6 2.579 | m11 2.507 |
| interchange | a8 3.971 | a6 2.447 | m11 2.335 |

At `a8`: mean|dCE| on members **0.7349**, off slice 0.149, signed dCE on members 0.0693. Second-best component is `a6` at 2.579 — a 1.91x margin.


**DAS (rank 1, held-out):** member dCE 0.0222, concentration 3.53, recovers 0.027 of the full component; overlap with the closed-form direction 0.006.


**Story (from the circuit file):** {'blind_name': 'no behavioral claim survives base-rate testing', 'program': [['NOT circ_r_2_1_0', 'prev1_starts_space', 'NOT prev1_class_sentend']], 'program_bacc': 0.531, 'program_null': 0.51, 'mechanism_level': 'none', 'behavioral_tests': {'punct_pred_help_true': {'kind': 'punct', 'pred_help': True, 'n_available': 29, 'per_seed': [{'seed': 1, 'hits': 2, 'n': 5, 'p_value': 0.7493}, {'seed': 2, 'hits': 3, 'n': 5, 'p_value': 0.4137}, {'seed': 3, 'hits': 3, 'n': 5, 'p_value': 0.4137}, {'seed': 4, 'hits': 4, 'n': 5, 'p_value': 0.135}, {'seed': 11, 'hits': 3, 'n': 5, 'p_value': 0.4137}], 'seed_pass_frac': 0.0, 'population': {'n': 29, 'hits': 17, 'base_rate_help': 0.454, 'expected_hits': 13.16, 'p_value': 0.1065, 'beats_base_rate': False}, 'ROBUST': False, 'n_tests': 2, 'alpha': 0.05, 'ROBUST_V2': False, 'gate_note': 'use ROBUST_V2; ROBUST v1 is underpowered'}, 'punct_pred_help_false': {'kind': 'punct', 'pred_help': False, 'n_available': 29, 'per_seed': [{'seed': 1, 'hits': 3, 'n': 5, 'p_value': 0.5863}, {'seed': 2, 'hits': 2, 'n': 5, 'p_value': 0.865}, {'seed': 3, 'hits': 2, 'n': 5, 'p_value': 0.865}, {'seed': 4, 'hits': 1, 'n': 5, 'p_value': 0.9808}, {'seed': 11, 'hits': 2, 'n': 5, 'p_value': 0.865}], 'seed_pass_frac': 0.0, 'population': {'n': 29, 'hits': 12, 'base_rate_help': 0.454, 'expected_hits': 15.84, 'p_value': 0.9472, 'beats_base_rate': False}, 'ROBUST': False, 'n_tests': 2, 'alpha': 0.05, 'ROBUST_V2': False, 'gate_note': 'use ROBUST_V2; ROBUST v1 is underpowered'}, 'n_tests_declared': 2, 'alpha': 0.05, 'note': 'sibling leaves r.13.2.1 and r.18.2.0 carry VERIFIED punctuation claims; tested here as pre-declared hypothesis per assignment; neither direction cleared ROBUST_V2 (p=0.1065 helps-dir vs alpha 0.05, p=0.9472 hurts-dir)'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…- published: 19 Mar 2013\n-` → ` views`  (dCE 0.12, base CE 3.44)
- `… pitching statistics (DIPS)\n- Defensive Runs Saved (` → `D`  (dCE 0.42, base CE 1.19)
- `…It helps to fight against nuclear, biological and chemical weapons (NR` → `BC`  (dCE 0.19, base CE 11.32)


### 17. `r.11.1.2` — a8, concentration 4.90

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.904 | a14 4.101 | a16 3.239 |
| interchange | a8 4.19 | a14 3.556 | a16 2.798 |

At `a8`: mean|dCE| on members **0.7343**, off slice 0.1497, signed dCE on members 0.0819. Second-best component is `a14` at 4.101 — a 1.20x margin.


**DAS (rank 1, held-out):** member dCE 0.0396, concentration 4.366, recovers 0.043 of the full component; overlap with the closed-form direction 0.07.


**Story (from the circuit file):** {'blind_name': 'punctuation-target CE reduction (weak, single mechanism not found)', 'program': [['prev1_seen_before']], 'program_bacc': 0.587, 'program_null': 0.587, 'mechanism_level': 'none', 'behavior': 'Members at punctuation targets (5.9% of members) get pushed toward lower CE 71% of the time vs 47% base rate (ROBUST_V2, p=0.0007, n=51); at non-punctuation targets the push is near base rate (46%).', 'behavior_test': {'kind': 'punct', 'pred_help': True, 'n_available': 51, 'per_seed': [{'seed': 1, 'hits': 4, 'n': 5, 'p_value': 0.156}, {'seed': 2, 'hits': 5, 'n': 5, 'p_value': 0.0238}, {'seed': 3, 'hits': 4, 'n': 5, 'p_value': 0.156}, {'seed': 4, 'hits': 5, 'n': 5, 'p_value': 0.0238}, {'seed': 11, 'hits': 3, 'n': 5, 'p_value': 0.4502}], 'seed_pass_frac': 0.4, 'population': {'n': 51, 'hits': 36, 'base_rate_help': 0.473, 'expected_hits': 24.14, 'p_value': 0.0007, 'beats_base_rate': True}, 'ROBUST': False, 'n_tests': 12, 'alpha': 0.0083, 'ROBUST_V2': True, 'gate_note': 'use ROBUST_V2; ROBUST v1 is underpowered'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…influenced by western European harvest festivals, and festivals of the` → ` dead`  (dCE -1.67, base CE 3.74)
- `… essay on “How I met and chose my bridal party` → `.`  (dCE -1.2, base CE 2.79)
- `… to the Missouri Food Bank Association.\nCodi Coats of` → ` Bray`  (dCE 3.97, base CE 2.41)


### 18. `r.3.1.1` — a17, concentration 4.80

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a17 4.802 | a16 4.223 | a15 3.5 |
| interchange | a17 4.003 | a16 3.743 | a15 3.059 |

At `a17`: mean|dCE| on members **0.4446**, off slice 0.0926, signed dCE on members 0.3147. Second-best component is `a16` at 4.223 — a 1.14x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…#### Festivals & Events\n\n#### Sleeping\n\n#### Eating` → `\n`  (dCE -1.43, base CE 22.08)
- `…#### Brighton & Hove\n\n#### Arundel\n` → `\n`  (dCE 1.98, base CE 16.82)
- `… 2am Thu-Sat, noon-midnight Sun)\n` → `\n`  (dCE -1.42, base CE 12.84)


### 19. `r.0.0.1` — a3, concentration 4.60

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `m5` -> `a3`. Its held-out concentration is 4.4341, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a3 4.604 | a4 4.516 | m5 4.475 |
| interchange | a3 4.194 | m5 3.687 | a2 3.654 |

At `a3`: mean|dCE| on members **1.3709**, off slice 0.2977, signed dCE on members 0.1213. Second-best component is `a4` at 4.516 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT class_digit', 'NOT prev2_class_bclose']], 'program_bacc': 0.814, 'program_null': 0.55, 'mechanism_level': 'surface'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\nABleak House\n\nAJeake's House` → `\n`  (dCE 3.51, base CE 23.6)
- `…Museum of LondonD2\n\n17PinnacleG3` → `\n`  (dCE 5.15, base CE 18.99)
- `…BarrafinaF6\n\n27BarrafinaD3` → `\n`  (dCE 3.59, base CE 6.97)


### 20. `r.1.2.1` — m16, concentration 4.51

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 4.576, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 4.51 | m15 4.188 | m13 4.077 |
| interchange | m15 4.268 | m14 3.935 | a17 3.931 |

At `m16`: mean|dCE| on members **1.3873**, off slice 0.3076, signed dCE on members 0.9845. Second-best component is `m15` at 4.188 — a 1.08x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… B&B\n\n( ` → ` MAP`  (dCE -0.91, base CE 10.2)
- `…s ShedMARKET, RESTAURANT\n\n(` → ` `  (dCE -0.76, base CE 11.81)
- `…;  h10am-5.50pm;  t` → `South`  (dCE -0.2, base CE 13.34)


### 21. `r.2.2.2` — a6, concentration 4.37

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a6 4.373 | a8 4.095 | a14 3.064 |
| interchange | a6 3.806 | a8 3.597 | a14 2.923 |

At `a6`: mean|dCE| on members **0.9663**, off slice 0.221, signed dCE on members -0.0159. Second-best component is `a8` at 4.095 — a 1.07x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… Museum.\n\n8Getting There & Away\n\nBoat` → `\n`  (dCE -0.32, base CE 19.94)
- `…\n\nTo/From the Airports\n\nGatwick` → `\n`  (dCE 0.29, base CE 22.03)
- `… miles northwest of the centre at Withdean, from where bus` → ` 27`  (dCE 0.51, base CE 9.93)


### 22. `r.1.2` — m16, concentration 4.37

5,760 member positions in a slice of 38,400 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `m16` -> `m13`. Its held-out concentration is 4.2448, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 4.368 | m13 4.217 | m15 4.142 |
| interchange | m15 3.839 | m13 3.823 | m14 3.733 |

At `m16`: mean|dCE| on members **1.133**, off slice 0.2594, signed dCE on members 0.822. Second-best component is `m13` at 4.217 — a 1.04x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… Embankment or Temple) displays a wealth of 14th-` → ` to`  (dCE 1.18, base CE 6.21)
- `…motelB&B\n\n(  MAP   G` → `OO`  (dCE 0.27, base CE 5.27)
- `… since 1905, riverside Formans boasts prime views over the Olympic` → ` stadium`  (dCE -1.34, base CE 7.28)


### 23. `r.11.3.1` — a8, concentration 4.31

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.313 | a14 3.224 | a3 3.079 |
| interchange | a8 3.628 | a14 2.902 | a16 2.705 |

At `a8`: mean|dCE| on members **0.6339**, off slice 0.147, signed dCE on members 0.046. Second-best component is `a14` at 3.224 — a 1.34x margin.


**DAS (rank 1, held-out):** member dCE 0.0636, concentration 3.754, recovers 0.084 of the full component; overlap with the closed-form direction 0.228.


**Story (from the circuit file):** {'blind_name': 'templated-completion vs list/numeric-break selector', 'text': 'Machinery detects atypical token transitions; pushes toward templated completions -- helps predictable collocations (time units, common verb-noun pairs) but actively hurts unexpected list/numeric continuations.', 'program': [['NOT class_other', 'NOT is_newline']], 'program_bacc': 0.556, 'program_null': 0.551, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… 4 with Caulker and Muric and it seems tighter and` → ` Does`  (dCE -2.8, base CE 13.96)
- `…30 wake up and take systemic enzyme\n5-5.30` → `pm`  (dCE 1.88, base CE 3.84)
- `… face and affect how much capital they need.\nOr it can` → ` overest`  (dCE 1.22, base CE 6.8)


### 24. `r.4.1.0` — a16, concentration 4.22

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 4.217 | a14 3.719 | a12 2.969 |
| interchange | a16 3.618 | a14 3.395 | a13 2.8 |

At `a16`: mean|dCE| on members **0.3081**, off slice 0.0731, signed dCE on members 0.2006. Second-best component is `a14` at 3.719 — a 1.13x margin.


**Story (from the circuit file):** {'blind_name': 'Two-signed circuit: hurts predictable words (said, tying, opener) but helps concrete nouns (restaurants); wrong for short function/number words it also suppresses.', 'program': [['circ_r_0_0_1']], 'program_bacc': 0.567, 'program_null': 0.456, 'mechanism_level': 'none', 'redteam_hits': '3/3', 'redteam_verdict': 'HELD'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… the 878th for Boeheim, moving him within one of` → ` tying`  (dCE -4.22, base CE 7.82)
- `… tax oxygen,” said Hodges.\nFranchot` → ` said`  (dCE -2.99, base CE 4.43)
- `… score since taking a punt back against Samford in the 2010 season` → ` opener`  (dCE -1.73, base CE 2.08)


### 25. `r.1.0` — m14, concentration 4.22

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m14 4.217 | m13 4.145 | m15 3.907 |
| interchange | m14 3.308 | m13 3.222 | m15 3.08 |

At `m14`: mean|dCE| on members **0.5403**, off slice 0.1281, signed dCE on members 0.0613. Second-best component is `m13` at 4.145 — a 1.02x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… ska and reggae beats.\n\n#### Islington\n` → `\n`  (dCE 6.28, base CE 21.62)
- `…eth that sit closest to the river.\n\nSouth Bank\n` → `\n`  (dCE 9.3, base CE 22.8)
- `….\n\nJubilee LibraryINTERNET\n\n( ` → ` G`  (dCE 4.79, base CE 11.66)


### 26. `r.2.3` — a8, concentration 4.18

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.178 | a6 3.391 | a14 3.046 |
| interchange | a8 3.479 | a6 3.02 | a14 2.871 |

At `a8`: mean|dCE| on members **0.4919**, off slice 0.1177, signed dCE on members 0.0577. Second-best component is `a6` at 3.391 — a 1.23x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… to Stay\n\nAHaymarket Hotel\n\nAZetter Hotel` → `\n`  (dCE 3.71, base CE 25.8)
- `…\n\n#### Battle\n\n#### Hastings\n\n#### Eastbourne` → `\n`  (dCE 2.61, base CE 22.77)
- `…\n\n### Best Places to Stay\n\nAHaymarket Hotel` → `\n`  (dCE 8.11, base CE 16.41)


### 27. `r.1.0.2` — m14, concentration 4.14

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `m12` -> `m14`. Its held-out concentration is 3.7273, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m14 4.138 | m12 3.897 | m13 3.876 |
| interchange | m14 3.191 | m13 3.021 | m15 3.016 |

At `m14`: mean|dCE| on members **0.6422**, off slice 0.1552, signed dCE on members -0.028. Second-best component is `m12` at 3.897 — a 1.06x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…** and the Peter Harrison Planetarium (  GOOGLE` → ` MAP`  (dCE 0.94, base CE 3.63)
- `…Post OfficePOST OFFICE\n\n(  GOOGLE` → ` MAP`  (dCE 1.17, base CE 3.08)
- `…OUTIQUE HOTEL\n\n(  GOOGLE` → ` MAP`  (dCE 1.06, base CE 1.97)


### 28. `r.2.1` — a8, concentration 4.13

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.126 | a6 3.334 | a14 3.201 |
| interchange | a8 3.401 | a6 2.971 | a14 2.879 |

At `a8`: mean|dCE| on members **0.4859**, off slice 0.1177, signed dCE on members 0.0388. Second-best component is `a6` at 3.334 — a 1.24x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… Swain's Lane, N6; East Cemetery adult/child` → ` £`  (dCE 5.19, base CE 5.53)
- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE 3.19, base CE 22.84)
- `… House, a Georgian mansion built in 1718 for the wealthy hop` → ` merchant`  (dCE 3.83, base CE 6.8)


### 29. `r.6.3.0` — a16, concentration 4.11

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `a16` -> `a14`. Its held-out concentration is 4.0277, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 4.111 | a14 4.035 | a8 3.207 |
| interchange | a14 3.559 | a16 3.537 | a8 2.843 |

At `a16`: mean|dCE| on members **0.305**, off slice 0.0742, signed dCE on members 0.1579. Second-best component is `a14` at 4.035 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': 'Hard-to-predict delimiters (newlines, semicolons) in travel-guide listings/headings; circuit usually pushes the wrong token, raising loss on the majority, helping only a minority.', 'program': [['circ_r_2_1_0'], ['NOT starts_space', 'NOT prev_newline', 'NOT prev1_class_comma']], 'program_bacc': 0.677, 'program_null': 0.3, 'mechanism_level': 'none', 'redteam': {'literal_membership_reading': {'hits': 2, 'of': 3, 'note': 'SOP wording as originally read (pre round-1/2 edit): does story structurally describe the example? 53433 (target " p", not a delimiter) = miss; 51111, 54271 (newline at listing/heading boundary) = hit.'}, 'causal_direction_reading': {'hits': 0, 'of': 3, 'note': 'SOP wording as corrected by concurrent round-2 edit (landed mid-task, discovered after my step 5 ran): does story predict machinery helps vs hurts from dCE sign? All 3 random examples have POSITIVE dCE (circuit helps = minority behavior, 39.7% of members) but story gives no context-level rule to distinguish minority-helps from majority-hurts cases, so it cannot correctly call any of the 3 in advance.'}, 'verdict': 'weak (per corrected/current SOP step-5 wording: causal-direction hits=0/3 <=1/3 threshold); flagged for revision -- story lacks a structural rule for which members are in the helping-minority vs hurting-majority split', 'hits': 0, 'of': 3}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE 6.79, base CE 22.84)
- `…\n\n34RevengeF4\n\n3Entertainment\n` → `\n`  (dCE -4.55, base CE 24.4)
- `…chen.co.uk; 11 Sicilian Ave, WC1` → `;`  (dCE -4.22, base CE 7.02)


### 30. `r.1.3.1` — m13, concentration 4.10

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `m14` -> `m13`. Its held-out concentration is 4.0055, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m13 4.098 | m14 4.085 | m12 3.942 |
| interchange | m14 3.485 | m13 3.458 | m12 3.3 |

At `m13`: mean|dCE| on members **0.7196**, off slice 0.1756, signed dCE on members 0.1091. Second-best component is `m14` at 4.085 — a 1.00x margin.


**Story (from the circuit file):** {'blind_name': 'Members split near-evenly: pushes CE down for clean word/name completions (cooperate, Koch, shall); pushes CE up for garbled/sub-word fragments (mojibake, angers, ication).', 'program': [['class_subword', 'NOT is_newline']], 'program_bacc': 0.609, 'program_null': 0.47, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\nWe have developed an interactive map driven website. Often enough,` → ` cooperate`  (dCE -0.87, base CE 17.81)
- `… first stint with the Marlins from 2003-05, even pulling Billy` → ` Koch`  (dCE -0.77, base CE 10.0)
- `…z, gave away seats from “The Madhouse on Madison` → `�`  (dCE 0.65, base CE 2.55)


### 31. `r.1.1.2` — m16, concentration 4.08

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.9925, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 4.083 | m13 3.981 | m14 3.914 |
| interchange | m15 3.791 | m14 3.713 | m13 3.618 |

At `m16`: mean|dCE| on members **1.2798**, off slice 0.3134, signed dCE on members 0.9068. Second-best component is `m13` at 3.981 — a 1.03x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['circ_r_1_1_2']], 'program_bacc': 0.773, 'program_null': 0.512, 'mechanism_level': 'surface'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### London Highlights\n\n#### History\n\n#### Sights` → `\n`  (dCE 1.74, base CE 24.86)
- `…pm Wed-Sat, noon-5pm Sun;  t` → `High`  (dCE -0.12, base CE 15.07)
- `…; Bartholomew Lane, EC2;  h10am` → `-`  (dCE 0.58, base CE 1.14)


### 32. `r.1.1.1` — m16, concentration 4.08

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `m15` -> `m16`. Its held-out concentration is 4.008, so the circuit still localises; it is the single component NAME that is not settled (§2061).

> **Band-localised, not component-localised.** The two methods disagree by one or two layers inside the `m13`–`m16` band, which is the signature of a circuit spread across adjacent MLPs rather than sitting on one.

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 4.082 | m15 4.081 | m14 4.066 |
| interchange | m14 3.828 | m13 3.612 | m15 3.606 |

At `m16`: mean|dCE| on members **1.2793**, off slice 0.3134, signed dCE on members 0.8831. Second-best component is `m15` at 4.081 — a 1.00x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT prev1_mid_word', 'NOT dist_nl_le2'], ['prev1_is_punct', 'NOT prev2_upper_initial', 'NOT prev2_class_sentend']], 'program_bacc': 0.76, 'program_null': 0.404, 'mechanism_level': 'surface'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… B&B\n\n(  MAP` → ` `  (dCE -1.29, base CE 4.38)
- `… outside.\n\nGeorge InnPUB\n\n(  MAP` → ` `  (dCE -1.27, base CE 4.18)
- `…\nEast Lee Guest HouseB&B\n\n(  MAP` → ` `  (dCE -1.4, base CE 4.66)


### 33. `r.2.2` — a8, concentration 4.05

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 4.054 | a6 3.285 | a14 3.056 |
| interchange | a8 3.371 | a6 2.932 | a14 2.8 |

At `a8`: mean|dCE| on members **0.4774**, off slice 0.1177, signed dCE on members 0.0606. Second-best component is `a6` at 3.285 — a 1.23x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…london.org.uk; 150 London Wall, EC2` → `;`  (dCE -2.61, base CE 6.02)
- `…ito.co.uk; 32 Exmouth Market, EC1` → `;`  (dCE -2.62, base CE 6.32)
- `…ethouse.org.uk; The Strand, WC2` → `;`  (dCE -2.5, base CE 5.52)


### 34. `r.1.0.3` — m14, concentration 4.00

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m14 3.997 | m13 3.925 | m12 3.861 |
| interchange | m14 3.227 | m13 3.143 | m12 2.972 |

At `m14`: mean|dCE| on members **0.6202**, off slice 0.1552, signed dCE on members -0.0221. Second-best component is `m13` at 3.925 — a 1.02x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… while the heavenly waft from the cheese room beckons.\n` → `\n`  (dCE -0.63, base CE 12.75)
- `…Hotel UnaBOUTIQUE HOTEL\n\n(` → ` `  (dCE -1.49, base CE 10.28)
- `…'s above Heath St, reached via the Holly Bush Steps.\n` → `\n`  (dCE 0.45, base CE 12.65)


### 35. `r.2.1.0` — a8, concentration 3.99

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 3.992 | a7 3.694 | a6 3.611 |
| interchange | a8 3.385 | a6 3.196 | a14 3.164 |

At `a8`: mean|dCE| on members **0.594**, off slice 0.1488, signed dCE on members -0.0132. Second-best component is `a7` at 3.694 — a 1.08x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT class_subword', 'NOT class_name']], 'program_bacc': 0.782, 'program_null': 0.485, 'mechanism_level': 'surface'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\n4Westminster AbbeyF3\n\n1Sights` → `\n`  (dCE -7.7, base CE 31.91)
- `…\n\n33BarbicanE1\n\n7Shopping` → `\n`  (dCE -7.46, base CE 21.63)
- `… Sir Walter Scott and Jane Austen.\n\n### Farnham` → `\n`  (dCE 10.39, base CE 3.55)


### 36. `r.11.1.1` — a8, concentration 3.98

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 3.979 | a14 3.534 | a3 3.024 |
| interchange | a8 3.488 | a14 3.096 | a16 2.605 |

At `a8`: mean|dCE| on members **0.5958**, off slice 0.1497, signed dCE on members 0.09. Second-best component is `a14` at 3.534 — a 1.13x margin.


**DAS (rank 1, held-out):** member dCE 0.012, concentration 2.587, recovers 0.017 of the full component; overlap with the closed-form direction 0.0.


**Story (from the circuit file):** {'blind_name': 'no surface program; punctuation-target push is the only surviving behavioral claim', 'program': [['NOT prev1_class_other', 'NOT prev2_class_other', 'NOT dist_nl_le2']], 'program_bacc': 0.588, 'program_null': 0.58, 'mechanism_level': 'none', 'mechanism_line': 'No STRONG single-writer mechanism in a4, a3, or a8 (top ratios 1.093/1.093/1.144 vs thresholds 1.364/1.300/1.300; headroom -0.321/-0.269/-0.225).', 'behavior_line': "Machinery is a symmetric 50/50 push overall (dce_pos -0.709, dce_neg 0.886, minority_share 0.5) but flips at punctuation targets: on the 63/864 punct-target members ablation LOWERS CE 69.8% of the time vs a 48% base rate (margin +21.8pp, p=0.0004, Bonferroni alpha=0.0083/12), i.e. this leaf's machinery actively hurts punctuation prediction, while on the other 801/864 (non-punct) members the push runs the other way (mean dCE +0.116, helps prediction).", 'story_test_class_pairs': {'subword_help': {'ROBUST_V2': False, 'n_available': 140, 'population': {'n': 140, 'hits': 64, 'base_rate_help': 0.48, 'expected_hits': 67.25, 'p_value': 0.7365, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'subword_hurt': {'ROBUST_V2': False, 'n_available': 140, 'population': {'n': 140, 'hits': 76, 'base_rate_help': 0.48, 'expected_hits': 72.75, 'p_value': 0.3216, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'space_word_help': {'ROBUST_V2': False, 'n_available': 629, 'population': {'n': 629, 'hits': 289, 'base_rate_help': 0.48, 'expected_hits': 302.12, 'p_value': 0.8616, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'space_word_hurt': {'ROBUST_V2': False, 'n_available': 629, 'population': {'n': 629, 'hits': 340, 'base_rate_help': 0.48, 'expected_hits': 326.88, 'p_value': 0.1569, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'digit_help': {'ROBUST_V2': False, 'n_available': 29, 'population': {'n': 29, 'hits': 16, 'base_rate_help': 0.48, 'expected_hits': 13.93, 'p_value': 0.2795, 'beats_base_rate': False}, 'seed_pass_frac': 0.2}, 'digit_hurt': {'ROBUST_V2': False, 'n_available': 29, 'population': {'n': 29, 'hits': 13, 'base_rate_help': 0.48, 'expected_hits': 15.07, 'p_value': 0.8303, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'punct_help': {'ROBUST_V2': True, 'n_available': 63, 'population': {'n': 63, 'hits': 44, 'base_rate_help': 0.48, 'expected_hits': 30.26, 'p_value': 0.0004, 'beats_base_rate': True}, 'seed_pass_frac': 0.2}, 'punct_hurt': {'ROBUST_V2': False, 'n_available': 63, 'population': {'n': 63, 'hits': 19, 'base_rate_help': 0.48, 'expected_hits': 32.74, 'p_value': 0.9999, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'capitalized_help': {'ROBUST_V2': False, 'n_available': 154, 'population': {'n': 154, 'hits': 72, 'base_rate_help': 0.48, 'expected_hits': 73.97, 'p_value': 0.6544, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'capitalized_hurt': {'ROBUST_V2': False, 'n_available': 154, 'population': {'n': 154, 'hits': 82, 'base_rate_help': 0.48, 'expected_hits': 80.03, 'p_value': 0.4067, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'newline_help': {'ROBUST_V2': False, 'n_available': 2, 'population': {'n': 2, 'hits': 2, 'base_rate_help': 0.48, 'expected_hits': 0.96, 'p_value': 0.2307, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}, 'newline_hurt': {'ROBUST_V2': False, 'n_available': 2, 'population': {'n': 2, 'hits': 0, 'base_rate_help': 0.48, 'expected_hits': 1.04, 'p_value': 1.0, 'beats_base_rate': False}, 'seed_pass_frac': 0.0}}, 'n_tests': 12}


**Top members** (context → target, dCE when the circuit is ablated):

- `…N/C:N/I:N/A:P` → `)`  (dCE -2.21, base CE 6.8)
- `… a single charge from a standard 110-volt outlet.\nThe` → ` Em`  (dCE -0.7, base CE 4.12)
- `…? | Air Canada | RBC | Samsung Galaxy S4 |` → ` Target`  (dCE 1.6, base CE 7.48)


### 37. `r.6.0.1` — a14, concentration 3.97

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `a16` -> `a14`. Its held-out concentration is 3.9932, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a14 3.973 | a16 3.95 | m17 3.064 |
| interchange | a14 3.93 | a16 3.663 | a13 2.836 |

At `a14`: mean|dCE| on members **0.4359**, off slice 0.1097, signed dCE on members 0.259. Second-best component is `a16` at 3.95 — a 1.01x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… INFORMATION\n\n(www.brighton.co.uk)` → `\n`  (dCE 0.99, base CE 12.12)
- `…3\n\n7London St Pancras YHAC2\n` → `\n`  (dCE -0.3, base CE 10.57)
- `…1Sights\n\nStadeNEIGHBOURHOOD` → `\n`  (dCE 0.69, base CE 12.45)


### 38. `r.1.1.0` — m14, concentration 3.96

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `m15` -> `m14`. Its held-out concentration is 3.8114, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m14 3.963 | m15 3.897 | m13 3.873 |
| interchange | m14 3.536 | m15 3.434 | m13 3.338 |

At `m14`: mean|dCE| on members **0.6206**, off slice 0.1566, signed dCE on members 0.0673. Second-best component is `m15` at 3.897 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['circ_r_1_1_2', 'NOT dist_nl_ge6']], 'program_bacc': 0.735, 'program_null': 0.551, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… hourly).\n\n### Hastings\n\nPop 90,300\n` → `\n`  (dCE 2.22, base CE 21.86)
- `… also available.\n\nNorth Central London\n\n1Top S` → `ights`  (dCE -0.68, base CE 3.5)
- `….arthousebandb.com; 24 London Rd; r` → ` £`  (dCE -0.07, base CE 11.06)


### 39. `r.0.3.0` — a3, concentration 3.91

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `m5` -> `a3`. Its held-out concentration is 3.7503, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a3 3.907 | m5 3.748 | a2 3.495 |
| interchange | a3 3.484 | m5 3.197 | a2 3.097 |

At `a3`: mean|dCE| on members **1.1891**, off slice 0.3044, signed dCE on members -0.0045. Second-best component is `m5` at 3.748 — a 1.04x margin.


**Story (from the circuit file):** {'blind_name': 'Members are rare/proper-noun-like continuation tokens (numbers, name-fragments); machinery helps predict these but wrongly pushes against ordinary common-word completions elsewhere.', 'program': [['NOT class_other', 'NOT is_newline']], 'program_bacc': 0.566, 'program_null': 0.504, 'mechanism_level': 'none', 'redteam_hits': '2/3', 'redteam_verdict': 'HELD', 'flag': 'weak: reviewer-two objection (2026-08-20) -- specific claim never exercised by fresh draw; own program_bacc failed; catch-all branch is base-rate. Verdict-rule v2 would say WEAKEN; recorded by driver.'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… the Battle of the Bulge - Charles B. MacDonald.\n` → `002`  (dCE 1.56, base CE 5.0)
- `…\nrestaurants. International restaurant guides are in development.\n` → `Cu`  (dCE 1.65, base CE 5.11)
- `… score since taking a punt back against Samford in the 2010 season` → ` opener`  (dCE -1.58, base CE 2.08)


### 40. `r.6.1.0` — m5, concentration 3.84

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.8558, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m5 3.836 | a14 3.447 | m6 3.308 |
| interchange | a14 3.062 | m5 2.985 | a16 2.84 |

At `m5`: mean|dCE| on members **1.1044**, off slice 0.2879, signed dCE on members 0.135. Second-best component is `a14` at 3.447 — a 1.11x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…orgeinrye.com; 98 High St; d from` → ` £`  (dCE -0.55, base CE 8.32)
- `…vern.com; 35 Chalk Farm Rd, NW` → `1`  (dCE 1.01, base CE 3.73)
- `….myhotels.com; 17 Jubilee St; r` → ` £`  (dCE -0.45, base CE 11.88)


### 41. `r.1.1` — m13, concentration 3.81

5,760 member positions in a slice of 38,400 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.7817, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m13 3.815 | m14 3.734 | m15 3.691 |
| interchange | m14 3.357 | m13 3.298 | m15 3.261 |

At `m13`: mean|dCE| on members **0.5404**, off slice 0.1416, signed dCE on members 0.1225. Second-best component is `m14` at 3.734 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['class_newline'], ['NOT class_other', 'NOT class_subword', 'NOT class_digit']], 'program_bacc': 0.73, 'program_null': 0.704, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\nAWallett's Court\n\nAReading Rooms` → `\n`  (dCE 1.98, base CE 28.53)
- `…\n\n### Best Places to Stay\n\nABleak House` → `\n`  (dCE 2.6, base CE 30.09)
- `…ights\n\n1Westminster Hall\n\n2Jewel Tower` → `\n`  (dCE 3.46, base CE 24.17)


### 42. `r.1.3` — m13, concentration 3.77

5,760 member positions in a slice of 38,400 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m13 3.775 | m14 3.647 | m15 3.631 |
| interchange | m13 3.298 | m14 3.295 | m15 3.22 |

At `m13`: mean|dCE| on members **0.5347**, off slice 0.1416, signed dCE on members 0.1132. Second-best component is `m14` at 3.647 — a 1.04x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… hmuseum 9.30am-5pm;  t` → `F`  (dCE 0.52, base CE 9.44)
- `…-Sat, noon-10.30pm Sun;  t` → `St`  (dCE 0.59, base CE 10.98)
- `… Dover\n\n#### Rye\n\n#### Battle\n\n#### Hastings` → `\n`  (dCE 0.45, base CE 33.14)


### 43. `r.6.0.3` — m17, concentration 3.65

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.5469, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m17 3.655 | m16 3.182 | a16 2.596 |
| interchange | m16 3.101 | m17 2.968 | a16 2.646 |

At `m17`: mean|dCE| on members **2.5202**, off slice 0.6896, signed dCE on members 2.0933. Second-best component is `m16` at 3.182 — a 1.15x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE 2.13, base CE 22.84)
- `…6 7890; www.uclh.org; 235 E` → `ust`  (dCE 2.62, base CE 0.48)
- `… 39a Canonbury Sq, N1; adult/child` → ` £`  (dCE 2.72, base CE 4.28)


### 44. `r.6.2.2` — m16, concentration 3.63

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m16 3.632 | a17 3.081 | m15 2.82 |
| interchange | m16 3.258 | a17 3.105 | m15 2.953 |

At `m16`: mean|dCE| on members **1.158**, off slice 0.3188, signed dCE on members 0.8188. Second-best component is `a17` at 3.081 — a 1.18x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['circ_r_1_1_2', 'NOT class_other', 'NOT is_punct']], 'program_bacc': 0.77, 'program_null': 0.662, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… has well-appointed rooms to let.\n\n7Shopping` → `\n`  (dCE -2.07, base CE 21.84)
- `…bury River Navigation CompanyB2\n\n4Sleeping\n` → `\n`  (dCE -4.23, base CE 26.74)
- `…\n\n2Marble ArchA5\n\n5Eating` → `\n`  (dCE -0.82, base CE 16.7)


### 45. `r.3.1.0` — a2, concentration 3.63

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `a14` -> `a2`. Its held-out concentration is 2.7737, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a2 3.629 | m5 3.557 | a17 3.512 |
| interchange | a16 3.225 | a17 3.206 | a3 3.175 |

At `a2`: mean|dCE| on members **1.3427**, off slice 0.37, signed dCE on members 0.1808. Second-best component is `m5` at 3.557 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT class_subword', 'NOT class_digit']], 'program_bacc': 0.862, 'program_null': 0.327, 'mechanism_level': 'surface'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE -10.78, base CE 22.84)
- `…'s GlobeD1\n\n4Tate ModernD1\n` → `\n`  (dCE -4.08, base CE 14.76)
- `…\n\nTo/From the Airports\n\nGatwick` → `\n`  (dCE -2.39, base CE 22.03)


### 46. `r.1.0.1` — m14, concentration 3.60

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `m12` -> `m14`. Its held-out concentration is 3.3564, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m14 3.601 | m13 3.55 | m12 3.449 |
| interchange | a16 2.969 | m14 2.931 | a14 2.741 |

At `m14`: mean|dCE| on members **0.5588**, off slice 0.1552, signed dCE on members -0.0252. Second-best component is `m13` at 3.55 — a 1.01x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['prev1_is_punct', 'NOT prev2_class_sentend', 'NOT prev2_class_comma'], ['prev2_class_ind', 'NOT prev2_starts_space', 'NOT class_digit']], 'program_bacc': 0.626, 'program_null': 0.429, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `….uk;  h5am-dusk;  t` → `Reg`  (dCE -1.58, base CE 13.93)
- `… up in the 15th century; the spire dates from the` → ` 19`  (dCE -1.07, base CE 4.28)
- `…4pm Mon-Fri; ` → ` t`  (dCE 2.11, base CE 13.49)


### 47. `r.6.2.1` — a9, concentration 3.59

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a9 3.592 | a11 2.695 | m9 2.654 |
| interchange | a9 2.88 | a7 2.449 | a16 2.449 |

At `a9`: mean|dCE| on members **0.8202**, off slice 0.2283, signed dCE on members -0.1922. Second-best component is `a11` at 2.695 — a 1.33x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…; www.barbican.org.uk; Silk St` → `,`  (dCE 0.1, base CE 5.6)
- `…rundel and the coast.\n\n1Sights` → `\n`  (dCE -5.28, base CE 15.32)
- `… www.wearebigchill.com; 257-259` → ` Pent`  (dCE 1.01, base CE 11.23)


### 48. `r.2.2.0` — a8, concentration 3.55

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 3.553 | a14 3.106 | a7 2.992 |
| interchange | a8 3.099 | a14 2.793 | a7 2.677 |

At `a8`: mean|dCE| on members **0.5293**, off slice 0.149, signed dCE on members 0.0098. Second-best component is `a14` at 3.106 — a 1.14x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…Eating\n\n16BohoB2\n\n17D` → `ees`  (dCE 1.77, base CE 9.74)
- `…nline; combined tickets with the London Eye, London Dungeon and London` → ` Seal`  (dCE -1.06, base CE 15.58)
- `… Lane, EC4; dm £17-25, d` → ` £`  (dCE -1.23, base CE 5.26)


### 49. `r.6.1.1` — a16, concentration 3.53

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `a8` -> `a16`. Its held-out concentration is 2.9672, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 3.531 | a14 3.387 | m5 3.361 |
| interchange | a16 3.063 | a14 3.056 | a8 3.048 |

At `a16`: mean|dCE| on members **0.2636**, off slice 0.0747, signed dCE on members 0.1349. Second-best component is `a14` at 3.387 — a 1.04x margin.


**Story (from the circuit file):** {'blind_name': "Domain/URL-listing separator token; helps complete website suffixes and prices after 'www.x;' but wrongly raises loss on other post-';' continuations like newlines.", 'program': [['NOT starts_space', 'NOT class_name']], 'program_bacc': 0.606, 'program_null': 0.454, 'mechanism_level': 'none', 'redteam': [{'gi': 36595, 'context': ' Getting hopelessly lost in the crooked medieval streets of Sandwich\n\n', 'target': '2', 'dce': 1.72, 'verdict': 'MISS', 'why': 'narrative town-guide text (not a URL/address-separator context); story does not predict this member at all'}, {'gi': 52954, 'context': '-5pm Mon-Sat, to 4pm Sun)\n', 'target': '\n', 'dce': -1.23, 'verdict': 'HIT', 'why': "listing-style hours text ending in newline, negative dce matches story's wrong-push-on-newlines direction"}, {'gi': 8581, 'context': 'designmuseum.org; 28 Shad Thames, SE1;', 'target': ' adult', 'dce': -0.89, 'verdict': 'HIT', 'why': "exact www-domain + ';' listing separator predicting ' adult', negative dce matches story"}]}


**Top members** (context → target, dCE when the circuit is ablated):

- `…  GOOGLE MAP ) ; www.nhm.` → `ac`  (dCE 3.03, base CE 1.74)
- `…riverbus.co.uk; adult/child 30min trip` → ` £`  (dCE 2.83, base CE 3.16)
- `… www.wearebigchill.com; 257-259` → ` Pent`  (dCE 2.02, base CE 11.23)


### 50. `r.8.1.0` — a3, concentration 3.50

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a3 3.499 | a4 3.328 | a5 3.292 |
| interchange | a3 3.13 | a8 2.936 | a5 2.881 |

At `a3`: mean|dCE| on members **1.0746**, off slice 0.3072, signed dCE on members -0.0637. Second-best component is `a4` at 3.328 — a 1.05x margin.


**Story (from the circuit file):** {'blind_name': None, 'program': [['circ_r_0_0_1', 'NOT is_newline']], 'program_bacc': 0.574, 'program_null': 0.498, 'mechanism_level': 'none', 'mechanism': "No writer/courier identified: input-writer decomposition into a4/a9/a7 (this leaf's machinery) is not enriched vs off-slice under the 5-seed bootstrap (min ratios 0.996-1.216, below the 1.3 ENRICHED bar; ENRICHED_STABLE=False for all three components). Surface program (NOT is_newline) also fails doc-disjoint heldout (bacc 0.574 vs null 0.498). No behavioral claim survives base-rate testing across 12 class/direction pairs (Bonferroni alpha 0.0083; best punct/helps population p=0.0455)."}


**Top members** (context → target, dCE when the circuit is ablated):

- `… and her Paramount swan song, "True Confession" --` → ` was`  (dCE -2.82, base CE 9.94)
- `… the Battle of the Bulge - Charles B. MacDonald.\n` → `002`  (dCE 3.81, base CE 5.0)
- `… contests put on by ActiveRain and its members. Everyone can join` → ` the`  (dCE 1.88, base CE 4.22)


### 51. `r.5.0.1` — a16, concentration 3.40

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `a3` -> `a16`. Its held-out concentration is 3.4141, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 3.398 | a3 3.373 | a2 3.001 |
| interchange | a16 3.034 | a14 2.762 | a3 2.727 |

At `a16`: mean|dCE| on members **0.2494**, off slice 0.0734, signed dCE on members 0.1283. Second-best component is `a3` at 3.373 — a 1.01x margin.


**Story (from the circuit file):** {'program': [['circ_r_0_0_1']], 'program_bacc': 0.556, 'program_null': 0.442, 'mechanism_level': 'none', 'behavior_line': 'no behavioral claim survives base-rate testing', 'story_test_by_kind': {'subword': {'n': 5, 'hits': 4, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.128, 'beats_base_rate': False}, 'newline': {'n': 5, 'hits': 4, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.128, 'beats_base_rate': False}, 'space_word': {'n': 5, 'hits': 2, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.7389, 'beats_base_rate': False}, 'punct': {'n': 5, 'hits': 2, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.7389, 'beats_base_rate': False}, 'digit': {'n': 5, 'hits': 3, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.4009, 'beats_base_rate': False}, 'capitalized': {'n': 5, 'hits': 2, 'base_rate_help': 0.447, 'expected_hits': 2.23, 'p_value': 0.7389, 'beats_base_rate': False}}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…efoodlikeItalianfood *inhale* Idon't` → `even`  (dCE -1.69, base CE 6.98)
- `… nave, chancel, south porch,\nand south is` → `le`  (dCE 5.41, base CE 1.79)
- `…ant lilies that dominate bouquets and their little bell shapes` → ` could`  (dCE -1.34, base CE 11.47)


### 52. `r.23.2.1` — a8, concentration 3.38

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a8 3.385 | a2 3.026 | a3 2.894 |
| interchange | a8 2.908 | a3 2.542 | a6 2.516 |

At `a8`: mean|dCE| on members **0.5044**, off slice 0.149, signed dCE on members 0.0919. Second-best component is `a2` at 3.026 — a 1.12x margin.


**DAS (rank 1, held-out):** member dCE 0.0215, concentration 3.075, recovers 0.036 of the full component; overlap with the closed-form direction 0.029.


**Story (from the circuit file):** {'blind_name': 'generic-vs-proper-noun completion push', 'text': 'High-loss continuations after named/branded contexts; machinery favors generic common-word completions, suppresses specific proper-noun/brand completions (except some place-name suffixes, e.g. Hungaroring).', 'program': [['NOT prev1_seen_before', 'NOT is_punct'], ['circ_r_0_0_1', 'NOT dist_nl_le2', 'NOT prev1_class_other']], 'program_bacc': 0.517, 'program_null': 0.541, 'mechanism_level': 'none', 'red_team': {'hits': 3, 'total': 3, 'detail': [{'gi': 764, 'target': ' miserable', 'dce': 0.94, 'predicted': 'helps (generic common adjective)', 'actual': 'helps', 'hit': True}, {'gi': 154700, 'target': 'ted', 'dce': -0.54, 'predicted': 'hurts (brand/proper-noun continuation of "Target")', 'actual': 'hurts', 'hit': True, 'note': 'ambiguous: completed word "Targeted" is itself generic; scored as hurts because context token is the brand name Target'}, {'gi': 240197, 'target': ' Ars', 'dce': -1.07, 'predicted': 'hurts (specific proper noun, Arsenal)', 'actual': 'hurts', 'hit': True}], 'caveat': 'top example gi=198076 ("Hungar"->"oring", dce=+2.51) contradicts the suppress-proper-noun claim: machinery HELPS complete a place-name compound there. Story is directional/majority, not exceptionless.'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `…so\n|First race||Hungarian Grand Prix||Hungar` → `oring`  (dCE 2.51, base CE 4.13)
- `…� debate highlighted at this year’s PSA Conference by` → ` Matthew`  (dCE -1.78, base CE 11.18)
- `… indigenous black Christians and animists, 100,000 in Darfur` → ` alone`  (dCE 2.45, base CE 5.56)


### 53. `r.5.3.1` — a15, concentration 3.33

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a15 3.33 | a16 3.093 | a2 2.88 |
| interchange | a15 2.947 | a16 2.772 | a14 2.474 |

At `a15`: mean|dCE| on members **0.222**, off slice 0.0667, signed dCE on members 0.048. Second-best component is `a16` at 3.093 — a 1.08x margin.


**Story (from the circuit file):** {'blind_name': 'no behavioral claim survives base-rate testing', 'program': [['circ_r_0_0_1', 'dist_nl_ge6']], 'program_bacc': 0.583, 'program_null': 0.488, 'mechanism_level': 'none', 'mechanism': {'note': 'no writer enriches machinery input across bootstrap draws (a2 top a0 mean 1.005 [0.827-1.246]; a4 top a2 mean 1.048 [0.985-1.082]); ENRICHED_STABLE=False both'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `… and its members. Everyone can join the\ngroup and help encourage` → ` each`  (dCE -1.74, base CE 2.4)
- `… who would enjoy the sight of pickle and spider-web or` → `nam`  (dCE -1.97, base CE 2.23)
- `…\nIn a different twist, Bob Fink, an independent music` → `ologist`  (dCE 1.84, base CE 1.37)


### 54. `r.1.3.0` — m13, concentration 3.23

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.2216, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | m13 3.228 | m12 3.151 | m15 3.009 |
| interchange | a16 2.703 | a17 2.682 | a14 2.678 |

At `m13`: mean|dCE| on members **0.5668**, off slice 0.1756, signed dCE on members 0.0048. Second-best component is `m12` at 3.151 — a 1.02x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT class_other', 'NOT prev2_upper_initial', 'NOT class_digit']], 'program_bacc': 0.689, 'program_null': 0.49, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… restaurant. Book well ahead.\n\nJubilee HouseB` → `&`  (dCE 1.65, base CE 8.21)
- `… Severini.\n\nRoyal London Walking Tour\n\n1S` → `ights`  (dCE 0.84, base CE 9.37)
- `… hostel-like vibe. Only one of the three rooms is` → ` en`  (dCE 0.86, base CE 5.43)


### 55. `r.6.0.2` — a9, concentration 3.19

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: rows-only** — stable across a row split, but the two interventions name different components. Its held-out concentration is 3.254, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a9 3.185 | a16 2.988 | a14 2.896 |
| interchange | a16 2.892 | a14 2.74 | a9 2.73 |

At `a9`: mean|dCE| on members **0.7357**, off slice 0.231, signed dCE on members -0.1393. Second-best component is `a16` at 2.988 — a 1.07x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['NOT prev2_starts_space', 'NOT class_other', 'NOT class_digit'], ['NOT prev1_starts_space', 'NOT prev2_class_upper_pair', 'NOT prev2_class_comma']], 'program_bacc': 0.624, 'program_null': 0.4, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `… John's Wood Rd, NW8; tours adult/child` → ` £`  (dCE 2.34, base CE 2.88)
- `… required for the Monday- and Wednesday-evening candlelit sessions` → ` (£`  (dCE 1.6, base CE 7.64)
- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE 2.16, base CE 22.84)


### 56. `r.6.2.0` — a16, concentration 3.17

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `m16` -> `a16`. Its held-out concentration is 2.9067, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 3.167 | m16 2.955 | m17 2.744 |
| interchange | a16 2.965 | m16 2.836 | a14 2.802 |

At `a16`: mean|dCE| on members **0.2375**, off slice 0.075, signed dCE on members 0.111. Second-best component is `m16` at 2.955 — a 1.07x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n\n33BarbicanE1\n\n7Shopping` → `\n`  (dCE 0.17, base CE 21.63)
- `…Ye Olde Cheshire CheeseB3\n\n3Entertainment` → `\n`  (dCE 0.11, base CE 23.3)
- `…3\n\n34RevengeF4\n\n3Entertainment` → `\n`  (dCE 0.72, base CE 29.94)


### 57. `r.6.3.1` — a16, concentration 3.13

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 3.133 | a9 3.032 | a14 2.95 |
| interchange | a16 2.802 | a14 2.699 | a9 2.612 |

At `a16`: mean|dCE| on members **0.2324**, off slice 0.0742, signed dCE on members 0.1049. Second-best component is `a9` at 3.032 — a 1.03x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `… alternative music most nights in this pub's back room (cover usually` → ` £`  (dCE 1.82, base CE 4.65)
- `…\n\n33BarbicanE1\n\n7Shopping` → `\n`  (dCE -1.92, base CE 21.63)
- `…re.co.uk; Belvedere Rd, SE1` → `;`  (dCE -2.36, base CE 6.2)


### 58. `r.18.2.0` — a7, concentration 3.10

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a7 3.104 | a9 2.803 | a16 2.802 |
| interchange | a7 2.763 | a16 2.549 | a9 2.488 |

At `a7`: mean|dCE| on members **0.7248**, off slice 0.2335, signed dCE on members -0.0276. Second-best component is `a9` at 2.803 — a 1.11x margin.


**Story (from the circuit file):** {'blind_name': 'no behavioral claim survives base-rate testing except a narrow punctuation-target effect', 'program': [['starts_space', 'prev1_starts_space']], 'program_bacc': 0.555, 'program_null': 0.498, 'mechanism_level': 'none', 'behavior_line': "On punctuation-target positions the leaf's machinery HURTS prediction (ablating it lowers CE 5/5 draws, story_test p=0.041, beats base rate 0.528); but overall member sign-split is near-even (minority_share=0.473) and no other surface class (subword/space_word/digit/capitalized/newline) showed an enriched push in either direction -- the punct effect does not generalize to a whole-leaf story.", 'story_test_punct_help': {'n': 5, 'hits': 5, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.041, 'beats_base_rate': True}, 'story_test_all_kinds': {'subword': {'gis': [165065, 19833, 48100, 136615, 235398], 'help': {'n': 5, 'hits': 2, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.8453, 'beats_base_rate': False}, 'hurt': {'n': 5, 'hits': 3, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 0.448, 'beats_base_rate': False}}, 'space_word': {'gis': [105887, 32533, 122749, 253437, 216122], 'help': {'n': 5, 'hits': 2, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.8453, 'beats_base_rate': False}, 'hurt': {'n': 5, 'hits': 3, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 0.448, 'beats_base_rate': False}}, 'digit': {'gis': [39325, 208117, 123808, 21741, 255089], 'help': {'n': 5, 'hits': 1, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.9765, 'beats_base_rate': False}, 'hurt': {'n': 5, 'hits': 4, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 0.1547, 'beats_base_rate': False}}, 'punct': {'gis': [182573, 49010, 171307, 181938, 204857], 'help': {'n': 5, 'hits': 5, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.041, 'beats_base_rate': True}, 'hurt': {'n': 5, 'hits': 0, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 1.0, 'beats_base_rate': False}}, 'capitalized': {'gis': [251284, 255203, 232254, 201962, 204239], 'help': {'n': 5, 'hits': 3, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.552, 'beats_base_rate': False}, 'hurt': {'n': 5, 'hits': 2, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 0.7759, 'beats_base_rate': False}}, 'newline': {'gis': [55048, 255810, 17138, 216712, 140365], 'help': {'n': 5, 'hits': 3, 'base_rate_help': 0.528, 'expected_hits': 2.64, 'p_value': 0.552, 'beats_base_rate': False}, 'hurt': {'n': 5, 'hits': 2, 'base_rate_help': 0.528, 'expected_hits': 2.36, 'p_value': 0.7759, 'beats_base_rate': False}}}}


**Top members** (context → target, dCE when the circuit is ablated):

- `… he knows that I want him to be in a position to be` → ` attacking`  (dCE -1.35, base CE 11.81)
- `… it but more lengthening. My main gripe with it back` → ` there`  (dCE -1.58, base CE 7.07)
- `…magical thinking to further her showbiz career -- starting with visual` → `izing`  (dCE -1.62, base CE 4.68)


### 59. `r.13.2.1` — a3, concentration 3.04

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `a3` -> `a7`. Its held-out concentration is 3.0512, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a3 3.042 | a7 3.009 | a6 2.844 |
| interchange | a7 2.771 | a14 2.631 | a6 2.607 |

At `a3`: mean|dCE| on members **0.9373**, off slice 0.3081, signed dCE on members 0.0746. Second-best component is `a7` at 3.009 — a 1.01x margin.


**Story (from the circuit file):** {'blind_name': 'punctuation-target CE reducer (two-signed, ~50/50 elsewhere)', 'program': [['NOT class_other', 'NOT circ_r_3_1_0']], 'program_bacc': 0.537, 'program_null': 0.484, 'mechanism_level': 'none', 'mechanism_note': 'a7/a6/a3 input writer composition NOT enriched for members: top writer a2 ratio ~1.03-1.05 (bootstrap mean, all 3 components), ENRICHED_STABLE=False everywhere (5-draw bootstrap).', 'behavior_note': 'Machinery pushes CE down (helps) specifically at punctuation targets (39/49=80%, p=0.0, ROBUST_V2 true, vs 51% base rate); elsewhere (space/digit/capitalized/subword/newline) no direction is robust -- push is ~50/50 base rate.', 'behavior_sweep_12pair': {'subword_True': {'n_available': 102, 'population_p': 0.1407, 'population_hits': 58, 'population_n': 102, 'seed_pass_frac': 0.2, 'ROBUST_V2': False}, 'subword_False': {'n_available': 102, 'population_p': 0.8991, 'population_hits': 44, 'population_n': 102, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'space_word_True': {'n_available': 680, 'population_p': 0.9428, 'population_hits': 327, 'population_n': 680, 'seed_pass_frac': 0.2, 'ROBUST_V2': False}, 'space_word_False': {'n_available': 680, 'population_p': 0.0665, 'population_hits': 353, 'population_n': 680, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'digit_True': {'n_available': 24, 'population_p': 0.7624, 'population_hits': 11, 'population_n': 24, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'digit_False': {'n_available': 24, 'population_p': 0.3796, 'population_hits': 13, 'population_n': 24, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'punct_True': {'n_available': 49, 'population_p': 0.0, 'population_hits': 39, 'population_n': 49, 'seed_pass_frac': 0.6, 'ROBUST_V2': True}, 'punct_False': {'n_available': 49, 'population_p': 1.0, 'population_hits': 10, 'population_n': 49, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'capitalized_True': {'n_available': 165, 'population_p': 0.3614, 'population_hits': 87, 'population_n': 165, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'capitalized_False': {'n_available': 165, 'population_p': 0.6951, 'population_hits': 78, 'population_n': 165, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'newline_True': {'n_available': 9, 'population_p': 0.2748, 'population_hits': 6, 'population_n': 9, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}, 'newline_False': {'n_available': 9, 'population_p': 0.8995, 'population_hits': 3, 'population_n': 9, 'seed_pass_frac': 0.0, 'ROBUST_V2': False}}}


**Top members** (context → target, dCE when the circuit is ablated):

- `… skills, ultimately that is what happened, I ran out of fucking` → ` coping`  (dCE -1.59, base CE 4.27)
- `… for the beginner it is a great way to start understanding Archi` → `M`  (dCE -1.8, base CE 4.64)
- `…_________________- woo! so pretty :)\ngivemesomethinganything asked` → ` fuck`  (dCE 1.31, base CE 8.37)


### 60. `r.7.1.1` — a7, concentration 2.98

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: methods-only** — both interventions agree, but the argmax moves when the rows are split. On a held-out row split the argmax moves `a7` -> `a16`. Its held-out concentration is 2.9765, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a7 2.977 | a16 2.785 | a11 2.683 |
| interchange | a7 2.679 | a16 2.535 | a14 2.446 |

At `a7`: mean|dCE| on members **0.6911**, off slice 0.2322, signed dCE on members -0.077. Second-best component is `a16` at 2.785 — a 1.07x margin.


**Story (from the circuit file):** {'blind_name': 'no STRONG single-writer mechanism into a7; weak capitalized-target CE reduction', 'program': [['prev1_starts_space', 'NOT is_punct']], 'program_bacc': 0.54, 'program_null': 0.532, 'mechanism_level': 'none', 'mechanism': {'note': 'No STRONG single-writer mechanism into a7 (top writer m1, bootstrap ratio 1.078 [1.048-1.156], threshold_v2 1.433 = max(1.3, null mean 1.17 + 2sd 0.131), headroom -0.385); downstream top consumer a17 ratio 1.191 also below the 1.3 bar (population gates (a)/(b) FAILED; random-subspace consumer 1.086).'}, 'behavior': "At capitalized-initial targets (137/864 members, 15.9%), ablation raises CE (leaf helps prediction) 90/137 times vs 72.3 expected by the leaf's own base rate (p=0.0015, ROBUST_V2, n_tests=12); no other tested class (subword, space_word, digit, punct, newline; both directions, 12 pairs total) clears the corrected bar -- the leaf's near-even two-signed split (minority_share=0.493) is NOT explained by any class outside capitalized targets.", 'behavior_test': {'kind': 'capitalized', 'pred_help': False, 'n_available': 137, 'per_seed': [{'seed': 1, 'hits': 4, 'n': 5, 'p_value': 0.2241}, {'seed': 2, 'hits': 4, 'n': 5, 'p_value': 0.2241}, {'seed': 3, 'hits': 4, 'n': 5, 'p_value': 0.2241}, {'seed': 4, 'hits': 4, 'n': 5, 'p_value': 0.2241}, {'seed': 11, 'hits': 3, 'n': 5, 'p_value': 0.552}], 'seed_pass_frac': 0.0, 'population': {'n': 137, 'hits': 90, 'base_rate_help': 0.472, 'expected_hits': 72.31, 'p_value': 0.0015, 'beats_base_rate': True}, 'ROBUST': False, 'n_tests': 12, 'alpha': 0.0083, 'ROBUST_V2': True, 'gate_note': 'use ROBUST_V2; ROBUST v1 is underpowered'}}


**Top members** (context → target, dCE when the circuit is ablated):

- `… 19, 1924\nTy Cobb can tell you. Personally causing your` → ` team`  (dCE 2.36, base CE 4.88)
- `…\nIn response to Blind faith won’t make climate science` → ` gaps`  (dCE -1.79, base CE 15.02)
- `…\n[–]Subhazard 21 Punkte22 Punkte23` → ` Punk`  (dCE 2.06, base CE 3.94)


### 61. `r.6.0.0` — a16, concentration 2.88

864 member positions in a slice of 5,760 (15.0% of the slice).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a16 2.884 | a14 2.508 | m16 2.303 |
| interchange | a16 2.633 | a14 2.458 | m16 2.289 |

At `a16`: mean|dCE| on members **0.2143**, off slice 0.0743, signed dCE on members 0.0617. Second-best component is `a14` at 2.508 — a 1.15x margin.


**Story (from the circuit file):** {'blind_name': '', 'program': [['prev1_class_digit']], 'program_bacc': 0.603, 'program_null': 0.502, 'mechanism_level': 'none'}


**Top members** (context → target, dCE when the circuit is ablated):

- `…vern.com; 35 Chalk Farm Rd, NW1` → `;`  (dCE -2.53, base CE 4.48)
- `…  GOOGLE MAP ) ; www.nhm.` → `ac`  (dCE 2.82, base CE 1.74)
- `…gatecemetery.org; Swain's Lane, N` → `6`  (dCE 1.94, base CE 5.35)


### 62. `r.6.2.3` — a9, concentration 2.61

864 member positions in a slice of 5,760 (15.0% of the slice).

> **Confidence: neither** — the named component is not stable across rows and the two interventions disagree. On a held-out row split the argmax moves `a17` -> `a9`. Its held-out concentration is 2.63, so the circuit still localises; it is the single component NAME that is not settled (§2061).

| method | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| mean | a9 2.611 | a17 2.61 | a16 2.533 |
| interchange | a17 2.661 | a16 2.485 | a13 2.441 |

At `a9`: mean|dCE| on members **0.5961**, off slice 0.2283, signed dCE on members -0.0643. Second-best component is `a17` at 2.61 — a 1.00x margin.


**Top members** (context → target, dCE when the circuit is ablated):

- `…\n#### Broadstairs\n\n#### Ramsgate\n\n#### Sandwich` → `\n`  (dCE 5.04, base CE 22.84)
- `…\n\n#### Woburn\n\n#### Waddesdon` → `\n`  (dCE 1.23, base CE 16.09)
- `…\n#### Canterbury\n\n#### Leeds Castle\n\n#### Margate` → `\n`  (dCE -0.5, base CE 23.87)

