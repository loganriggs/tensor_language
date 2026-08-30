# Causal-response factorization v1 — Amendment 15: candidate-freeze v2 recovery

Status: prospective after the v1 freeze published and independent audit returned
NO-GO, before any validation loader or validation/EVAL value was opened.

The v1 artifact is preserved verbatim and is nonpromotive. Although its current 9-rank,
27-program census reconciles exactly, it violated Amendment 14 by including per-program
training scores, reread analysis/terminal/source/program inputs across mutation windows,
and lacked post-link semantic replay and lifecycle race tests. Its exact hashes and
failure reasons are recorded in
`causal_response_factorization_v1_candidate_freeze_v1_failure.json`.

V2 may change only the freeze lifecycle and remove the forbidden score fields. It must:

1. freeze the same union of pooled and robust training frontiers and all three seeds;
2. contain program identity, literal price, bytes, and hashes, but no fit, validation,
   or EVAL score;
3. assert that the published training analysis binds the same grid terminal and grid
   manifest used by the freezer;
4. snapshot all source/input/program bytes, revalidate them immediately before link,
   and revalidate them after link;
5. semantically reload the linked output and replay its logical manifest hash;
6. fail closed under source, analysis, terminal, or program mutation; and
7. add tests for build bindings, pre-link mutation, existing-output corruption, and
   post-link mutation.

The v2 output remains anti-selection infrastructure. It contains no response tensor,
document code, candidate score, selected winner, validation/EVAL access, semantic
claim, or ledger promotion.
