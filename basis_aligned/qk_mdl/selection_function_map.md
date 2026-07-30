# Selection-vs-function map: 30 programmatic heads (2026-07-30, held FW[448:600])
Predicate = ATTENDED-key class (census v2); cat-top = category whose CE knockout damages most
(function); ind = induction-advantage drop. Two ledgers coincide for copy heads, diverge for others.

## Two CLEAN, tight clusters (selection predicts function directly)
- KEY_cap (L15H3/H4, L16H1/H5): attend capital keys -> predict CAPITALS. Identity; tight
  (capital-selective knockout, other categories ~0 or improve). A distributed late capitalization
  predictor.
- MATCH_same ANTI-SELF (L2H5, L3H8): the induction NECESSITY CORE -- ind-drop 0.58 / 0.94, by far
  the model's largest. Negative same-token coefficient (suppress self-attention) IS the induction
  match; sign causally confirmed by the L3H8 steer.

## Divergent heads (attend-X, predict-Y): the honest majority
- KEY_newline x9: attend newline keys but predict capital/punct/subword, NOT newline (boundary-
  anchor hypothesis FALSIFIED -- damage not post-newline-specific; route open).
- MATCH_prev family (L5H5/L7H3/L12H6/L14H6/L6H5): the induction/succession heads -> predict
  DIGIT/NEWLINE (the succession & list categories, high floor); carry induction (0.08-0.09 the
  strongest live ones).
- PREV1 (L1H3): subword-continuation head -> subword+capital.

## Reading
Selection (WHERE a head looks) predicts function (WHAT it predicts) CLEANLY only for copy-like heads
(KEY_cap, and the anti-self match core). For match/anchor heads the predicted-category profile is
dominated by the high-floor categories their outputs serve (digit/newline/capital), not the attended
class -- the divergence is the mechanism (context-conditioned prediction vs direct copy). Caveat: most
cat-top values are small (0.001-0.035) and mix head-importance with category-difficulty; the two
clean clusters (KEY_cap 0.046 joint, MATCH_same anti-self 0.58/0.94) are the load-bearing findings.
