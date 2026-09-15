# Bracket layered-pending key-rank-two interaction V1

The rank-two delimiter payload table transferred prospectively from the third to the fourth construction with `0.0100` downstream relative error when multiplied by the exact live donor score. The remaining donor-state dependency is that score. At L13H8 it is exactly the product of two normalized rotary query-key contractions, `p = a*b`.

Freeze a fifth construction before model access in `BRACKET_LAYERED_PENDING_OOD_V1_ROWS.json`: 36 inner-delimiter substitutions and 36 answer-preserving middle-delimiter controls across twelve new lexical groups and a new layered surface. Both directions yield twelve endpoints in every ordered closer-pair target cell. The outer brace, aligned lengths, and final inner-opener positions are fixed within each pair.

Using target endpoints of the already opened third construction and no logits or causal outcomes, independently average the normalized pre-rotary opener keys `k1` and `k2` by delimiter type. For each three-row type table, retain its mean and centered rank-two SVD representation. Re-derive the already confirmed centered rank-two projected-payload table from the same inputs. There is no rank selection or sweep.

On each fifth-construction recipient endpoint, rotate the donor-type key prototypes at the recipient opener position and contract them with the live recipient final query vectors:

- `a_hat = <q1_r, rotate(k1_type(d))>/128`
- `b_hat = <q2_r, rotate(k2_type(d))>/128`
- `p_hat = a_hat*b_hat`
- `u_hat_d = u_r + payload_type(d)-payload_type(r)`
- full donor-free compressed term `t_hat = p_hat*u_hat_d`.

The exact joint ceiling is `p_d*u_d`. Registered arms isolate `p_d*u_hat_d` (payload with exact live donor score), `a_hat*b_d*u_hat_d` (first key only), `a_d*b_hat*u_hat_d` (second key only), both key prototypes, and the recipient-score baseline `p_r*u_hat_d`. All logits move to CPU between forwards. No donor state enters the fully compressed arm; the live recipient query, recipient payload, native suffix, and delimiter label remain inputs. No outcome fit, scalar gain, rank growth, gradients, updates, full-term vectors, or quantization.

Price is eight forwards over 288 combined training/evaluation endpoints, or 2,304 sequence evaluations, plus three fixed rank-two SVDs and zero backwards/updates.

Frozen bars:

- **A — instrument and fifth capability:** replay error at most `1e-5`, exact 8/2,304 accounting, all three centered type tables rank at most two, 72 target and 72 control endpoints, and every native capability cell accuracy at least `0.75` with positive mean margin.
- **B — exact parent:** exact joint effects are donorward on at least `0.90` of each ordered-pair cell.
- **C — key source transfer:** each predicted pre-rotary key versus actual donor key has cosine at least `0.90` and relative L2 error at most `0.50`.
- **D — score transfer:** `a_hat`, `b_hat`, and their product versus exact donor factors each have cosine at least `0.80`, relative L2 error at most `0.60`, and sign agreement at least `0.85` over target endpoints.
- **E — donor-free downstream transfer:** fully compressed versus exact effects have overall cosine at least `0.90`, relative L2 error at most `0.40`, sign agreement at least `0.90`, and norm ratio in `[0.60,1.40]`; every ordered-pair cell has cosine at least `0.75`, relative L2 error at most `0.50`, and sign agreement at least `0.85`.
- **F — key substitution matters:** fully compressed relative L2 error is at least `0.10` lower than the recipient-score baseline.
- **G — selectivity:** fully compressed control RMS is at most `0.50` of exact target RMS.

All predicates passing yields `key_rank2_payload_rank2_interaction_transfer`. A valid capable failure yields `key_rank2_interaction_null` and closes key-prototype compression without refit or rank growth. Failed capability yields `fifth_construction_capability_null`; replay, rank, or accounting failure is invalid.
