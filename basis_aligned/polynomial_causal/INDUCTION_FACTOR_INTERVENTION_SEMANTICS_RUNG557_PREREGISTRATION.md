# Rung 557 preregistration: selector/payload factor-intervention semantics

**Frozen:** 2026-09-03 16:58 UTC, before executing the planted computation

## Purpose

Validate that the R552 counterfactuals support separate score and payload interventions before implementing those
interventions inside bilin18. This is an instrument test, not evidence about the trained model.

For token sequence $t_0,\ldots,t_q$ ending at query position $q$, define an equality score over possible payload
positions $k$:

$$
p_k=\mathbf{1}[1\leq k\leq q]\,\mathbf{1}[t_{k-1}=t_q].
$$

Let $u_{k,v}=\mathbf{1}[t_k=v]$ be a one-hot payload value. The planted equality fetch is

$$
o_v=\sum_k p_k u_{k,v}.
$$

This is the discrete version of the trained attention circuit's score-times-value contraction. It has no learned
parameters and cannot inspect the registered answer field when computing $p$, $u$, or $o$.

## Frozen checks

Across all 180 groups and four splits:

1. Every one of the 720 factorial conditions has exactly one selected earlier payload and $\arg\max_v o_v$ equals
   the registered answer.
2. For both directions of every `two_valid_sources_selector_swap` row, transplanting the donor score $p'$ while
   retaining target payloads $u$ makes the planted output equal the donor answer.
3. For both directions of every `payload_swap_match_preserved` row, retaining the target score $p$ while transplanting
   donor payloads $u'$ makes the output equal the donor answer.
4. For both directions of every `selector_payload_joint_answer_preserved` row, transplanting both $p'$ and $u'$ makes
   the output equal the unchanged registered answer.
5. Every match-breaking donor removes the selected equality edge; transplanting the base score into the donor restores
   the registered answer.
6. Every irrelevant-source edit leaves the equality score exactly unchanged.

All checks must pass exactly, with no tolerance. The script must bind the R552 rows hash, process 1,800 registered pair
rows and 720 unique factorial conditions, load no model, execute zero model forwards/backwards, and record
`outcomes_opened=[]`. A failure blocks the proposed model-facing score-versus-payload intervention.
