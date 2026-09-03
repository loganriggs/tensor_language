# Rung 558 preregistration: induction selector-score × payload-value site lattice

**Frozen:** 2026-09-03 17:12 UTC, before R554 outcomes

## Dependency and question

This run is authorized only if the frozen R554 native-capability screen and its R555 audit both hold. Otherwise R558
must not run.

The question is whether the known equality-related attention computations causally carry two different variables:

- a **selector score**, which says which earlier position matches the final query; and
- a **payload value**, which says what token information is copied from that position.

This is below the whole-head boundary. For head $h$, query position $q$, and earlier position $k$, the isolated term is

$$
o_{h,q}=\sum_k \left[p_{h,qk}E_{qk}\right]u_{h,k},
$$

where $p_{h,qk}$ is the head's continuous bilinear attention score, $E_{qk}$ is the exact token-equality support used
to isolate the registered induction term, and $u_{h,k}=W^O_hv_{h,k}$ is the value already projected into the residual
stream. The intervention subtracts this exact term and adds one with the score, payload, or both taken from a paired
donor prompt. All non-isolated attention terms and all subsequent model computations remain live.

The four channels are fixed from earlier equality-subroutine work:

$$
H=\{\mathrm{L5H5},\mathrm{L7H3},\mathrm{L8H3},\mathrm{L8H4}\}.
$$

R558 evaluates every subset $S\subseteq H$. This is not a search over rank, hidden dimensions, or architectural heads
in general; it is an exhaustive causal accounting over the four previously named equality terms.

## Frozen counterfactual families

Only R552 FIT rows may be used to select a subset. SELECT is evaluated once afterward. FINAL_TEST and OOD remain
unopened.

1. **Selector swap:** donor $pE$, base $u$. The desired answer changes because the final query selects the other source.
2. **Payload swap:** base $pE$, donor $u$. The desired answer changes because source followers are exchanged.
3. **Joint answer-preserving diagonal:** donor $pE$ and donor $u$. Each factor alone changes the desired answer, while
   changing both restores the original answer.
4. **Match break:** donor $pE$, base $u$. The donor removes the selected equality match while retaining the payload.
5. **Irrelevant-source edit:** donor $pE$, base $u$. Editing the unselected source should have little causal effect.

The crossed selector/payload arms are required controls: payload transplantation on selector rows and score
transplantation on payload rows should not imitate the requested change.

## Metrics

For an answer-changing row with base answer $b$ and donor answer $d$, define

$$
m(z)=\operatorname{logit}_d(z)-\operatorname{logit}_b(z),\qquad
R_S=\frac{\mathbb{E}[m(z_S)-m(z_\text{base})]}
{\mathbb{E}[m(z_\text{donor})-m(z_\text{base})]}.
$$

The denominator is the complete prompt-swap effect, not a rank or reconstruction score. Bootstrap intervals resample
semantic groups.

For match breaking, $m$ is instead the base-answer minus other-payload margin, and recovery is the reproduced native
margin drop. For the joint diagonal, let $m$ be the unchanged correct-answer minus other-payload margin. If $m_s$ and
$m_p$ are the two single-factor interventions, joint restoration is

$$
J_S=\frac{m_{sp}-\min(m_s,m_p)}{m_\text{base}-\min(m_s,m_p)}.
$$

The complete subset-response table is transformed into Boolean-lattice interaction coefficients

$$
I(S)=f(S)-\sum_{T\subsetneq S}I(T).
$$

These coefficients report redundancy or synergy among the four channels. They are descriptive FIT quantities and do
not select extra sites after SELECT is opened.

## Frozen FIT decision

The full four-channel set must first establish a usable factor-level ceiling. It passes only if, for selector-score,
payload-value, match-break score, and joint restoration separately:

- point recovery is at least $0.30$; and
- the group-bootstrap 95% lower bound is greater than zero.

It must also satisfy:

- absolute crossed-factor recovery at most $0.25$ for selector and payload rows;
- absolute payload-control recovery at most $0.25$ on match-break rows; and
- irrelevant-source score effect at most $0.25$ of the match-break score effect.

If the full set fails, the result is a site-capacity null and no smaller subset is promoted.

If it passes, a subset is FIT-eligible when all four target point recoveries are at least $0.40$, all four bootstrap
lower bounds exceed $0.10$, and the same three selectivity bars hold. Choose the eligible subset with the fewest
channels; break ties by the largest minimum lower bound, then lexicographically by channel name. No other choice is
allowed.

## Frozen SELECT decision

Evaluate only the FIT-selected subset and the full set. The selected subset holds only if all four target point
recoveries are at least $0.30$, all four bootstrap lower bounds exceed zero, and all three selectivity bars remain at
most $0.25$. The full set is reported as the fixed complete-term reference. Any failure is preserved without adding
channels, changing thresholds, or opening FINAL_TEST/OOD.

## Execution and audit requirements

- Replayed unmodified attention must match the native forward within a frozen numerical tolerance before any causal
  result is accepted.
- Artifact hashes, checkpoint hash, row roles, row counts, prompt lengths, factor reconstruction, model forwards, and
  opened splits are recorded.
- Every nonempty subset is evaluated on FIT; the empty subset is the unmodified base response.
- The script must expose a no-model dry run that plants a known eligible subset and exercises selection, scoring, and
  Möbius inversion.
- A separate CPU audit must recompute subset selection, decision inequalities, and the interaction transform from saved
  sufficient statistics before the claim is registered as held or null.

Passing R558 means that specific equality-score and projected-value terms are causally sufficient for the registered
counterfactuals. It does not yet identify a shared Q/K feature basis, compile the result into weights, or establish OOD
generalization.
