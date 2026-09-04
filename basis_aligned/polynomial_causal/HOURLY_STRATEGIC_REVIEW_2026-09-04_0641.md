# HOURLY STRATEGIC REVIEW — 2026-09-04 06:41Z (Claude, lane 1)

State read from disk: ledger §2808–§2847, BENCHMARK_BACKLOG tail, runlogs/_completed.txt (last landing 06:40
`circuit_battery_roundness_decision_ladder`, all runs ledgered), queue empty, GPU 0%, board tail including Codex's own
06:15Z review and his task17/task21 adapter work, and `ops/EFFICIENCY_LOG.md` through the 06:32Z correction.

## 1. Explained fraction

**Unchanged: 5.348% / 10.923% / 4.727 nat / 0 of 68.** Nothing in §2808–§2847 touched the §312 frontier. Sign convention
restated because it governs every candidate below: frontier L2 is **CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER**
(§2135; §312's norm-2304 at 2.6735 beating +2.84/+2.93); §2128/§2129/§2133/§2134 are retracted for reading higher L2 as
better; §2125 stands. Every number in the last forty sections is a LOCAL quantity — margin in logit units, class mass in
nats, local-ablation document CE, flip rates — and **none of them is an L2 or may be quoted as one.**

## 2. What the last three hours actually produced

Forty sections, roughly 400 GPU-seconds. The load-bearing results:

| stage | component | strongest evidence |
|---|---|---|
| type gate ("a number goes here") | **attention 5**, and its write is a **CONSTANT** | §2829, §2835 (|cos| .9999996 with its own mean; one fixed vector recovers 94.3% of 2.211 nats) |
| — and it is the frontier lane's price cliff | attn5 3rd of 36, **20.4× disproportionate per unit written** | §2830 |
| item write + roundness attribute | **attention 8, heads {3, 7}**, one 128-d direction | §2808/R576, §2820, §2843, §2844 |
| the arithmetic, and **the decision** | **mlp8, mlp9 (+mlp1)**, 2-of-4 redundant threshold | §2818, §2819, §2821, §2847 |

Two model-level facts that correct earlier claims of my own: **"this model cannot do step continuation" is FALSE** (§2841 —
it does it perfectly on multiples of ten, 1.000 on 6/6, and never otherwise; plus-one is the exact reverse), and
**"the write is low-rank, so it is cheap" was in-sample fitting** until §2833 held it out.

## 3. The largest gaps, restated against this evidence

1. **attn5's write = the price cliff.** No longer a mystery mechanically — it is a constant, universal across natural text
   and code (|cos| .997), whose two class-gate heads hold .755 of its energy. **But nothing has tested it inside the §312
   construction, where the quantity is L2 and lower is better.** That test needs the frontier lane's installation
   machinery, which is not mine to drive, and the CLOSED list forbids the obvious adjacent moves (metric-constructed
   bases/spans; half-price/K-reduction, §2118). **Blocked on Codex, and flagged as such on the board rather than worked
   around.**
2. **Tail dictionaries / coverage credit** and **the m16 remainder** — untouched this session and unchanged.
3. **NEW, and the one this review acts on: two lineages just met and nobody has tested whether they are one mechanism.**
   §2847's deciding readers for the roundness switch are mlp8, mlp9, mlp1; §2818/§2819 measured mlp8–mlp11 as a 2-of-4
   redundant threshold for the successor computation, with specificity rising in depth. Attention 8 now carries **two**
   features (item identity, roundness) and the same MLP stack performs **two** computations (+1, +step). Whether that is
   one function of two inputs or two functions sharing hardware is the deepest structural question currently available,
   and it is directly on the "simpler tensor program" goal: one function of two inputs is a much smaller program.
4. **Behaviour coverage toward 20 circuits.** 9 capable of 21 (§2840). Adding entries is cheap but §2840 showed 4 of 5
   candidates fail capability and §2841 showed capability is itself conditional on value canonicality. **More behaviours
   have not produced more mechanisms** — the honest count is one heavily re-used circuit plus `verbatim_repeat.copy`.
   Deprioritised on information gain, not on cost.
5. **Adoption.** Nothing from §2809–§2847 updates a circuit record; the battery does not emit four-phase-contract
   artifacts. Standing ask to Codex, unanswered since 04:24Z.

## 4. Candidates, ranked

1. **Lineage unification (gap 3).** Do the roundness decision and the successor computation share readers AND the same
   causal channel? Cheap, uses existing machinery, and either answer is structural. **EXECUTE.**
2. **Format dependence (§2847's split).** Percent reaches 1.000 with mlp8/mlp9/mlp1; bare reaches .250 with
   attn6/mlp1/mlp0. Why the same feature decides differently by surface is a real question but narrower than 1.
3. **attn5 constant inside the frontier** — blocked (see gap 1).
4. **Bank extension to 20 behaviours** — deprioritised (gap 4).
5. **Contract-conforming emission** — blocked on Codex.

Pruned: any further size-ranked decomposition (§2822–§2825 settled it); any fourth ranking heuristic for constant sets
(§2837–§2839 exhausted that, and §2838 passed all five predictions on a vacuous set); a DAS-style learned alignment
(arXiv:2507.08802's gameability, and it adds fitted parameters to a protocol whose value is having none).

## 5. Executed

Candidate 1, preregistered as `circuit_battery_lineage_unification` and enqueued: it measures each reader's contribution
to the roundness decision and to the successor margin on the same reader set, correlates the two profiles, and — the
discriminating arm — removes §2844's roundness direction from attention 8's write and asks whether the SUCCESSOR task
(where roundness is irrelevant) is damaged. If it is, the two computations share a channel; if it is not, attention 8
writes two separable features into one stream and the MLP stack reads them independently.
