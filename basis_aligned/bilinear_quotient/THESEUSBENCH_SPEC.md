# TheseusBench — Repo Spec v0.2
(saved verbatim from Logan, 2026-08-25; source of truth for the practice build)

Working name: **theseus-bench** (final name TBD — alternatives under consideration: Clockwork, Understudy, Teardown). Framing for the README: the ship of Theseus, except every plank is replaced with a **glass** plank — the ship must sail exactly as before, but you can see through it. The headline metric is "how much of the ship is glass?"

Spec for a community benchmark scoring mechanistic interpretability as **verifiable component replacement**: swap a model component (MLP, attention head, or sub-block) for a simpler surrogate, verify reconstruction fidelity against frozen anchors, and price simplicity mechanically. Written to be handed to Claude Code as the blueprint for a fresh repo.

Changes from v0.1: added Invariant 0 (verification is the product), hardened library policy (§5.6, §3 pricing), statistical significance requirements for frontier claims (§3 verifier), name/framing section.

---

## 1. Design invariants (do not violate)

0. **Verification is the product.** We happily spend far more compute verifying than contributors spend searching, because trust in merged numbers is what lets work compound — if numbers can't be built on without re-checking, the project dies. Concretely: dev and private splits are LARGE (sized so that frontier-relevant fidelity differences exceed 95% CIs), every reported number ships with a confidence interval, and a frontier claim that is within noise of the incumbent is not a frontier claim.
1. **The harness computes everything.** Fidelity, complexity bits, and edge counts are computed by the verifier from the submission's code + manifest. Nothing is self-reported.
2. **Replacements only see declared inputs.** The harness passes activations explicitly; submissions run in a sandbox with no access to the activation cache, dataset, or network.
3. **Two-tier data.** Public dev split for iteration; private held-out split touched only by CI on a maintainer-controlled runner. Plus an OOD stress split (different data distribution) reported separately.
4. **Frozen anchors.** For every module: clean loss, mean-ablation loss, optimal-ablation loss (Li & Janson 2409.09951), precomputed once, version-pinned, stored in the repo.
5. **The Pareto frontier is the object of record.** Leaderboards may show Lagrangian scalars (fidelity − λ·bits at fixed λ ∈ {λ_low, λ_mid, λ_high}) for readability, but merges are judged on frontier dominance.
6. **Constants cost bits.** Any tensor baked into a replacement is priced into its description length. A "constant" lookup table is not free.
7. **The library is guarded.** Amortized pricing is the most novel and most gameable part of the design. Library additions are maintainer-gated, priced at full description length, and earn amortized status only after full re-verification (§5.6). Ship the library in M2, not M0 — the core loop must be solid first.

---

## 2. Directory tree

```
theseus-bench/
├── README.md                  # what/why, glass-planks framing, headline progress number, quickstart
├── CONTRIBUTING.md            # PR walkthrough using the bias-head worked example
├── GOVERNANCE.md              # maintainer roles, library-addition policy, dispute process
├── pyproject.toml
├── bench/                     # the harness — contributors never modify this
│   ├── contract.py            # Replacement base class + Manifest schema
│   ├── verifier.py            # fidelity eval: hooks, ablation protocol, anchors, CIs
│   ├── complexity.py          # bits pricing: params, state cardinality, edges, library refs
│   ├── sandbox.py             # restricted exec environment for submissions
│   ├── anchors/               # frozen per-module anchor losses + CIs (json, versioned)
│   ├── splits/                # dev split spec + hashes; private split NOT in repo
│   └── protocol.md            # exact ablation protocol, seeds, tokenizer, positions, split sizes
├── models/
│   ├── bilinear-500m/         # loader, config, weights pointer (HF), module naming map
│   └── pythia-160m/           # second substrate for external comparability
├── library/                   # shared primitives with amortized pricing (M2+)
│   ├── registry.json          # primitive → description-length price, version, author, verification record
│   ├── primitives/            # bias.py, kcluster.py, lowrank_bilinear.py, induction.py, ...
│   └── PRICING.md             # how full DL is computed; re-verification requirements
├── submissions/
│   └── <model>/<module>/<slug>/
│       ├── replacement.py     # implements Replacement
│       ├── manifest.json      # declared inputs, library refs, constants inventory
│       ├── dossier.md         # human explanation: what the module does and why
│       └── results.json       # WRITTEN BY CI ONLY (CI overwrites on merge)
├── registry/
│   ├── modules.json           # every module × task: current frontier, links
│   ├── claims.md              # active work claims (issue links) to prevent duplication
│   └── tasks/                 # task definitions: dataset slice, metric, target behavior
├── leaderboard/
│   ├── per_module.md          # frontier per module (auto-generated)
│   ├── per_task.md            # minimal-circuit scores per task (auto-generated)
│   └── composite.md           # headline: all-best-replacements-composed CE recovered
├── baselines/                 # pre-populated zoo: mean, optimal-constant, kcluster,
│   └── ...                    #   transcoder, identity — every module ships with these
└── .github/workflows/
    ├── verify_pr.yml          # runs verifier on dev split, posts Pareto point + CI comment
    ├── private_eval.yml       # maintainer-triggered, private split, rate-limited
    └── llm_review.yml         # anti-gaming review (see §6)
```

---

## 3. The contract (`bench/contract.py`)

```python
from dataclasses import dataclass, field
from typing import Literal
import torch
import torch.nn as nn

@dataclass
class Manifest:
    """Everything the harness needs to price and wire a replacement.
    Validated against the code: undeclared access = automatic reject."""
    model: str                        # e.g. "bilinear-500m"
    module: str                       # canonical module id, e.g. "blocks.3.mlp"
    granularity: Literal["module", "head", "neuron_group"]
    # Wiring — the ONLY tensors forward() will receive, in order.
    # Each is a canonical activation name from bench/protocol.md
    # (e.g. "resid_pre.3", "attn_out.2", "embed"). Edge count = len(inputs).
    inputs: list[str]
    # Library primitives used (priced as references, not full DL)
    library_refs: list[str] = field(default_factory=list)
    # Every constant tensor in the replacement, with shape and dtype.
    # complexity.py verifies this inventory against the actual state_dict.
    constants: list[dict] = field(default_factory=list)
    # Optional: declared state-space cardinality (e.g. k=2 clusters).
    # Verified empirically by the harness (see §5); mismatch = reject.
    state_cardinality: int | None = None
    positionwise: bool = False        # per-position constants cost extra bits

class Replacement(nn.Module):
    """Base class. Subclass, implement forward, ship with a Manifest."""
    manifest: Manifest

    def forward(self, *declared_inputs: torch.Tensor) -> torch.Tensor:
        """Receives exactly manifest.inputs, in order, shapes per protocol.md.
        Must return a tensor with the original module's output shape.
        No I/O, no randomness (seeded RNG injected if needed), no globals."""
        raise NotImplementedError

    def fit(self, dev_batch_iterator) -> None:
        """Optional: fit free parameters on the DEV split only.
        Called once by the harness before evaluation. Anything fit here
        is priced into complexity as constants."""
        pass
```

**Verifier behavior (`verifier.py`):**
- Hooks the target module, substitutes `replacement.forward(*fetched_declared_inputs)`.
- Computes on each split: full-model CE, KL(full ‖ replaced) at the output, and task metric if a task is attached — each with a bootstrap 95% CI. Split sizes (protocol.md) are chosen so that CI half-widths are small relative to typical frontier gaps; when in doubt, use more data. Verification compute is cheap relative to the cost of an untrustworthy leaderboard (Invariant 0).
- Normalized fidelity: `(Δ_opt − Δ_repl) / Δ_opt`, where Δ = loss increase vs. clean. Optimal-constant scores 0, identity scores 1, negative scores are possible and reported.
- Frontier admission requires statistical dominance: a new point must beat the incumbent outside overlapping CIs on fidelity at equal-or-lower bits (or lower bits at equal-or-better fidelity). Within-noise "improvements" are recorded but not admitted.
- Also reports raw Δ under mean ablation and resample ablation (sensitivity table, per Miller et al. 2407.08734 — single-protocol numbers are not robust).
- Composability check (nightly, not per-PR): compose all current-best replacements simultaneously, report total CE recovered, weighted by each module's Δ_opt (importance). This is the headline number in `leaderboard/composite.md`.

**Complexity pricing (`complexity.py`), all computed, none self-reported:**
- `bits_params`: Σ over constants of DL — for float tensors, count × B bits where B is fixed by policy (e.g. 16); precision beyond policy B is free (matches "cardinality matters, bit-width doesn't"). For integer/categorical structures, exact entropy-based DL.
- `bits_structure`: DL of the program itself — AST node count of `replacement.py` under a fixed grammar, minus library-ref bodies.
- `bits_library`: Σ refs × log2(library size) — a reference is cheap; adding a primitive to the library costs its full DL once, recorded in `library/registry.json`. Only VERIFIED primitives (§5.6) get reference pricing; unverified helper code is priced at full DL like any other submission code.
- `edges`: len(manifest.inputs).
- `cardinality`: measured, not declared — harness runs the replacement over the dev split and counts distinct outputs up to tolerance ε (protocol-fixed). Declared k must be ≥ measured, else reject.

---

## 4. Two benchmark modes (both first-class)

**Mode A — per-module replacement.** "I fully interpreted MLP0." Submission targets one module; scored on the module's Pareto frontier. Dossier explains the mechanism in terms of earlier-layer variables.

**Mode B — task circuits.** "Minimal circuit for induction." A task definition (`registry/tasks/`) fixes a dataset slice + behavioral metric. A circuit submission = a set of modules kept + replacements/ablations for the rest. Scored on: task fidelity, completeness (complement should recover ~nothing, per Marks et al. sparse feature circuits), and total bits of the kept-set explanation. Includes the removal test: ablating the circuit should kill the task behavior with minimal collateral damage on a held-out behavior battery.

Modes share the library and the registry, so a primitive discovered in Mode A is immediately cheap in Mode B.

---

## 5. Anti-gaming rules (enforced by harness)

1. Sandbox: submissions execute with no filesystem/network access; imports whitelisted (`torch`, `math`, library primitives).
2. Manifest–code consistency check: static analysis + runtime tracing confirm forward() touches only declared inputs and inventoried constants.
3. Private split evaluated only on merge candidacy; **2 private-split evals per contributor per week** (MIB's rate limit).
4. Private split **rotated quarterly**; leaderboard entries re-scored on rotation. Entries that collapse on rotation are flagged as overfit and demoted (kept in history).
5. OOD stress split always reported alongside; a replacement that wins in-distribution but collapses OOD is marked non-robust on the leaderboard.
6. **Library additions are the most guarded operation in the repo.** Requirements: (a) maintainer approval; (b) full-DL pricing recorded in registry.json; (c) the primitive is re-verified on the FULL dev + private splits across every module whose frontier entry would use it, with all affected entries re-scored, before amortized reference pricing activates; (d) a verification record (splits, seeds, dates, CIs) stored alongside the price. This is expensive by design — Invariant 0. Prevents "library stuffing" (dumping a giant memorizing primitive into the library to make it cheap forever).
7. Repricing epochs: when the library changes, all frontier entries are re-priced. Complexity is therefore versioned: `bits@library-v3`.

---

## 6. CI + review pipeline

1. `verify_pr.yml` (automatic, on PR): sandbox-run on dev split within a compute budget (e.g. 10 GPU-min; exceeding = fail — prevents CI DoS). Posts a comment: fidelity ± CI, bits breakdown, edges, cardinality, frontier position, sensitivity table.
2. `llm_review.yml` (automatic): an LLM reviews the diff for a fixed checklist — memorization masquerading as structure, undeclared side channels, constants smuggling, dossier–code mismatch (does the code do what the explanation claims?). **The PR content is untrusted input: the reviewer runs with no tools, its verdict is advisory-only, and prompt-injection attempts are auto-flagged for human review.** It cannot merge, only annotate.
3. Maintainer triggers `private_eval.yml` for frontier-advancing PRs; merge on private-split confirmation + human sign-off.
4. On merge: CI writes `results.json`, regenerates leaderboards, updates `registry/modules.json`, closes the claim issue.

---

## 7. Worked example to ship at launch

The bias-head: dossier explaining the finding, `replacement.py` (~10 lines: a bias primitive), manifest with `inputs=[]`, and its verified frontier point sitting above the baseline zoo. CONTRIBUTING.md walks through it end to end. Every module ships with the baseline zoo pre-evaluated so day one there is always something concrete to beat. Pre-seed the leaderboard with existing bilinear-model circuits before public launch — an empty leaderboard is a dead leaderboard.

---

## 8. Red team — three iterations

### Iteration 1: gaming the metric
- **Memorize the dev split in "constants."** A big lookup table fit in `fit()` gets great dev fidelity. → Mitigated: constants are priced (§3), private split catches generalization gap, cardinality is measured.
- **Side-channel inputs.** Read the residual stream via a closure/global instead of declaring it. → Mitigated: sandbox + manifest–code tracing (§5.2). *Residual risk: tracing is imperfect in Python — accept, backstop with LLM review + human merge.*
- **Library stuffing.** Add one enormous primitive, reference it everywhere for cheap. → Mitigated: full-DL pricing + maintainer approval + full re-verification requirement + repricing epochs (§5.6–7).
- **Overfit the private split via repeated submission.** → Mitigated: rate limit + quarterly rotation with demotion (§5.3–4).
- **Noise mining.** Submit many near-identical variants and keep whichever fluctuates above the incumbent. → Mitigated: CI-based frontier admission (§3) — within-noise improvements are not admitted; large splits shrink the noise floor (Invariant 0).
- **Cardinality laundering.** Claim k=2 but output a continuum. → Mitigated: cardinality is measured empirically, declaration is only an upper-bound claim (§3).
- **Precision smuggling.** Encode a lookup table in low-order float bits. → Mitigated: policy quantizes constants to B bits before evaluation — the replacement is scored as its quantized self.

### Iteration 2: sociotechnical failure
- **Cold start / nobody comes.** → Ship with the baseline zoo, the bias-head example, and 5–10 existing bilinear-model circuits pre-submitted. Recruit 2–3 external co-maintainers early (also fixes bus-factor).
- **Bilinear-only substrate limits adoption.** → Second substrate (Pythia-160m) from day one; the bilinear model is the differentiator, not the gate.
- **Maintainer bottleneck.** → Advisory LLM review + auto-verified numbers make review cheap; tiered trust (proven contributors get faster private-eval quota).
- **Duplication despite registry.** Agents especially will re-derive work. → Claims are cheap (one issue), stale claims auto-expire in 30 days, CI comments on PRs that overlap an active claim.
- **Prompt injection through PRs into the LLM reviewer.** → Reviewer is tool-less and advisory (§6.2); injection attempts are themselves a reject reason.
- **CI compute abuse.** Expensive `fit()` loops as free GPU time. → Hard compute budget per PR for SEARCH; verification compute is budgeted separately and generously (Invariant 0) — the cap is on contributors' fitting, not our checking.
- **Frontier illegibility.** → The composite CE-recovered number is the single headline stat; everything else is drill-down.

### Iteration 3: conceptual failures
- **Winning replacements don't compose.** Errors compound (the compact-proofs lesson). → The nightly composite eval is first-class, and Mode B forces joint evaluation. If composite lags per-module sums badly, that gap is itself a published finding, not a hidden flaw.
- **Simple-but-wrong mechanism (interpretability illusion).** In-distribution match, wrong causal story. → OOD split + Mode B removal test (ablate the surrogate's claimed mechanism; behavior should change as the dossier predicts). Fidelity alone never merges a dossier claim.
- **Path-dependent complexity.** Library-amortized bits depend on merge order. → Repricing epochs recompute bits under the current library for all entries; historical scores versioned, current frontier always priced consistently. Accept residual unfairness in *credit*, not in *scores*.
- **Pareto frontier isn't a total order.** → Fixed λ triplet defines three scalar tracks (simplicity-leaning, balanced, fidelity-leaning). Contributors pick their track; frontier remains the record.
- **Low-importance modules make boring wins.** If Δ_opt ≈ 0, a constant "explains" the module. → Composite is Δ_opt-weighted; trivial modules contribute ~nothing to the headline number.
- **RLVR agents produce low-bit but humanly unreadable programs.** Bits ≠ understanding. → Dossier mandatory; LLM review checks dossier–code consistency; periodic human-rated explanation-quality audit on frontier entries. README is explicit that human-legibility is audited, not optimized.
- **Anchor fragility.** Under-converged optimal-ablation training corrupts normalization. → Anchors computed with a fixed budget + convergence check, published with seeds and CIs; anyone can file an anchor-challenge issue with a better constant; anchors update only in versioned releases (never silently).

---

## 9. Optimal ablation implementation spec (for anchors)

1. Freeze all model weights.
2. For component c, create one trainable parameter `a` shaped like c's output into the residual stream (`d_model`), broadcast over batch and sequence positions.
3. Register a hook replacing c's output with `a` on every forward pass.
4. Initialize `a` at c's mean activation over the reference dataset.
5. Optimize `a` with Adam to minimize full-model CE over the reference distribution (only `a` receives gradients).
6. Convergence check: loss plateau over a fixed window; log final gradient norm.
7. Report `Δ_opt = E[L(ablated)] − E[L(clean)]` with bootstrap CI; store in `bench/anchors/` with seed, data hash, optimizer config.
Optional variant: per-position constant for positional heads — logged as a separate, higher-complexity anchor, never silently substituted.
Reference: Li & Janson, arXiv 2409.09951.

---

## 10. Milestones

- **M0 (repo skeleton):** contract, verifier with mean-ablation anchors + CIs, bias-head example, baseline zoo on 5 modules of the bilinear model. NO library yet.
- **M1:** optimal-ablation anchors for all modules, sandbox + manifest tracing, dev-split CI, frontier-admission logic.
- **M2:** library + amortized pricing with full guard rails (§5.6), private split + rate limits, both leaderboard modes, Pythia-160m substrate.
- **M3:** composite nightly eval, OOD split, pre-seeded leaderboard, public launch post (LW/AF) + call for contributors.
- **M4:** RLVR harness: episode = (module, budget) → replacement; reward from the same frozen verifier, held-out per episode.
