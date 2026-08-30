# Two-hour section deadline

- Start: `2026-08-30T03:46:40Z`
- Hard pivot: `2026-08-30T05:46:40Z`
- Scope before pivot: finish the exact MLP0 token/context tensor factorial, preserve its numerical result or precise failure, and consolidate the conclusion.
- Scope after pivot: stop expanding this section and begin the ten-circuit discovery/certification campaign.
- Alarm sentinel: `.two_hour_section_deadline_reached` in this directory.
- Persistent fallback alarm (the container has no systemd bus): detached process
  `3878889`, installed at 2026-08-30 04:19 UTC after the original exec-scoped process
  exited. It touches the same sentinel at the registered cutoff.

The deadline is a scope cutoff, not a license to claim an unfinished experiment succeeded. At the cutoff, any remaining failure is preserved and work moves to circuits.

## Section wrap-up (completed before the cutoff)

The MLP0 token/context factorial finished in 44.69 seconds on 96 FIT plus 96
document-disjoint SELECT documents.  The exact three-way split replays the folded
tensor to relative MSE `3.11e-13` (`5.48e-6` after bf16 execution).  SELECT Shapley CE
allocations were `1.177809` nat to the context-context branch, `0.928074` to the
token-token branch, and `0.400776` to the token-context cross branch.  The large pair
interactions (`+1.721576`, `-1.153678`, and `-1.032800` nat) reject three independently
swappable compressors.  The useful structural conclusion is a shared lexical/token
DAG coupled to a continuous contextual tensor, with joint downstream pricing.

The section is therefore closed; the ten-circuit campaign is already active.  The
sentinel remains useful only as a hard guard against reopening this section before the
registered cutoff.
