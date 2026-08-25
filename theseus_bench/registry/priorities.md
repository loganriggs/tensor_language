# Priority board — where to start looking

Ranked by **unexplained global CE** = Δ_opt × (1 − best fidelity).
A low-importance head at 100% understood ranks below a big MLP at 50%.
Anchors from the optimal-ablation sweep (57/198 components so far;
attention layers land last). Generated 2026-08-25 20:43 UTC; regenerate with
`python bench/make_priorities.py` after any frontier move or sweep progress.

## Top targets

1. **mlp2** — unexplained 0.726 nats (Δ_opt 0.726, fidelity 0.00) — NO certified stand-in despite front-tier stake
2. **mlp1** — unexplained 0.508 nats (Δ_opt 7.253, fidelity 0.93) — static per-token table held-out, S1088
3. **mlp0** — unexplained 0.091 nats (Δ_opt 0.908, fidelity 0.90) — class-code writer / token map, S905/S1045
4. **mlp6** — unexplained 0.076 nats (Δ_opt 0.076, fidelity 0.00) — baseline zoo only
5. **head0.3** — unexplained 0.062 nats (Δ_opt 0.062, fidelity 0.00) — baseline zoo only
6. **mlp7** — unexplained 0.056 nats (Δ_opt 0.056, fidelity 0.00) — baseline zoo only
7. **mlp17** — unexplained 0.053 nats (Δ_opt 0.332, fidelity 0.84) — fitted linear read, S1131-32
8. **mlp9** — unexplained 0.050 nats (Δ_opt 0.050, fidelity 0.00) — baseline zoo only
9. **mlp8** — unexplained 0.047 nats (Δ_opt 0.047, fidelity 0.00) — baseline zoo only
10. **mlp11** — unexplained 0.046 nats (Δ_opt 0.046, fidelity 0.00) — baseline zoo only

## Full table

| component | Δ_opt | best fidelity | unexplained CE | current best |
|---|---|---|---|---|
| mlp2 | 0.7260 | 0.00 | 0.7260 | NO certified stand-in despite front-tier stake |
| mlp1 | 7.2533 | 0.93 | 0.5077 | static per-token table held-out, S1088 |
| mlp0 | 0.9080 | 0.90 | 0.0908 | class-code writer / token map, S905/S1045 |
| mlp6 | 0.0760 | 0.00 | 0.0760 | baseline zoo only |
| head0.3 | 0.0621 | 0.00 | 0.0621 | baseline zoo only |
| mlp7 | 0.0563 | 0.00 | 0.0563 | baseline zoo only |
| mlp17 | 0.3323 | 0.84 | 0.0532 | fitted linear read, S1131-32 |
| mlp9 | 0.0496 | 0.00 | 0.0496 | baseline zoo only |
| mlp8 | 0.0474 | 0.00 | 0.0474 | baseline zoo only |
| mlp11 | 0.0460 | 0.00 | 0.0460 | baseline zoo only |
| mlp12 | 0.0416 | 0.00 | 0.0416 | baseline zoo only |
| mlp10 | 0.0409 | 0.00 | 0.0409 | baseline zoo only |
| mlp13 | 0.0393 | 0.00 | 0.0393 | baseline zoo only |
| mlp15 | 0.0379 | 0.00 | 0.0379 | baseline zoo only |
| mlp4 | 0.1051 | 0.69 | 0.0326 | lin5 ridge on [attn4,mlp0-3], opt-anchored, S1428/S1433 |
| mlp3 | 0.6099 | 0.95 | 0.0305 | own-basis projection r256, S1130 |
| mlp14 | 0.0301 | 0.00 | 0.0301 | baseline zoo only |
| mlp5 | 0.0821 | 0.65 | 0.0287 | linall+quad ladder, S1434 |
| head2.5 | 0.0284 | 0.00 | 0.0284 | baseline zoo only |
| mlp16 | 0.1399 | 0.81 | 0.0266 | fitted linear read, S1131-32 |
| head1.1 | 0.0259 | 0.00 | 0.0259 | baseline zoo only |
| head4.0 | 0.0133 | 0.00 | 0.0133 | baseline zoo only |
| head1.4 | 0.0126 | 0.00 | 0.0126 | baseline zoo only |
| head4.1 | 0.0114 | 0.00 | 0.0114 | baseline zoo only |
| head3.5 | 0.0109 | 0.00 | 0.0109 | baseline zoo only |
| head3.8 | 0.0106 | 0.00 | 0.0106 | baseline zoo only |
| head2.6 | 0.0102 | 0.00 | 0.0102 | baseline zoo only |
| head2.3 | 0.0094 | 0.00 | 0.0094 | baseline zoo only |
| head1.3 | 0.0094 | 0.00 | 0.0094 | baseline zoo only |
| head2.2 | 0.0083 | 0.00 | 0.0083 | baseline zoo only |
| head1.8 | 0.0073 | 0.00 | 0.0073 | baseline zoo only |
| head3.4 | 0.0065 | 0.00 | 0.0065 | baseline zoo only |
| head2.7 | 0.0062 | 0.00 | 0.0062 | baseline zoo only |
| head3.6 | 0.0057 | 0.00 | 0.0057 | baseline zoo only |
| head3.0 | 0.0055 | 0.00 | 0.0055 | baseline zoo only |
| head1.5 | 0.0051 | 0.00 | 0.0051 | baseline zoo only |
| head2.8 | 0.0043 | 0.00 | 0.0043 | baseline zoo only |
| head0.8 | 0.0043 | 0.00 | 0.0043 | baseline zoo only |
| head0.6 | 0.0041 | 0.00 | 0.0041 | baseline zoo only |
| head3.3 | 0.0039 | 0.00 | 0.0039 | baseline zoo only |
| head1.7 | 0.0034 | 0.00 | 0.0034 | baseline zoo only |
| head0.7 | 0.0034 | 0.00 | 0.0034 | baseline zoo only |
| head2.0 | 0.0027 | 0.00 | 0.0027 | baseline zoo only |
| head3.7 | 0.0026 | 0.00 | 0.0026 | baseline zoo only |
| head3.1 | 0.0024 | 0.00 | 0.0024 | baseline zoo only |
| head0.0 | 0.0018 | 0.00 | 0.0018 | baseline zoo only |
| head4.2 | 0.0017 | 0.00 | 0.0017 | baseline zoo only |
| head1.0 | 0.0016 | 0.00 | 0.0016 | baseline zoo only |
| head0.5 | 0.0015 | 0.00 | 0.0015 | baseline zoo only |
| head3.2 | 0.0014 | 0.00 | 0.0014 | baseline zoo only |
| head1.6 | 0.0010 | 0.00 | 0.0010 | baseline zoo only |
| head0.4 | 0.0010 | 0.00 | 0.0010 | baseline zoo only |
| head2.4 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head2.1 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head1.2 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head0.2 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
| head0.1 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
