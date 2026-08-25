# Priority board — where to start looking

Ranked by **unexplained global CE** = Δ_opt × (1 − best fidelity).
A low-importance head at 100% understood ranks below a big MLP at 50%.
Anchors from the optimal-ablation sweep (198/198 components so far;
attention layers land last). Generated 2026-08-25 22:27 UTC; regenerate with
`python bench/make_priorities.py` after any frontier move or sweep progress.

## Top targets

1. **mlp1** — unexplained 0.181 nats (Δ_opt 7.253, fidelity 0.97) — tok table + residual ridge [attn1,mlp0] + quad, fid_opt, S1438
2. **attn2** — unexplained 0.159 nats (Δ_opt 0.159, fidelity 0.00) — baseline zoo only
3. **attn5** — unexplained 0.136 nats (Δ_opt 0.136, fidelity 0.00) — baseline zoo only
4. **attn3** — unexplained 0.116 nats (Δ_opt 0.116, fidelity 0.00) — baseline zoo only
5. **attn6** — unexplained 0.064 nats (Δ_opt 0.064, fidelity 0.00) — baseline zoo only
6. **attn9** — unexplained 0.064 nats (Δ_opt 0.064, fidelity 0.00) — baseline zoo only
7. **head0.3** — unexplained 0.062 nats (Δ_opt 0.062, fidelity 0.00) — baseline zoo only
8. **mlp0** — unexplained 0.062 nats (Δ_opt 0.908, fidelity 0.93) — tok map + residual ridge [attn0,embed] + quad, fid_opt, S1439
9. **attn7** — unexplained 0.059 nats (Δ_opt 0.059, fidelity 0.00) — baseline zoo only
10. **mlp7** — unexplained 0.056 nats (Δ_opt 0.056, fidelity 0.00) — baseline zoo only

## Full table

| component | Δ_opt | best fidelity | unexplained CE | current best |
|---|---|---|---|---|
| mlp1 | 7.2533 | 0.97 | 0.1813 | tok table + residual ridge [attn1,mlp0] + quad, fid_opt, S1438 |
| attn2 | 0.1585 | 0.00 | 0.1585 | baseline zoo only |
| attn5 | 0.1362 | 0.00 | 0.1362 | baseline zoo only |
| attn3 | 0.1162 | 0.00 | 0.1162 | baseline zoo only |
| attn6 | 0.0642 | 0.00 | 0.0642 | baseline zoo only |
| attn9 | 0.0640 | 0.00 | 0.0640 | baseline zoo only |
| head0.3 | 0.0621 | 0.00 | 0.0621 | baseline zoo only |
| mlp0 | 0.9080 | 0.93 | 0.0617 | tok map + residual ridge [attn0,embed] + quad, fid_opt, S1439 |
| attn7 | 0.0594 | 0.00 | 0.0594 | baseline zoo only |
| mlp7 | 0.0563 | 0.00 | 0.0563 | baseline zoo only |
| mlp9 | 0.0496 | 0.00 | 0.0496 | baseline zoo only |
| attn8 | 0.0479 | 0.00 | 0.0479 | baseline zoo only |
| mlp17 | 0.3323 | 0.86 | 0.0479 | linread+quad, S1443 |
| mlp8 | 0.0474 | 0.00 | 0.0474 | baseline zoo only |
| mlp11 | 0.0460 | 0.00 | 0.0460 | baseline zoo only |
| attn11 | 0.0460 | 0.00 | 0.0460 | baseline zoo only |
| attn4 | 0.2226 | 0.81 | 0.0421 | distance kernel S1446 |
| mlp12 | 0.0416 | 0.00 | 0.0416 | baseline zoo only |
| mlp10 | 0.0409 | 0.00 | 0.0409 | baseline zoo only |
| mlp2 | 0.7260 | 0.94 | 0.0407 | lin2+quad S1437; rank frontier S1440 (r128 .82@7Mbit) |
| attn1 | 0.2186 | 0.82 | 0.0401 | distance kernel S1446 |
| mlp13 | 0.0393 | 0.00 | 0.0393 | baseline zoo only |
| mlp15 | 0.0379 | 0.00 | 0.0379 | baseline zoo only |
| mlp4 | 0.1051 | 0.69 | 0.0326 | lin5 ridge on [attn4,mlp0-3], opt-anchored, S1428/S1433 |
| mlp3 | 0.6099 | 0.95 | 0.0305 | own-basis projection r256, S1130 |
| mlp14 | 0.0301 | 0.00 | 0.0301 | baseline zoo only |
| attn14 | 0.0296 | 0.00 | 0.0296 | baseline zoo only |
| mlp6 | 0.0760 | 0.62 | 0.0289 | ladder linall+quad, S1436 |
| mlp5 | 0.0821 | 0.65 | 0.0287 | linall+quad ladder, S1434 |
| head2.5 | 0.0284 | 0.00 | 0.0284 | baseline zoo only |
| mlp16 | 0.1399 | 0.81 | 0.0266 | fitted linear read, S1131-32 |
| attn10 | 0.0263 | 0.00 | 0.0263 | baseline zoo only |
| head1.1 | 0.0259 | 0.00 | 0.0259 | baseline zoo only |
| head6.3 | 0.0211 | 0.00 | 0.0211 | baseline zoo only |
| head9.7 | 0.0191 | 0.00 | 0.0191 | baseline zoo only |
| head7.8 | 0.0180 | 0.00 | 0.0180 | baseline zoo only |
| attn13 | 0.0164 | 0.00 | 0.0164 | baseline zoo only |
| head7.0 | 0.0136 | 0.00 | 0.0136 | baseline zoo only |
| attn16 | 0.0135 | 0.00 | 0.0135 | baseline zoo only |
| head4.0 | 0.0133 | 0.00 | 0.0133 | baseline zoo only |
| head6.1 | 0.0128 | 0.00 | 0.0128 | baseline zoo only |
| head1.4 | 0.0126 | 0.00 | 0.0126 | baseline zoo only |
| head5.5 | 0.0125 | 0.00 | 0.0125 | baseline zoo only |
| head4.5 | 0.0125 | 0.00 | 0.0125 | baseline zoo only |
| attn0 | 0.2395 | 0.95 | 0.0120 | token+position pattern + per-token value table, sink arc (~1.00 claimed; conservatively seeded) |
| attn17 | 0.0118 | 0.00 | 0.0118 | baseline zoo only |
| head4.1 | 0.0114 | 0.00 | 0.0114 | baseline zoo only |
| head3.5 | 0.0109 | 0.00 | 0.0109 | baseline zoo only |
| head8.3 | 0.0107 | 0.00 | 0.0107 | baseline zoo only |
| head3.8 | 0.0106 | 0.00 | 0.0106 | baseline zoo only |
| head11.2 | 0.0103 | 0.00 | 0.0103 | baseline zoo only |
| head2.6 | 0.0102 | 0.00 | 0.0102 | baseline zoo only |
| head11.6 | 0.0102 | 0.00 | 0.0102 | baseline zoo only |
| attn12 | 0.0095 | 0.00 | 0.0095 | baseline zoo only |
| head2.3 | 0.0094 | 0.00 | 0.0094 | baseline zoo only |
| head1.3 | 0.0094 | 0.00 | 0.0094 | baseline zoo only |
| head4.7 | 0.0089 | 0.00 | 0.0089 | baseline zoo only |
| head5.6 | 0.0085 | 0.00 | 0.0085 | baseline zoo only |
| head5.3 | 0.0085 | 0.00 | 0.0085 | baseline zoo only |
| head2.2 | 0.0083 | 0.00 | 0.0083 | baseline zoo only |
| head5.8 | 0.0080 | 0.00 | 0.0080 | baseline zoo only |
| head6.7 | 0.0076 | 0.00 | 0.0076 | baseline zoo only |
| head14.4 | 0.0075 | 0.00 | 0.0075 | baseline zoo only |
| attn15 | 0.0074 | 0.00 | 0.0074 | baseline zoo only |
| head1.8 | 0.0073 | 0.00 | 0.0073 | baseline zoo only |
| head3.4 | 0.0065 | 0.00 | 0.0065 | baseline zoo only |
| head2.7 | 0.0062 | 0.00 | 0.0062 | baseline zoo only |
| head3.6 | 0.0057 | 0.00 | 0.0057 | baseline zoo only |
| head8.1 | 0.0055 | 0.00 | 0.0055 | baseline zoo only |
| head3.0 | 0.0055 | 0.00 | 0.0055 | baseline zoo only |
| head13.0 | 0.0053 | 0.00 | 0.0053 | baseline zoo only |
| head10.5 | 0.0052 | 0.00 | 0.0052 | baseline zoo only |
| head1.5 | 0.0051 | 0.00 | 0.0051 | baseline zoo only |
| head5.0 | 0.0049 | 0.00 | 0.0049 | baseline zoo only |
| head9.8 | 0.0044 | 0.00 | 0.0044 | baseline zoo only |
| head2.8 | 0.0043 | 0.00 | 0.0043 | baseline zoo only |
| head0.8 | 0.0043 | 0.00 | 0.0043 | baseline zoo only |
| head9.6 | 0.0042 | 0.00 | 0.0042 | baseline zoo only |
| head9.1 | 0.0041 | 0.00 | 0.0041 | baseline zoo only |
| head8.4 | 0.0041 | 0.00 | 0.0041 | baseline zoo only |
| head0.6 | 0.0041 | 0.00 | 0.0041 | baseline zoo only |
| head3.3 | 0.0039 | 0.00 | 0.0039 | baseline zoo only |
| head13.8 | 0.0039 | 0.00 | 0.0039 | baseline zoo only |
| head5.2 | 0.0037 | 0.00 | 0.0037 | baseline zoo only |
| head11.3 | 0.0036 | 0.00 | 0.0036 | baseline zoo only |
| head8.8 | 0.0034 | 0.00 | 0.0034 | baseline zoo only |
| head16.3 | 0.0034 | 0.00 | 0.0034 | baseline zoo only |
| head14.6 | 0.0034 | 0.00 | 0.0034 | baseline zoo only |
| head11.1 | 0.0034 | 0.00 | 0.0034 | baseline zoo only |
| head1.7 | 0.0034 | 0.00 | 0.0034 | baseline zoo only |
| head0.7 | 0.0034 | 0.00 | 0.0034 | baseline zoo only |
| head8.7 | 0.0033 | 0.00 | 0.0033 | baseline zoo only |
| head4.6 | 0.0032 | 0.00 | 0.0032 | baseline zoo only |
| head4.4 | 0.0032 | 0.00 | 0.0032 | baseline zoo only |
| head5.4 | 0.0031 | 0.00 | 0.0031 | baseline zoo only |
| head4.8 | 0.0030 | 0.00 | 0.0030 | baseline zoo only |
| head10.2 | 0.0030 | 0.00 | 0.0030 | baseline zoo only |
| head4.3 | 0.0029 | 0.00 | 0.0029 | baseline zoo only |
| head17.2 | 0.0028 | 0.00 | 0.0028 | baseline zoo only |
| head7.1 | 0.0027 | 0.00 | 0.0027 | baseline zoo only |
| head5.1 | 0.0027 | 0.00 | 0.0027 | baseline zoo only |
| head2.0 | 0.0027 | 0.00 | 0.0027 | baseline zoo only |
| head10.4 | 0.0027 | 0.00 | 0.0027 | baseline zoo only |
| head8.2 | 0.0026 | 0.00 | 0.0026 | baseline zoo only |
| head7.2 | 0.0026 | 0.00 | 0.0026 | baseline zoo only |
| head3.7 | 0.0026 | 0.00 | 0.0026 | baseline zoo only |
| head11.5 | 0.0025 | 0.00 | 0.0025 | baseline zoo only |
| head7.3 | 0.0024 | 0.00 | 0.0024 | baseline zoo only |
| head3.1 | 0.0024 | 0.00 | 0.0024 | baseline zoo only |
| head16.0 | 0.0023 | 0.00 | 0.0023 | baseline zoo only |
| head15.1 | 0.0023 | 0.00 | 0.0023 | baseline zoo only |
| head14.0 | 0.0022 | 0.00 | 0.0022 | baseline zoo only |
| head7.7 | 0.0021 | 0.00 | 0.0021 | baseline zoo only |
| head7.5 | 0.0021 | 0.00 | 0.0021 | baseline zoo only |
| head16.4 | 0.0021 | 0.00 | 0.0021 | baseline zoo only |
| head6.5 | 0.0020 | 0.00 | 0.0020 | baseline zoo only |
| head11.8 | 0.0020 | 0.00 | 0.0020 | baseline zoo only |
| head17.6 | 0.0019 | 0.00 | 0.0019 | baseline zoo only |
| head14.7 | 0.0019 | 0.00 | 0.0019 | baseline zoo only |
| head13.5 | 0.0019 | 0.00 | 0.0019 | baseline zoo only |
| head9.4 | 0.0018 | 0.00 | 0.0018 | baseline zoo only |
| head12.4 | 0.0018 | 0.00 | 0.0018 | baseline zoo only |
| head0.0 | 0.0018 | 0.00 | 0.0018 | baseline zoo only |
| head4.2 | 0.0017 | 0.00 | 0.0017 | baseline zoo only |
| head15.3 | 0.0017 | 0.00 | 0.0017 | baseline zoo only |
| head14.1 | 0.0017 | 0.00 | 0.0017 | baseline zoo only |
| head10.1 | 0.0017 | 0.00 | 0.0017 | baseline zoo only |
| head10.3 | 0.0016 | 0.00 | 0.0016 | baseline zoo only |
| head1.0 | 0.0016 | 0.00 | 0.0016 | baseline zoo only |
| head17.0 | 0.0015 | 0.00 | 0.0015 | baseline zoo only |
| head13.3 | 0.0015 | 0.00 | 0.0015 | baseline zoo only |
| head0.5 | 0.0015 | 0.00 | 0.0015 | baseline zoo only |
| head9.3 | 0.0014 | 0.00 | 0.0014 | baseline zoo only |
| head3.2 | 0.0014 | 0.00 | 0.0014 | baseline zoo only |
| head10.8 | 0.0014 | 0.00 | 0.0014 | baseline zoo only |
| head7.4 | 0.0013 | 0.00 | 0.0013 | baseline zoo only |
| head8.6 | 0.0012 | 0.00 | 0.0012 | baseline zoo only |
| head6.6 | 0.0012 | 0.00 | 0.0012 | baseline zoo only |
| head11.0 | 0.0012 | 0.00 | 0.0012 | baseline zoo only |
| head17.4 | 0.0011 | 0.00 | 0.0011 | baseline zoo only |
| head13.2 | 0.0011 | 0.00 | 0.0011 | baseline zoo only |
| head12.3 | 0.0011 | 0.00 | 0.0011 | baseline zoo only |
| head11.4 | 0.0011 | 0.00 | 0.0011 | baseline zoo only |
| head6.4 | 0.0010 | 0.00 | 0.0010 | baseline zoo only |
| head6.0 | 0.0010 | 0.00 | 0.0010 | baseline zoo only |
| head12.7 | 0.0010 | 0.00 | 0.0010 | baseline zoo only |
| head12.6 | 0.0010 | 0.00 | 0.0010 | baseline zoo only |
| head10.7 | 0.0010 | 0.00 | 0.0010 | baseline zoo only |
| head1.6 | 0.0010 | 0.00 | 0.0010 | baseline zoo only |
| head0.4 | 0.0010 | 0.00 | 0.0010 | baseline zoo only |
| head6.8 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head2.4 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head2.1 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head16.5 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head11.7 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head10.6 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head10.0 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head1.2 | 0.0009 | 0.00 | 0.0009 | baseline zoo only |
| head9.5 | 0.0008 | 0.00 | 0.0008 | baseline zoo only |
| head9.0 | 0.0008 | 0.00 | 0.0008 | baseline zoo only |
| head17.1 | 0.0008 | 0.00 | 0.0008 | baseline zoo only |
| head15.5 | 0.0008 | 0.00 | 0.0008 | baseline zoo only |
| head14.8 | 0.0008 | 0.00 | 0.0008 | baseline zoo only |
| head14.3 | 0.0008 | 0.00 | 0.0008 | baseline zoo only |
| head16.8 | 0.0007 | 0.00 | 0.0007 | baseline zoo only |
| head12.2 | 0.0007 | 0.00 | 0.0007 | baseline zoo only |
| head12.1 | 0.0007 | 0.00 | 0.0007 | baseline zoo only |
| head15.4 | 0.0006 | 0.00 | 0.0006 | baseline zoo only |
| head12.8 | 0.0006 | 0.00 | 0.0006 | baseline zoo only |
| head12.0 | 0.0006 | 0.00 | 0.0006 | baseline zoo only |
| head9.2 | 0.0005 | 0.00 | 0.0005 | baseline zoo only |
| head17.8 | 0.0005 | 0.00 | 0.0005 | baseline zoo only |
| head17.7 | 0.0005 | 0.00 | 0.0005 | baseline zoo only |
| head13.6 | 0.0005 | 0.00 | 0.0005 | baseline zoo only |
| head13.1 | 0.0005 | 0.00 | 0.0005 | baseline zoo only |
| head8.5 | 0.0004 | 0.00 | 0.0004 | baseline zoo only |
| head6.2 | 0.0004 | 0.00 | 0.0004 | baseline zoo only |
| head17.3 | 0.0004 | 0.00 | 0.0004 | baseline zoo only |
| head16.1 | 0.0004 | 0.00 | 0.0004 | baseline zoo only |
| head15.7 | 0.0004 | 0.00 | 0.0004 | baseline zoo only |
| head14.2 | 0.0004 | 0.00 | 0.0004 | baseline zoo only |
| head13.7 | 0.0004 | 0.00 | 0.0004 | baseline zoo only |
| head16.7 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
| head16.2 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
| head15.6 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
| head15.0 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
| head14.5 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
| head13.4 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
| head12.5 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
| head0.2 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
| head0.1 | 0.0003 | 0.00 | 0.0003 | baseline zoo only |
| head17.5 | 0.0002 | 0.00 | 0.0002 | baseline zoo only |
| head15.8 | 0.0002 | 0.00 | 0.0002 | baseline zoo only |
| head15.2 | 0.0002 | 0.00 | 0.0002 | baseline zoo only |
| head5.7 | 0.0119 | 0.98 | 0.0002 | ONE fixed vector (the bias-head), S1089/S1091 |
| head7.6 | 0.0001 | 0.00 | 0.0001 | baseline zoo only |
| head16.6 | 0.0001 | 0.00 | 0.0001 | baseline zoo only |
| head8.0 | 0.0000 | 0.00 | 0.0000 | baseline zoo only |
