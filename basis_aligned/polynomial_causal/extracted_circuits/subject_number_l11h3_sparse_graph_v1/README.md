# Subject-number L11H3 sparse graph V1

This package exports a nine-edge causal graph for subject–verb number agreement.
Five native ports at the pre-L11 subject position are used:

1. embedding recurrence;
2. layer 0–3 writes;
3. layer 4–7 writes;
4. MLP8;
5. MLP10.

The graph contains all five main effects plus four pair edges:
`MLP8×MLP10`, `middle×MLP8`, `embedding×MLP8`, and `middle×MLP10`.
`node.py` extracts these edges from ten required intervention-corner margins and
composes their predicted joint damage. No coefficient is learned.

The bound token executor derives the five ports internally from token IDs and
checkpoint weights. On the original crossed authority the frozen graph has
6.34–6.79% component-relative error. On disjoint nouns and new two-attractor
templates it has 4.71–5.35% error. Fresh removal damage is positive on all 32
prompts, 14.63 times the median of equal-L2 same-site controls, with unrelated
`can`/`will` collateral at 3.51% of target damage.

The graph meets the registered OOD, extraction, selective-removal, and
composition/reuse tests. Its limitation is computational rather than causal:
the token executor still runs exact checkpoint layers to construct native ports
and suffix corners, so this is not yet a FLOP-compressed replacement model.
