# Single-head regional normalization candidate

Two native residual7 inputs; later suffix external. Only head8.2 attention weights are stored. Normalization uses mixed residual8 plus head8.2, frozen across the intervention. This approximates the full-context response; it is not an exact fold of that response.

Load `program.pt` with PyTorch and call `execute.execute(program, **inputs)`. Supported token IDs are explicitly enumerated; others are rejected.

CPU, native120forward and isolated40fixture replay pass. Isolated imports include no repository modules. Fresh formula evidence: SINGLE_HEAD_FRESH_V1_RESULT.json (all six gates); same endpoints, paired intervention, no composition claim.
