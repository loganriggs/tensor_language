# Bracket nested-pending OOD capability v1

This capability-only gate freezes a third pending-opener construction before any model score is read. Each target prompt leaves two delimiters pending; the unchanged outer delimiter uses the third type, while the inner stack-top delimiter changes between parenthesis, square bracket, and quote. The correct next closer is the inner delimiter. Controls change the outer pending delimiter while holding the inner delimiter and answer fixed. This differs from the existing single direct opener and completed-then-reopened constructions.

The frozen panel has 72 pairs and 144 endpoints: 36 target pairs balanced at 12 endpoints for each of six ordered closer pairs, plus 36 outer-change controls. One native forward scores all endpoints. No causal intervention, fixed program vector, fit, gradient, update, threshold search, or quantization is allowed.

- **A:** row hash, 72/144 price, target/control balance, six ordered-pair cells, and six control side/answer cells replay exactly.
- **B:** every ordered target cell has native closer accuracy >= `0.75` and positive mean three-closer margin.
- **C:** every answer-preserving control cell has native closer accuracy >= `0.75` and positive mean three-closer margin.
- **D:** observed price is exactly one forward and 144 sequence evaluations, with zero interventions and program installations.

Only an A/B/C/D pass licenses the already frozen six-vector ordered-pair program on these same rows. Capability failure closes this construction without filtering rows or changing templates, delimiter pairs, or bars.
