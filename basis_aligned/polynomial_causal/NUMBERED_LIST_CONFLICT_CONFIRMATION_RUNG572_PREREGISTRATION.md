# Rung 572 preregistration: row-level list `+2` conflict confirmation

R569 passed the list conflict but saved only aggregate statistics. R572 independently reevaluates the same FIT and
SELECT conflict prompts and saves each group-by-endpoint margin

$$
m_{\mathrm{conflict}}=z_{\mathrm{final\ label}+1}-z_{\mathrm{arithmetic\ }+2}.
$$

Prediction A: checkpoint, row hashes, sequence count, and at most three forwards are exact; the recomputed mean,
positive fraction, and 2,000-group-bootstrap lower mean match each R569 aggregate within $10^{-6}$.

Prediction B: all 64 FIT endpoint margins are positive and the bootstrap lower mean is positive.

Prediction C: all 32 SELECT endpoint margins are positive and the bootstrap lower mean is positive.

FINAL_TEST/OOD remain closed. This confirmation cannot select a component. Failure blocks localization.
