# Plain-English update — 2026-08-31 13:35Z

(Yardstick: damage = extra prediction error above the real model; LOWER IS BETTER.)

## The afternoon's two instruments paid off
1. **Where the damage flows**: over half of our best config's error travels through attention reading a
   corrupted stream — both attention families (mid-layer "motifs" and tail dictionaries) carry overlapping
   ~1.0-nat shares of the 1.95 total. Retrieval-type predictions suffer most; punctuation is nearly free.
2. **Why circuit certificates stay at 0/62 while total error halves**: we split each circuit's failure into
   BIAS (systematically wrong in one direction — 69% of it) versus NOISE-like dispersion; and we showed the
   dispersion is intrinsic, not fitting noise (averaging four independent fits changed nothing).

## What's running now, mapped to the three properties (per the operator's framing)
- REMOVAL: the full 16-component × 62-circuit knockout matrix — how selectively can each circuit be removed,
  and how much substrate do circuits share (running).
- EXTRACTION fidelity: the cheapest certificate candidate ever — 62 per-circuit bias vectors, 71k values
  total, fit on half the data and scored on the other half (queued).
- MINIMALITY: the first audit — for 12 sampled circuits, does the top component carry >= 70% of the
  knockout damage, or are these genuinely multi-component mechanisms? (queued)

OOD prediction remains the least-developed property (~10%); its first experiment (do circuit fingerprints
transfer to fresh text?) is next in line.
