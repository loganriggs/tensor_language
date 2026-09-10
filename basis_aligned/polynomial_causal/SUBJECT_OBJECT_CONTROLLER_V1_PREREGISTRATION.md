# Verb-conditioned noun-slot selector: native capability and factorial screen

Frozen before trained execution,10 September2026. Original handoff/pilot controls.
This is a new operation proposal, not revival of query-phase/iswas branches.
Existing reflexive-number and object-control-person candidates were inspected;
they do not cross promised/persuaded with two independently numbered human
nouns. Existing perception-complement and duration candidates mention these
verbs but ask different questions. v475's inanimate distractor does not resolve
nearest-human versus subject selection; possessive his/their also admits
nonlocal reference. No existing receipt is retracted here.

Primary linguistic basis: Demestre, On the rapid use of verb-control information
in sentence processing, https://doi.org/10.3389/fpsyg.2023.1320966, introduces
promise as subject-control and persuade as object-control with reflexives.
This supports the stimulus distinction, not a claim about our model's competence.

## Fixed domain and mathematics

16 worlds = eight male noun singular/plural pairs × defend/introduce. Same noun
stem occurs in both slots: e.g. The king promised the kings to defend;
The king persuaded the kings to defend. Cross three independent signs:
c=+1 promised/-1 persuaded, s=subject plural(+1)/singular(-1), o=object number.
Expected reflexive-number sign is y=(s+o+c*(s-o))/2, i.e. s or o selected by c.
The answers are themselves(+1) versus himself(-1). This is a fixed two-slot
construction, not a test of arbitrary parse structures.

All eight corners per world are scored,128 prompts total, none filtered.
Every world is token-length matched; each varied noun/verb is a single token.
The two opposite-number configurations have exactly the same token multiset.
CPU preflight replaced uncle/uncles with son/sons before any model access,
because uncles takes two tokens. Even noun indices define8 fit-design worlds;
odd indices define8 held-design worlds. No fitting occurs in this screen.
Both sets' native outputs become opened evidence; future causal tests may
be intervention-held but cannot call these texts pristine OOD again.

Compute full centered vocabulary logits and signed margin m=logit(themselves)
-logit(himself). The eight Boolean basis terms are1,c,s,o,cs,co,so,cso.
Walsh coefficients are exact finite-grid averages m_term=mean_corners(m*term).
They reconstruct this eight-point table; they do not prove a globally
low-degree native transformer. The ideal declared selector has coefficients
s=o=cs=1/2, co=-1/2 and all others0. Its (1,c) × (s,o) coefficient matrix has
rank2, an exact task-algebra fact rather than a native tensor-rank claim.

## Predictions and decision

A. Instrument: fixed row/source hashes, CPU exact truth-table controls, native
batch answer-margin versus independent full-logit capture maxabs<=1e-3 AND
relativeFrobenius<=1e-5, full-grid coefficient reconstruction maxabs<=1e-9,
finite outputs, hooks restored, exactly8forwards/128sequence evaluations,
zero backwards/fits. Watchdog600s. Supported construction/positions unchanged.

B. Controller capability: in EACH of the16 split-by-corner cells, at least
75% of rows have y*m>0 and mean(y*m)>=1 logit unit. Pooling the eight corners
is forbidden: first-only/nearest-only rules each get75% overall by construction.
A B failure blocks promotion of the proposed controller-selector circuit.

C. Selective number dependence: for each split and each verb, evaluate active
number half-differences with the other number held fixed. Every mean active
half-difference >=1. The RMS inactive half-difference across worlds/active
number settings must be <=.25 times RMS active half-difference. Subject is
active for promised, object for persuaded. RMS avoids cancellation of opposite
inactive effects. This is behavioral factor selectivity, not neural ablation.

D. Shared output-interaction screen: for each split, flatten world-by-vocabulary
cs and co coefficients. Cosine(cs,-co)>=.90 and ||cs||/||co|| in[.8,1.25],
both norms>1e-8. This asks whether complementary gates share a broad output
direction. Passing is an output-level screen, not identified internal reuse.

E/F. Registered opposing mechanisms: apply B's exact per-cell bars with label
o (nearest noun) or s (first noun), respectively. Report separately even if
neither passes. No alternative labels or threshold tuning after outcomes.

If B passes, preserve its C/D outcome and localize verb-dependent number
selection with neural interventions next. If B fails, report the failure and
the registered first/nearest alternatives; do not filter worlds, replace a
verb/action, or weaken a bar to rescue controller selection. A passing E/F
would motivate explaining that actual heuristic on new texts, not calling
the model syntactically competent. Otherwise choose a different operation.

## Price and limits

8 native forwards/128 sequences; one batch16 per corner. Full output array
16×8×50304 FP64 is about49.1MiB. No analysis tensor exceeds256MiB. No new
weights, training, or compression; all545902902 native parameters retained.
Save margin tables and coefficient norms/cosines in JSON, not the full tensor.
This establishes or rejects native capability and a semantic hypothesis.
OOD prediction of a discovered circuit, independent extraction, selective
neural removal and compositional reuse are not established by this screen.
