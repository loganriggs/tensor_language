# Article choice: a/an versus the

## CURRENT tier: 3

The front attention/MLP chain and a compact write subspace are localized, but their
composed quadratic program has not passed sufficiency replay.

## Behavior and tensor program

Endpoint: `logsumexp(logit(a),logit(an))-logit(the)` at article positions, with target
CE and AUC secondary.  Controls are other determiners and matched nonarticles.

Fixed tensor form: exact layer-0 token-pair lookup, MLP0 quadratic gates, and rank-16
article write.  Router: the MLP0 gate values provide continuous context selection; no
post-outcome token-class router may be inserted.  Extraction runs the composed program
in a front-null background; removal deletes its article direction/gates.

## Evidence

- [`article_choice_verify.py`](../../article_choice_verify.py) and result: AUC `0.87`,
  front-attention/MLP drops `0.1671/0.1330`.
- [`article_circuit_depth.py`](../../article_circuit_depth.py) and
  [`article_trigger_trace.py`](../../article_trigger_trace.py): front-chain traces; the
  latter's failed null is not promotive.
- [`article_write_rank.py`](../../article_write_rank.py): rank-16 write evidence.

## Terminal gates

Collateral includes other determiners, prepositions, capitalization, and global CE.
OOD holds out noun bigrams, frequency, vowel/consonant onset, and sentence position.
The trigger null must pass before target gates are scored; default gates then apply.

Shared-owner caveat: layer-0 lookup is shared with previous-token behavior and priced
once.

**Next experiment:** exact attention-0 × MLP0-gate × rank-16 replay and sufficiency curve
against matched random gates.
