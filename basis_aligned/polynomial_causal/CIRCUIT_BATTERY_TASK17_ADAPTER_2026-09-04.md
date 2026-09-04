# Positional-list circuit-battery adapter — 2026-09-04

## Decision this adapter supports

The reusable battery needs a first genuinely new behavior whose counterfactuals are unambiguous. Positional list
retrieval asks whether the model can use an index to select a value from a list, and—only if it can—whether the internal
variable is localized, reused, or split across writers and downstream readers.

This adapter does not run the model and does not claim a circuit. It creates a frozen, inspectable authority that a
later managed capability run may use without changing its examples after seeing model internals.

## Exact generated computation

Let a list contain values

$$
v=(v_0,\ldots,v_{n-1})
$$

and let the query be an index $q$. The registered answer is the lookup function

$$
f(v,q)=v_q.
$$

Each generated group fixes one base list and constructs four counterfactual pairs:

1. **A1 — change the index:** keep $v$ fixed and replace $q$ by $r$. The answer changes from $v_q$ to $v_r$.
2. **A2 — change the selected value:** keep $q$ fixed and swap $v_q$ with one distractor $v_r$. The answer changes
   while list length, token positions, and the query remain fixed.
3. **P — irrelevant-value control:** keep $q$ fixed and replace one $v_j$ with $j\ne q$. The answer stays $v_q$.
4. **C — duplicated-target control:** set $v_r=v_q$ and change the query from $q$ to $r$. The abstract index changes,
   but the observable answer stays the same. At least one other list value remains as a distinct foil.

FIT, SELECT, and TEST use four-item lists. OOD uses six-item lists. All four phases use disjoint payload vocabularies.
The default authority has 24 groups per phase, four rows per group, hence

$$
4\text{ phases}\times24\text{ groups}\times4\text{ transforms}=384\text{ rows}.
$$

## Token and statistical validity

The model uses GPT-2 tokenization. For every base and donor pair, the adapter computes

$$
E(p),\qquad E(p+a),

$$

where $p$ is the prompt, $a$ is the answer, and $E$ is the tokenizer. It requires $E(p)$ to be an exact prefix of
$E(p+a)$ and requires the remaining suffix to contain exactly one token. This checks the answer at the real
continuation boundary; separately tokenizing the prompt and answer would not be sufficient.

Base and donor prompts must also have equal token lengths so the same token position has the same semantic role during
an interchange intervention. Group IDs are shared across A1/A2/P/C, allowing uncertainty intervals to resample whole
generated situations rather than treating correlated transformations as independent examples.

Randomness comes from a SHA-256 digest of the schema, public seed, phase, and group number. It therefore does not depend
on Python's randomized in-process `hash()` function. The complete generated authority receives a canonical SHA-256
digest before any model call.

## Verification receipt

The adapter and the general battery integration tests pass together:

```text
13 passed
```

The default authority contains 384 rows and has content digest
`16307b8bb9273d56f7c3d09cd629fca78fa1db7f110278e959b6ee301cfb7571`.

This licenses only the next CPU-reviewed step: compile a managed **capability-only** FIT invocation and preregister the
accuracy and donor-margin criteria. It does not authorize localization, SELECT/TEST/OOD opening, or a GPU run yet.
