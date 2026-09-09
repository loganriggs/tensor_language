# Best-circuit computation notebook: tokens to `is`/`was`

This is a notebook-style reconstruction of the best current temporal circuit. It uses code cells,
saved outputs, actual v23 input rows, and explicit tensor shapes. Nothing below should be read as a
new experiment.

> **Task correction.** This circuit changes `is` versus `was`, which is present versus past tense.
> `is` versus `are` is subject-number agreement and belongs to a different circuit. The two both use
> forms of *be*, but they are not the same computation.

The bottom line is:

- We can execute the four-head circuit **inside the native model** by transplanting its internal
  head outputs. That intervention recovers `72.588%` of the native `is`/`was` change.
- We can write exact equations from those head outputs through RMSNorm and the local MLP11 reader.
- We cannot yet execute this as a standalone token-to-logit program. The native model still computes
  the four heads' query/key routing and most of the downstream residual suffix. Those are the two
  large missing pieces.

## Cell 1 — Load the actual dataset

```python
import sys
sys.path.insert(0, "basis_aligned/bilinear_quotient/ops")

from collections import Counter
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23 as v23

rows = v23.build_rows()
print(len(rows))
print(Counter(row["family"] for row in rows))
print(Counter(len(row["base_ids"]) for row in rows))
print(v23.authority_sha256())
```

```text
64
Counter({'A1': 16, 'A2': 16, 'P': 16, 'C': 16})
Counter({7: 25, 8: 24, 9: 9, 6: 6})
46e9e493a4b2b42ce0c121b30978f08208537300363fa91902184c8f8d6565c7
```

`A1` and `A2` are two target constructions. `P` changes temporal wording without changing the
answer. `C` changes unrelated place wording. All 64 rows enter the causal forward passes. Target
summary metrics use the 30 A1/A2 rows on which both native base and donor predictions were correct:
16 A1 rows and 14 A2 rows. Controls are not correctness-filtered.

The prompts have between six and nine tokens and are padded to a batch width of nine. The final
input token is the semantic intervention endpoint for each row.

## Cell 2 — Inspect the real tokens and answers

```python
fields = (
    "family", "base_text", "base_ids", "base_semantic_position",
    "base_answer", "base_answer_id", "donor_text", "donor_ids",
    "donor_semantic_position", "donor_answer", "donor_answer_id",
)
for row in rows[:4]:
    print({name: row[name] for name in fields})
```

```text
{'family': 'A1',
 'base_text': 'In the present day, the theorist',
 'base_ids': [818, 262, 1944, 1110, 11, 262, 46410],
 'base_semantic_position': 6,
 'base_answer': ' is', 'base_answer_id': 318,
 'donor_text': 'In days gone by, the theorist',
 'donor_ids': [818, 1528, 3750, 416, 11, 262, 46410],
 'donor_semantic_position': 6,
 'donor_answer': ' was', 'donor_answer_id': 373}

{'family': 'A2',
 'base_text': 'In the current age, the theorist',
 'base_ids': [818, 262, 1459, 2479, 11, 262, 46410],
 'base_semantic_position': 6,
 'base_answer': ' is', 'base_answer_id': 318,
 'donor_text': 'In a former age, the theorist',
 'donor_ids': [818, 257, 1966, 2479, 11, 262, 46410],
 'donor_semantic_position': 6,
 'donor_answer': ' was', 'donor_answer_id': 373}

{'family': 'P',
 'base_text': 'In the present day, the theorist',
 'base_ids': [818, 262, 1944, 1110, 11, 262, 46410],
 'base_semantic_position': 6,
 'base_answer': ' is', 'base_answer_id': 318,
 'donor_text': 'At the present time, the theorist',
 'donor_ids': [2953, 262, 1944, 640, 11, 262, 46410],
 'donor_semantic_position': 6,
 'donor_answer': ' is', 'donor_answer_id': 318}

{'family': 'C',
 'base_text': 'Near the harbor, the theorist',
 'base_ids': [40640, 262, 25451, 11, 262, 46410],
 'base_semantic_position': 5,
 'base_answer': ' is', 'base_answer_id': 318,
 'donor_text': 'Near the canyon, the theorist',
 'donor_ids': [40640, 262, 42775, 11, 262, 46410],
 'donor_semantic_position': 5,
 'donor_answer': ' is', 'donor_answer_id': 318}
```

Odd-numbered reporter groups reverse the target direction. For example:

```text
A1: In days gone by, the logician
    [818, 1528, 3750, 416, 11, 262, 2604, 6749]  -> token 373 (' was')
    In the present day, the logician
    [818, 262, 1944, 1110, 11, 262, 2604, 6749] -> token 318 (' is')
```

The complete reporter/direction manifest is:

```text
 0 theorist          present_to_past  is  -> was
 1 logician          past_to_present  was -> is
 2 geometer          present_to_past  is  -> was
 3 algebraist        past_to_present  was -> is
 4 microbiologist    present_to_past  is  -> was
 5 anesthesiologist  past_to_present  was -> is
 6 psychiatrist      present_to_past  is  -> was
 7 educator          past_to_present  was -> is
 8 publisher         present_to_past  is  -> was
 9 author            past_to_present  was -> is
10 playwright        present_to_past  is  -> was
11 screenwriter      past_to_present  was -> is
12 critic            present_to_past  is  -> was
13 essayist          past_to_present  was -> is
14 grammarian        present_to_past  is  -> was
15 semiotician       past_to_present  was -> is
```

The row builder is the complete data authority: every prompt, token ID, answer, semantic position,
row ID, and construction check is available in `rows`. Nothing is sampled at execution time.

## Cell 3 — Model and tensor shapes

```python
config = {
    "layers": 18,
    "residual_width": 1152,
    "heads_per_layer": 9,
    "head_width": 128,
    "mlp_factors": 4608,
    "vocabulary": 50304,
    "batch": 64,
    "padded_tokens": 9,
}
config
```

```text
{'layers': 18,
 'residual_width': 1152,
 'heads_per_layer': 9,
 'head_width': 128,
 'mlp_factors': 4608,
 'vocabulary': 50304,
 'batch': 64,
 'padded_tokens': 9}
```

For the padded v23 batch, the important shapes are:

| object | shape |
|---|---:|
| token IDs | `[64, 9]` |
| residual state at a block boundary | `[64, 9, 1152]` |
| each Q, K, Q2, K2, or V tensor | `[64, 9, 9, 128]` |
| causal attention pattern | `[64, 9, 9, 9]` = `[batch, head, query, key]` |
| all pre-output-projection head results | `[64, 9, 9, 128]` |
| one selected head result | `[64, 9, 128]` |
| four selected head results | four tensors of `[64, 9, 128]` |
| residual write after `c_proj` | `[64, 9, 1152]` |
| MLP11 left and right factor activations | each `[64, 9, 4608]` |
| MLP11 product factors | `[64, 9, 4608]` |
| final logits | `[64, 9, 50304]` |

Padding is an implementation shape. Every score uses the row's true semantic position, so padded
positions do not become examples.

## Cell 4 — The native computation before the intervention

Let $$x_{ell,t}\in\mathbb R^{1152}$$ be the residual state at token position $$t$$ entering block
$$\ell$$. The model first forms the learned residual mixture and RMS-normalizes it:

$$
\widetilde{x}_{\ell,t}
= \lambda_{\ell,0}x_{\ell,t}+\lambda_{\ell,1}x_{0,t},
\qquad
n_{\ell,t}=\operatorname{RMSNorm}(\widetilde{x}_{\ell,t}).
$$

This checkpoint uses double-bilinear, squared attention. For head $$h$$:

$$
q^{(1)}_{t,h}=W^{Q1}_{\ell,h}n_{\ell,t},\quad
k^{(1)}_{s,h}=W^{K1}_{\ell,h}n_{\ell,s},
$$

$$
q^{(2)}_{t,h}=W^{Q2}_{\ell,h}n_{\ell,t},\quad
k^{(2)}_{s,h}=W^{K2}_{\ell,h}n_{\ell,s}.
$$

After headwise RMS normalization and rotary position encoding, the causal routing coefficient is

$$
a_{\ell,h,t,s}
= \mathbf 1[s\le t]
\left(\frac{q^{(1)}_{t,h}\cdot k^{(1)}_{s,h}}{128}\right)
\left(\frac{q^{(2)}_{t,h}\cdot k^{(2)}_{s,h}}{128}\right).
$$

The value also mixes the current layer's value with the first layer's persistent value stream:

$$
v_{\ell,s,h}
=(1-\gamma_\ell)W^V_{\ell,h}n_{\ell,s}
+\gamma_\ell v_{1,s,h}.
$$

The 128-dimensional head result and its 1,152-dimensional residual write are

$$
z_{\ell,h,t}=\sum_{s\le t}a_{\ell,h,t,s}v_{\ell,s,h},
\qquad
w_{\ell,h,t}=W^O_{\ell,h}z_{\ell,h,t}.
$$

Attention writes are added to the residual, followed by a bilinear MLP:

$$
x'_{\ell,t}=\widetilde{x}_{\ell,t}+\sum_{h=0}^{8}w_{\ell,h,t},
$$

$$
u_{\ell,t}=\operatorname{RMSNorm}(x'_{\ell,t}),
\qquad
H_{\ell,t,n}=(L_{\ell,n}^{\mathsf T}u_{\ell,t})
              (R_{\ell,n}^{\mathsf T}u_{\ell,t}),
$$

$$
x_{\ell+1,t}=x'_{\ell,t}+D_\ell H_{\ell,t}+b_\ell.
$$

This matters for interpretation: a head does not simply “contain tense.” Its input computation
uses all preceding token states to build two query/key matches, uses those products to route value
vectors, and then writes the routed result into the shared residual stream.

## Cell 5 — The causal four-head intervention

The identified writers are

```python
routes = ("L8H1", "L9H1", "L9H4", "L11H3")
```

For a base row $$b$$ and matched donor row $$d$$, the actual intervention replaces each selected
pre-`c_proj` head result at every position through the semantic endpoint:

$$
z^{\mathrm{patched}}_{\ell,h,t}
=z^b_{\ell,h,t}+(z^d_{\ell,h,t}-z^b_{\ell,h,t})
=z^d_{\ell,h,t}.
$$

Everything else remains live: unselected heads, output projections, residual additions, all MLPs,
later attention, final normalization, and the vocabulary decoder. In pseudocode:

```python
base_cache = capture_pre_cproj_heads(base_ids)
donor_cache = capture_pre_cproj_heads(donor_ids)

with patch_heads(base_cache, donor_cache,
                 routes=("L8H1", "L9H1", "L9H4", "L11H3"),
                 positions="prefix_through_semantic_endpoint"):
    patched_logits = model(base_ids)
```

This is a causal extraction **interface**, but it is donor-dependent. It proves what changes when
the four internal results are supplied; it does not yet compute those results independently from
the input tokens.

## Cell 6 — Convert logits into the measured behavior

At each row's final semantic position, define

$$
m(x)=\operatorname{logit}_{\mathrm{answer}}(x)
     -\operatorname{logit}_{\mathrm{foil}}(x).
$$

For present-to-past rows the base answer is token 318 (` is`) and the donor answer is token 373
(` was`); the roles reverse for past-to-present rows. The rowwise target and patched effects are

$$
\Delta_i^{\mathrm{target}}=m(d_i)-m(b_i),
\qquad
\Delta_i^{\mathrm{patch}}=m(p_i)-m(b_i).
$$

Across the 30 jointly capable target rows, signed recovery is

$$
\operatorname{recovery}
=\frac{\langle\Delta^{\mathrm{patch}},\Delta^{\mathrm{target}}\rangle}
       {\lVert\Delta^{\mathrm{target}}\rVert_2^2}.
$$

## Cell 7 — Load the saved intervention output

```python
import json
from pathlib import Path

receipt = json.loads(Path(
    "basis_aligned/bilinear_quotient/circuits/followups/"
    "temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
).read_text())

receipt["union_report"]
```

```text
target behavior:
  signed_projection = 0.7258835021
  cosine            = 0.9952411287
  direction_fraction= 1.0
  relative_residual = 0.2831798813

target MLP11 reader response Q:
  signed_projection = 0.6700661758
  cosine            = 0.9767638408
  direction_fraction= 1.0

control leakage ratios:
  P behavior = 0.1003349194    P Q = 0.0834367169
  C behavior = 0.0039624680    C Q = 0.0063966505

exact closure checks:
  self patch maximum error   = 0.0
  full donor maximum error   = 0.0
  donor hidden relative error= 0.0
```

The cosine near one says the four-head intervention moves almost exactly in the correct direction
across rows. The recovery of about `.726` says it has only about 73% of the target magnitude. It is
therefore a large partial circuit, not the complete tense computation.

The confirmation used 19 forwards: native base, native donor, closure controls, and all 15 nonempty
subsets of the four heads. That is 1,216 sequence evaluations and zero fitted intervention
parameters.

## Cell 8 — Do the heads interact or merely point similarly?

The four behavioral Shapley allocations were:

```text
L8H1   0.1483774495
L9H1   0.2173505291
L9H4   0.2266994081
L11H3  0.1334561154
sum    0.7258835021
```

All four contributions are positive. Pair interactions range from about `-0.027` to `+0.020`, much
smaller than the union. Thus the observed behavioral effects are approximately additive under this
particular transplant.

“Approximately additive” does not mean the heads compute the same function. They appear to approach
a similar final behavioral direction through different transport paths:

```text
tokens and earlier residual states
        |
        +-- L8H1 -- blocks 9 and 10 transform its write --+
        +-- L9H1 -- block 10 rotates its write ------------+--> block-11 residual
        +-- L9H4 -- already aligned earlier ---------------+
        +-- L11H3 -- adjacent, live-RMSNorm path ----------+
                                                               |
                                      +------------------------+------------------+
                                      |                                           |
                              small MLP11 branch                     dominant residual suffix
                                      |                                  (reader unresolved)
                                      +------------------------+------------------+
                                                               |
                                                        final is/was logits
```

The shared-weight result is more precise than “similar directions.” After contracting one occupied
MLP11 reader mode, each head has a map $$M_h\in\mathbb R^{1152\times128}$$. Their raw matrices are
not copies, because the 128-dimensional private coordinates have independent gauges. Their physical
context-side Grams $$M_hM_h^{\mathsf T}$$ are similar, supporting

$$
M_h=UA_h+E_h,
$$

where $$U$$ is a shared residual-context direction, $$A_h$$ is a head-private adapter, and $$E_h$$
is a private tail. This is currently a weight-capability grouping result. The common core and private
tails have not yet passed causal swaps, so it would be premature to call $$U$$ the extracted tense
variable.

## Cell 9 — Exact local computation through MLP11

The reader basis has shape `[1152, 4]`. Folding it into the four head/MLP11 paths gives two saved
weight tensors:

```text
cross tensor: [4 heads, 4 readers, 1152 context coordinates, 128 head coordinates]
              [4, 4, 1152, 128]

self tensor:  [4 heads, 4 readers, 128 head coordinates, 128 head coordinates]
              [4, 4, 128, 128]
```

For context $$x\in\mathbb R^{1152}$$, head result $$z_h\in\mathbb R^{128}$$, and reader coordinate
$$a$$, the exact restricted MLP response is

$$
y_{h,a}(x,z_h)
=\sum_{i,k}C_{h,a,i,k}x_i z_{h,k}
+\sum_{k,r}S_{h,a,k,r}z_{h,k}z_{h,r}.
$$

The four per-head formula relative errors are between
$$3.06\times10^{-7}$$ and $$8.78\times10^{-7}$$. This is the strongest current “write out the
computation” result: it is an exact checkpoint-derived local tensor, not an activation regression.

But MLP11 is only a minor branch. When all four heads are removed, restoring the complete induced
MLP11-factor change recovers only

$$
0.1511417768
$$

of the full behavior. One occupied MLP11 mode captures

$$
0.8737570330
$$

of that local MLP11 rescue, but removing that mode directly accounts for only

$$
0.1819655289
$$

of the overall four-head effect. Most of the computation therefore remains in the residual route,
not inside this attractive rank-one local description.

## Cell 10 — Final native decoder

After block 17, the native model computes

$$
r_t=\operatorname{RMSNorm}(x_{18,t}),
$$

$$
\ell^{\mathrm{raw}}_{t,v}=W^{\mathrm{vocab}}_v r_t,
$$

$$
\ell_{t,v}=30\tanh\left(\frac{\ell^{\mathrm{raw}}_{t,v}}{30}\right).
$$

The observed answer is determined by the difference between vocabulary rows 318 and 373 after the
softcap. We know this exact native formula. What is not yet known is the smallest intervening set of
attention/MLP readers that transports the four-head state from block 11 to the final residual in a
selective, replaceable way.

## Cell 11 — What has actually been extracted?

| stage | extracted status | what remains native |
|---|---|---|
| Input rows and tokens | Complete | Nothing for the v23 data authority |
| Four causal writer sites | Strongly identified | Other writers contributing the missing 27% |
| Writer output intervention | Executable with donor cache | Q/K/Q2/K2/V computation that produces each donor head result |
| Writer-to-block-11 transport | Partly localized | A compact common transport program across all four routes |
| RMSNorm at L11H3 | Exact finite formula | Nothing locally |
| MLP11 reader branch | Exact restricted tensor, causally minor | Most downstream mediation |
| Dominant downstream residual reader | Not identified yet | A12/M12 through A17/M17 localization |
| Final RMSNorm/unembedding/softcap | Exact native formula | Compact circuit-specific suffix implementation |
| Standalone token-to-logit program | **Not yet available** | Input routing plus dominant downstream suffix and missing effect |

So the honest executable object today is:

```python
def current_partial_circuit(base_tokens, donor_tokens, native_model):
    # Native background computes all token embeddings, Q/K routing, values, and non-circuit modules.
    donor_head_results = capture_heads(
        native_model, donor_tokens,
        sites=("L8H1", "L9H1", "L9H4", "L11H3"),
    )
    # The causal circuit interface installs those four results into the base computation.
    return run_with_head_patch(native_model, base_tokens, donor_head_results)
```

The desired standalone object is instead:

```python
def future_extracted_circuit(base_tokens, requested_tense_state):
    routed_head_state = compact_input_router(base_tokens, requested_tense_state)
    block11_interface = compact_writer_transport(routed_head_state)
    final_state = compact_downstream_reader_program(block11_interface)
    return exact_is_was_decoder(final_state)
```

The second function is a specification, not implemented evidence.

## Cell 12 — Experiments that close the computation story

1. **Input-side source and routing atlas.** For each of the four heads, decompose the causal result by
   source token and by Q1K1 versus Q2K2 versus value contribution. Require held-construction
   prediction and patch/reset evidence. This answers which input tokens and features produce the
   tense write.
2. **Dominant downstream reader atlas.** The queued reciprocal module test searches A12/M12 through
   A17/M17 using both transfer sufficiency and reset necessity. This identifies who reads the shared
   block-11 state.
3. **Exact branch closure.** Residual-only plus MLP11-only restoration must reconstruct the joint
   effect. Otherwise the branch accounting interface is wrong.
4. **Minimum causal subspace.** Once the allowed state changes and downstream readers are fixed, use
   the exact minimum-rank certificate

   $$
   k_{\min}=\operatorname{rank}(RV),
   $$

   and test the resulting projector with held-out swaps. This is the principled alternative to
   declaring DIM's rank-one mean direction sufficient.
5. **Standalone extraction and price.** Replace both the input router and downstream reader with
   checkpoint-derived tensors, compose them, and compare exact storage/compute plus OOD intervention
   behavior against the native path.

Until those pass, the strongest accurate claim is: **four distributed full-head writer outputs form
a selective, necessary, high-magnitude causal interface for `is`/`was`, with exact local tensor
semantics at MLP11, but the complete token-to-logit circuit has not yet been extracted.**

## Evidence files used by this notebook

- Dataset builder: `basis_aligned/bilinear_quotient/ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py`
- Native capability: `basis_aligned/bilinear_quotient/circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v23_capability_v1_result.json`
- Four-head intervention: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json`
- Necessity and MLP11 rescue: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_four_head_necessity_occupied_mode_rescue_v1_result.json`
- Exact RMSNorm transport: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_l11h3_exact_rmsnorm_transport_v1_result.json`
- Restricted weight tensor: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1_result.json`
- Writer grouping: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_grouping_v1_result.json`
