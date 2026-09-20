# Head 1.8: low replacement loss does not predict removal

The frozen negative-prefix-mean approximation fails to predict native head1.8 removal effects on all four tested families. Relative centered-logit errors are2.0–5.7 and the repetition-family cosine is negative. Some replacement CE changes remain tiny; that is not evidence of faithful causal extraction. The candidate is not promoted to a circuit.

## Counterfactual specification

Write the native head as m+e, where m is the parameter-free negative prefix mean and e is its exact remainder on the current input. Let F include the remaining model, norms and softcap with the surrounding state fixed at the intervention site. We evaluate all four states: F(b+m+e) native, F(b) true head removal, F(b+e) subtract-program prediction, F(b+m) program-only replacement. Each arm recomputes the entire suffix.

True removal effect is F(b)-F(b+m+e); the predicted effect is F(b+e)-F(b+m+e). Their error is F(b+e)-F(b). Replacement fidelity instead tests F(b+m)-F(b+m+e). These are different backgrounds for the remainder. There is no general implication through a nonlinear suffix.

An executed CPU control uses F(x)=x², b=0, m=1, e=-2: replacement is exact, actual removal effect=-1, predicted removal effect=+3. This is a mathematical counterexample, not a model of the native head.

## Results

Logit effects are centered over vocabulary per token after native softcap. Relative error is ||predicted-actual||/||actual||, and cosine retains direction. NLL MAE is the mean absolute per-token error in the predicted next-token loss effect (nats). Sign agreement excludes true NLL effects below0.01nats.

| Family | Tokens | Logit-effect relative error | Cosine | NLL effect MAE | NLL sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 224 | 3.292 | 0.703 | 0.292 | 0.839 |
| arithmetic | 128 | 4.733 | 0.289 | 0.395 | 0.577 |
| repetition | 203 | 5.692 | -0.051 | 0.124 | 0.444 |
| opened_prose | 512 | 1.987 | 0.581 | 0.228 | 0.761 |

The preregistered limits were relative error<=.20, cosine>=.90, NLL MAE<=.02 in every family. All three substantive gates fail; the algebraic intervention check passes. No parameters were fitted.

| Family | Replacement CE added | True removal CE added | Predicted removal CE added | Mean/remainder CE interaction |
| --- | ---: | ---: | ---: | ---: |
| code | 0.0426 | -0.0074 | 0.0909 | -0.1409 |
| arithmetic | 0.0015 | -0.0054 | 0.2764 | -0.2833 |
| repetition | -0.0118 | 0.0038 | 0.0545 | -0.0389 |
| opened_prose | 0.0084 | 0.0008 | 0.0745 | -0.0821 |

The interaction is L(native)-L(program only)-L(remainder only)+L(zero head), in nats/token. It measures a nonlinear interaction of these two proposed pieces through the full suffix and loss; it is not a coefficient of the local bilinear layer. Negative CE added means improvement on these rows. Small mean CE changes can conceal substantial token-level discrepancies and sign errors.

## Evidence boundaries and next folding target

Four new constructed prompts per code/arithmetic/repetition family test shifts from prose for this frozen candidate. Each family shares a template; this is narrow template-level evidence, not broad OOD generalization or proof of absence from pretraining. Four prose rows come from a previously opened panel. Token hashes and full prompts are in the receipt.64native forwards,0fits/updates. Raw squared norms and signed inner products are retained; failure is not diagnosed from relative errors alone.

Low CE justified exploring the running-mean program but does not justify calling it an exact native computation or a faithful removable replacement for the whole head. The operation remains independently executable at a declared mixed-value port. Its exact decomposition with a remainder is tautological; the remainder cannot be omitted in a causal claim. No conclusion that all mean-based circuits are useless follows.

The large interaction suggests the next algebraic target: fold the mean and remainder writes jointly through the following bilinear MLP, retaining self and cross terms and explicit RMS denominators. First localize where their interaction appears; do not promote an approximate head solely because replacement loss is small. Semantic preservation, useful reuse, and a simpler complete circuit remain unverified.

[Native receipt](../bilinear_quotient/circuits/followups/mean_head_effect_v637_result.json) · [factorial CE analysis](MEAN_HEAD_EFFECT_FACTORIAL_V637.json) · [counterexample](REPLACEMENT_VS_REMOVAL_CONTROL.json)
