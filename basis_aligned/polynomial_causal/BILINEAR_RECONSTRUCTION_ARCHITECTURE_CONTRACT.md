# Architecture contract — bilinear reconstruction pilot

Inspected 2026-09-09: `jacclust/tt_model.py`, `ops/fastload.py`, checkpoint config and
state-dict shapes/coefficients (mmap, CPU). The established loader converts checkpoint
BF16/FP32 tensors to FP32 and puts the model in evaluation mode.

## Actual bilin18

18 blocks; D=1152; 9 heads of width 128; vocabulary=50304; MLP width=4608.
Config selects bilinear_attn=true, squared_attn=true, bilinear=true, gated=false.
This chooses `CausalBilinearSelfAttention.squared_attention`, not the separate
single-QK squared-attention implementation with a row-sum denominator.

Initial state is e=RMSNorm(embedding[token]). Each block first forms
u=lambda[0]*r+lambda[1]*e, then a=Attention(RMSNorm(u),v0), r'=u+a,
and r_next=r'+Down(Left(RMSNorm(r'))*Right(RMSNorm(r')))+Down_bias.
All projection layers are bias-free; the MLP has a separate learned output bias.
Functional RMSNorm uses eps=None, hence torch.finfo(input.dtype).eps; FP32 epsilon
is 1.1920928955078125e-7. There is no learned affine norm parameter here.

Attention projects q1,k1,q2,k2,raw_v from normalized u. Each query/key head is
RMS-normalized separately before RoPE. Width-128 rotations use half-split coordinates:
(x_left*c+x_right*s, -x_left*s+x_right*c). Frequencies originate in FP32;
the native rotary cache casts sin/cos to BF16, even with FP32 activations. This rounding
is part of deployed execution, not an exact real-arithmetic orthogonal rotation claim.

At the first block v0=raw_v before mixing. Each block then uses
v=(1-lamb)*raw_v+lamb*v0, with the same indexed head slice of v0. v0 is token-only
in an unmodified forward: it is computed from normalized mixed embedding state before
any attention. It is not an independent learned token table. Edits to its producer
must propagate to every consumer. Lamb is scalar per layer and unconstrained.

Pattern(t,s)=(q1_t dot k1_s /128)*(q2_t dot k2_s /128) for s<=t and zero otherwise.
It is signed, full-prefix and includes the diagonal. There is no softmax, window,
row-sum normalization or stochastic dropout on this selected forward path.
The accumulator can therefore update before reading, with scaling retained on reads.

The two residual coefficients are independent scalars, not complements. Checkpoint
block 0 is [6.09375,6.09375]; block 1 is [0.0126953125,8.0]. Attention value lambs
range at least from -4.1875 to 4.625; interpreting them as convex weights is incorrect.
Final logits are 30*tanh(lm_head(RMSNorm(r_final))/30). Embedding and lm_head are
separate parameters/storage: neither source constructor nor loader ties them.

No full-size recurrent state is allocated by this pilot. At B=1,T=512, recurrent
attention state alone is 339,738,624 scalars (1296 MiB FP32); the conventional two-key
plus mixed-value cache is 121.5 MiB. This does not include weights/workspace.

## Diagnostic executor versus trained small checkpoint

The tiny reference deliberately uses D=16, H=2, key/value width=4, MLP width=32 and
vocabulary=8. Its projected attention width is 8 rather than D; that is a diagnostic
architecture, not a resized native checkpoint. It retains live norms, independent
residual mixing, first-value mixing, bias, untied output and softcap. Tiny FP64 RoPE
uses FP64 phase tables; native phase rounding is documented separately. No claim of
bitwise native compilation is made from the tiny fixture.

Existing `runs_hop/attn-mlp-attn-rms-seed0` is loaded by `hop_ablate.load` unchanged.
Its spec is attention/MLP/attention, D=128, H=4, head width=32, vocab=29, context=240,
MLP width=512. Source: `deep_model.py`, `model.py`, `hop_data.py`. Its attention has
input RMSNorm but no Q/K head norm, uses the opposite rotary sign convention, and
mixes residuals with torch.lerp(...,scale=0.5). MLP residual is additive. It has no
original-embedding injection, no first-value sharing, no MLP bias and no final norm
or softcap. Its rotary tables were computed in FP32; converting the model to FP64
does not regenerate them. This architecture mismatch is explicit; it is a contextual
product-attention test, not a replication of bilin18's special sharing mechanisms.

Supported pilot masks: inclusive full causal prefix. Sliding-window expiry, arbitrary
pairwise masks, differential softmax attention, training/dropout and arbitrary edits
to hidden native weights have no claimed compiler coverage.

The recurrence is established kernel-attention machinery; see
[Katharopoulos et al.](https://arxiv.org/abs/2006.16236). Observable-preserving linear
closure is inspired by [CLUE](https://arxiv.org/abs/2004.11961), whose polynomial ODE
algorithm is not directly applied to this normalized, discrete transformer.
