# The global residual-basis gauge

## Exact architecture symmetry

Bilin18's 1152-dimensional residual coordinates are not intrinsically identified.
For any orthogonal matrix `Q`, replace every residual row state by `xQ`. RMS
normalization commutes with this change because `||xQ||_2=||x||_2`.

The rest of the network transforms by tensor-leg variance:

| Object | PyTorch/storage shape | Gauge transform |
|---|---|---|
| token embedding, unembedding, residual bias/table | `[N,D]` | `M -> M Q` |
| Q/K/V, MLP Left/Right residual readers | `[out,D]` | `W -> W Q` |
| attention output and MLP Down writers | `[D,in]` | `W -> Q^T W` |
| quadratic input factors | `[D,k]` | `A,B -> Q^T A,Q^T B` |
| quadratic output factors | `[k,D]` | `C -> C Q` |
| residual/attention scalar schedule | scalar | unchanged |

RoPE and product routing act in head-internal coordinates after the residual reader,
so they do not transform. Residual additions remain valid because every writer lands
in the same rotated space. Finally,

\[
(xQ)(UQ)^\top=xU^\top
\]

for unembedding rows `U`, so logits are unchanged. A randomized float64 network test
exercises RMS, Q/K/V product routing, the attention writer, a bilinear MLP, residual
mixing, and unembedding; states rotate by `Q` and logits agree to `1e-10`.

This removes `D(D-1)/2 = 662,976` continuous gauge degrees at `D=1152`. That count
is a symmetry dimension, **not** a bit saving.

## A generic canonical section

Choose a tall, full-column-rank residual anchor such as the token embedding `E` and
compute its thin SVD. With distinct nonzero singular values, its right singular frame
defines `Q`; signs are fixed by requiring the largest-magnitude entry of each
canonical anchor column to be positive. If the original program is first rotated by
`R`, the new frame is `R^T Q`, hence

\[
(ER)(R^TQ)=EQ.
\]

The implementation verifies this on randomized rotations. It fails closed on rank
deficiency or repeated singular values because the frame is then non-identifiable.
Those strata require block-subspace canonicalization rather than arbitrary numerical
eigenvectors.

## Consequence for simplicity

Independent per-module compressed lengths are basis-dependent until this shared gauge
is fixed. A rotation can make one tensor easier to compress while making its readers
or writers harder, without changing the model. Quotient-aware accounting should:

1. choose and version one global canonical residual frame;
2. transform every candidate and retained shared object consistently;
3. serialize the frame rule once, not an independent basis per module;
4. then apply local CP/SVD/head gauges and codecs;
5. reject or separately price singular-stratum fallbacks.

The current 12.59-Gbit hybrid ledger remains a valid checkpoint-basis conditional
payload, but it is not invariant under this global symmetry and therefore is not a
global quotient price.

There is also a numerical boundary. The real-valued tensor program is exactly gauge
equivalent, but rotating stored bfloat16/float32 tensors and requantizing them need not
bit-replay the checkpoint. Any production canonical codec must measure that distortion
and jointly reverify behavior; algebraic equivalence alone cannot inherit operational
scores.

## Audit against the actual forward

`residual_basis_architecture_contract.json` pins both the production model source and
the independent reference forward by SHA256. Its static audit closes the concrete
operations rather than extrapolating from the tiny model:

- the token embedding is RMS-normalized before becoming `x0`;
- every block performs only scalar affine residual recurrence before residual RMS;
- Q/K/Q2/K2 and V are linear residual readers;
- head RMS and RoPE occur after those readers in unchanged head coordinates;
- the block-0 V tensor is shared and mixed only in unchanged head coordinates;
- `c_proj` writes routed values back to the residual space;
- Left/Right read the RMS-normalized post-attention residual and Down/bias write it;
- final RMS precedes the unembedding, and the tanh cap acts only on invariant logits.

The source audit fails on hash drift, missing equations, or promotion of any unsupported
claim. It certifies exact covariance over real arithmetic. It deliberately does not
certify transformed checkpoint bit replay, float32 logit identity, applicability of
the embedding-SVD generic stratum to the checkpoint, or a global quotient price.

## Literature mapping

[Pérez-García, Verstraete, Wolf, and Cirac](https://arxiv.org/abs/quant-ph/0608197)
derive representation freedom and canonical forms for matrix product states. The
relevant lesson is that internal tensor coordinates are gauges and must be fixed
before comparing representations. Bilin18 is a recurrent residual tensor network,
not an injective MPS, so their MPS uniqueness theorem is not invoked here.

[Acuaviva et al.](https://arxiv.org/abs/2209.14358) develop minimal canonical forms
for broader tensor-network gauge actions and emphasize orbit closure/singular cases.
Our embedding-SVD section is a simpler architecture-specific construction for the
compact global group `O(D)`; it does not solve the model's additional local CP or
attention gauges.
