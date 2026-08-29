# Broad-MLP suffix de-alias v1 — authoritative result

**Measured:** 2026-08-29 00:45 UTC

**Source commit:** `cd9bd33b183c35f342c09d7746f15e98024ff7ed`

**Registered claim:** **FAIL** on both roles and both directed transports

**Claim boundary:** same-corpus finite-replacement CE/top-1 evidence only; no OOD,
semantic-circuit, executable-compression, or whole-model ledger credit

## Plain-language result

The old experiment made the combined late attention-plus-MLP suffix look simple. The
new missing corner shows why that was misleading: late attention cancels most of the
interaction between the early MLPs and the late MLPs.

Replacing only MLP outputs in layers 3 through 8 is extremely damaging. Relative to
the empty-prefix condition, it adds about 10.60–10.87 cross-entropy nats and removes
38.31–41.21 top-1 percentage points. Replacing parts of the early MLP prefix at the
same time strongly mitigates that damage. This mitigation is large and depends on
which early MLPs are replaced.

The old attention-plus-MLP measurements predict the **sign** of all seven effects,
but predict only a small, almost constant response around −0.44 to −0.69 nats. The
actual MLP-only interactions range from −2.20 to −6.95 nats. The registered law is
therefore not a useful quantitative model.

This is not noise. Both independent document populations give almost the same
vectors. The MLP-only interaction profiles correlate 0.9964 between roles; the
three-way profiles correlate 0.9965. What replicates is the failure.

## Registered quantities

For early-prefix mask (P_i), suffix (S\in\{E,A,M,AM\}), and token-weighted CE
(C), define

$$
D_i^S=C(P_iS)-C(P_i)-C(S)+C(E).
$$

The frozen prediction was

$$
\widehat{D_i^M}=D_i^{AM}-D_i^A,
$$

with error

$$
Q_i=\widehat{D_i^M}-D_i^M
   =D_i^{AM}-D_i^A-D_i^M.
$$

Here (Q_i) is the three-way early-prefix by attention by MLP interaction. A small
(Q_i) would mean attention approximately preserves the early-prefix/MLP-suffix
interaction. Instead, (Q_i) is nearly the negative of (D_i^M): cosine −0.9983
and −0.9982 on the two roles. Attention therefore cancels most of the interaction.

## Preregistered CE decision

| role | prediction RMSE | zero-interaction RMSE | NRE | (R^2) | signs | pass? |
|---|---:|---:|---:|---:|---:|---:|
| skip7000 | 3.6961 | 4.1940 | 0.8813 | −4.5227 | 7/7 | no |
| skip11000 | 3.9940 | 4.5178 | 0.8841 | −3.9284 | 7/7 | no |

The required point NRE was below 0.5 and the required (R^2) was above 0.5 with a
positive bootstrap lower bound. Both fail decisively. Bootstrap 95th-percentile NRE
is 0.8852 and 0.8892; the 2.5th-percentile (R^2) is −5.0416 and −4.4385.

Both fixed-source directed transfers also fail:

| source → target | point NRE | bootstrap 95th-percentile NRE | pass? |
|---|---:|---:|---:|
| skip7000 → skip11000 | 0.8909 | 0.8948 | no |
| skip11000 → skip7000 | 0.8738 | 0.8785 | no |

## The actual interaction vectors

The seven entries follow the frozen binary-mask order: MLP0, MLP1, MLP0+1, MLP2,
MLP0+2, MLP1+2, and MLP0+1+2.

### skip7000

$$
D^M=(-2.200,-2.648,-4.411,-2.479,-6.121,-3.189,-6.167)
$$

$$
\widehat D^M=(-0.635,-0.564,-0.607,-0.444,-0.452,-0.442,-0.642)
$$

$$
Q=(1.565,2.083,3.804,2.035,5.670,2.747,5.526).
$$

### skip11000

$$
D^M=(-2.295,-2.736,-4.424,-2.643,-6.666,-3.298,-6.946)
$$

$$
\widehat D^M=(-0.670,-0.601,-0.642,-0.478,-0.479,-0.468,-0.686)
$$

$$
Q=(1.625,2.136,3.781,2.165,6.187,2.830,6.260).
$$

Negative (D^M) means the joint early-prefix-plus-late-MLP replacement is **less
damaging than adding their separate CE effects**. This is a strong compatibility or
compensation effect. Once late attention is also replaced, most of that compatibility
disappears, producing the large positive (Q).

## Top-1 secondary outcome

Top-1 was preregistered as mandatory but non-gating. It tells the same story. The
standalone MLP3–8 replacement costs 38.31 and 41.21 percentage points. Actual early
prefix/MLP interactions recover 22.40–36.30 and 23.93–39.68 points, whereas the old
frozen prediction expects only 3.72–5.14 and 4.01–5.74 points.

## Does a nearly free scale or bias correction rescue it?

No, not as a cell-specific law. This was checked after the registered outcome and is
strictly descriptive.

A scalar multiplier around 7.05 lowers NRE to 0.390–0.409 because it restores the
average magnitude. But its (R^2) remains negative (−0.083/−0.056). Adding a constant
bias gives NRE 0.374–0.396 but (R^2) only 0.007–0.010. The raw predictor and truth
have within-role Pearson correlation only 0.086/0.099. Source-learned scalar or affine
corrections transfer to the other role at NRE 0.376–0.409, still with (R^2\le0).

So a cheap correction can predict the **mean scale**, but cannot predict which early
prefix has which effect. It does not recover the missing structure.

## Consequence for the model story

The broad late attention-plus-MLP suffix should not be treated as a separable MLP
interface. Its apparent simplicity came from a reproducible cancellation between
attention and MLP interventions. The next useful object is therefore the **joint
attention/MLP suffix interaction**, preferably as residual/logit response vectors,
not an independently compressed MLP-only suffix inferred from the combined cells.

This result prunes the frozen attention-invariance hypothesis. It does not invalidate
the retrospective 16-term sparse grammar as a description of the measured grid, but
it prevents promoting that grammar to a causal or executable interface without an
adjacent-cut or vector-valued prospective test.

## Artifact receipts

- measurement receipt file SHA256:
  `513772809648a2344bbf47aa6809627ff0a3f80868861043cc289d617cffb287`
- measurement payload file SHA256:
  `11bc0157372f114c45f2221288a7dfb5724cd9591c406ac7821e97da5dc90dcc`
- score results file SHA256:
  `dc8edddf95f89d9a9cbfb258be3dc0b897ce361f9f67aa1d9848b025a1079b78`
- score receipt file SHA256:
  `6acbdeec6ede6ce7a9a7f5db08023a7bb08bc1f83066768035b01b8d4311719a`
- measurement source closure SHA256:
  `1002e8199a8c6921cd72c952ce63d90a65c46cdc5cd95c85a4cd9ba3f8e004b5`

No failure artifact exists.
