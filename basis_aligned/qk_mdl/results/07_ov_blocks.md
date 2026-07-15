# OV circuit + bilinear-MLP interaction blocks (546M, layer/block 0)

Logan's steer: the O projection "needs to hook up somewhere" — its output basis means
nothing alone; fold it against its readers. For a bilinear MLP reading x = n(e + a)
(embedding + attention-out), the hidden splits EXACTLY (frozen empirical rms) into

    (Lx)⊙(Rx) = [Le⊙Re] + [Le⊙Ra + La⊙Re] + [La⊙Ra]
                  self       cross             source-pair

## Block importance (drop one block, ΔCE; split gate exact to 2.4e-7)

| dropped block | ΔCE |
|---|---|
| self (embedding × itself) | +1.291 |
| **cross (current token × attention-out)** | **+0.840** |
| source-pair (attention-out × itself) | +0.187 |

The cross block — Logan's "attention-out conditioned on that specific token" — is a
first-class citizen (~2/3 of self). The near-one-hot intuition is MOSTLY right: the
source×source block is 5–7× smaller, but previous tokens do interact inside the bilinear
layer (+0.19 nats), via the a⊙a term.

## OV sparse-on-its-own: content is NOT coarsely classable

Layer-0 value tables v_h(t) (V×128 per head, the folded OV input; patch replaces layer-0
values everywhere they're used, incl. the v1 share to later layers):

| compression | DL ratio (v-tables) | ΔCE (L2-fit) |
|---|---|---|
| vq64 | 0.003 | +2.019 |
| vq1024 | 0.023 | +0.883 |
| svd16 | 0.125 | +1.295 |
| svd64 (half rank) | 0.501 | +0.114 |
| zero | 0 | +4.362 |

**The selection/content dichotomy:** QK (selection) is a coarse ~256-token-class
computation (vq64 ≈ +0.015 raw, negative CE-trained); OV (content) needs fine token
identity (vq64 +2.02) — the transported content behaves like the raw embedding did in
basis_aligned e6 ("the tokens are the objects"), while selection is classable.

## CE-trained OV tables

<!-- OV_CE_TABLE -->

Next: the V×V cross-block codebook as its own object (token t × transported token s →
hidden), per Logan's suggestion — now justified by the +0.84 block importance.
