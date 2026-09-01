# Rung 444 preregistration: Archetypal SAE in the attention0 downstream-response metric

Status: registered before implementation/execution. GPU only through the managed runner. Identification screen;
no adoption or native-generator removal is authorized.

## Changed object

Rung439 constrained sparse Q/K decoder atoms to convex mixtures of raw folded token-factor rows. It found genuine
token geometry but destroyed computation. Rung442 then proved that raw head-score space is diffuse while rung424's
rank6/rank6/rank32 block succeeds under the two-score product and downstream-response metric.

Rung444 therefore moves the Archetypal constraint to the object where the structure actually appears. For each
realized causal attention0 edge `e=(document,query,source)`, define its exact16-dimensional routed response

`y_e[u] = sum_h score1[e,h] * score2[e,h] * payload[source,h,u]`.

The fixed rung419/424 response Gram supplies a factor `F`; reconstruction loss is

`sum_e ||(y_e-yhat_e) F^T||^2 / sum_e ||y_e F^T||^2`.

This vector already contains both QK branches and the OV payload, and the norm measures their downstream use. It is
gauge-invariant to rotations of the private U16 interface when the vectors and Gram are transformed together.

## Frozen roles and models

- Rebuild rung424's exact U16 interface, response Gram, payload fold, and185,760 FIT/SELECT realized edges from the
  same hash-pinned96+96 documents. FINAL remains closed for this rung.
- Fit32-atom, top4 nonnegative sparse autoencoders for512 Adam steps, batch4096, learning rate.01.
- `U32`: unconstrained response decoder atoms.
- `A32`: each decoder atom is a softmax convex combination of a deterministic2,048-edge FIT support pool.
- `P32`: same convex construction, but its support pool uses an independently permuted source-payload assignment;
  score products, payload marginals, dimensions, optimizer, and support count are unchanged.
- Fit independent seeds444/445 for U32/A32 and one A32 model on each FIT document half. Select seed444 for binding
  consequences; restarts/halves are identification tests, not selectors.

The encoder is one affine16→32 map followed by ReLU and exact top4 masking. There is no decoder bias. At deployment
the direct32×16 response atoms are stored; convex support weights are retained only as a certificate.

## Exact consequences

On SELECT, reconstruct every causal edge, sum reconstructed edges for each query, add the exact native U16 remainder,
and replace only the U16 slice of the native attention0 write. Measure:

- response-metric edge relative squared error;
- routed-U16 R2;
- six native consumer R2 values and their mean;
- suffix CE damage, in nats added above native (lower is better).

Native Q/K, normalization, values, output map, and suffix remain live. This is not a Q/K generator or compression.

## Frozen predictions and null

- A, instrument: exact row hashes/counts; response-Gram reproduction≤1e-6; payload fold≤1e-10; native edge-sum
  identity relative squared error≤1e-10; convex weights nonnegative/sum1 and atom replay≤1e-6; all gradients live,
  losses decrease, roles disjoint, FINAL unopened.
- B, causal-response convex geometry: A32 SELECT response error≤.15 and≤.85×P32; U32≤.12.
- C, identifiability: A32 restart median matched metric-cosine≥.70 and≥U32+.15; A32 FIT-half median matched
  metric-cosine≥.60.
- D, downstream consequence: A32 routed R2≥.90, every consumer R2≥.80, CE damage≤+.005 nat, and A32 is no worse
  than U32 by.10 mean-consumer R2 or.003 nat CE.

Strong null: any instrument failure; A32 error≥.98×P32; A32 restart stability≤U32; or A32 routed R2<.60 / CE
damage>+.020 nat. If B/C/D pass, license a new fresh-corpus response-state family with registered state removals and
composition, not adoption. If the strong null fires, close convex response-edge atoms at K32/top4 and retain the
continuous rung424 quotient.

Literal screen price per stored model:32×16 decoder +16×32 encoder +32 encoder biases =1,056 float32 values =4,224
bytes, plus no native savings; certificates excluded from deployed price but fully serialized. Native generators are
retained. No semantic name is claimed from an atom index alone.
