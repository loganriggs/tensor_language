# bilin18 program — state of the decomposition (2026-07-30, reviewed)
## The four-ledger status (see ROADMAP.md §1; never conflate)

**1. REPRESENTATION — complete (architecture identity, verified 1e-7).**
Exact rewriting as a tensor network with analytic scalar-gauge nodes. MLP(rms(x)) = D·T(x,x)/‖x‖²+bias
and pattern = quartic-multilinear/four-norm-gauges. These are method licenses (hold for any weights of
this class), not findings; they license the stream algebra. Confirmed on 3 attention families.

**2. SUBSTITUTABILITY — near-total, whole-model, with fair nulls and SEs.**
- Whole model through per-layer PCA/head bottlenecks (residual truncated): bilin18 +0.047 (no
  committed SE; ~0.003/layer accumulation), 99.4% of uniform-ceiling headroom; head-span nulls
  20-30x, random-576-dim null 100x. [distinct experiment from the chain below]
- Entire MLP stack replaced causally by the composed-fold analytic chain: +0.034 (SE .001) = 99.56%
  of uniform-ceiling headroom, own head-span null 18x; joint substitution ~free once correctly
  scaled (the "gap" was a lambda bug, retracted).
- Architecture-general (the PCA/head-bottleneck test): bilin18 +0.047 / bilin12 +0.116 /
  bilinsm12 +0.077, head-span nulls 10-47x.
- Honest limit: composed forms reference the FULL weight tensors (no compression win — that lives in
  the streams/selection side); data-fit programs are 27x smaller. Fidelity-vs-compression frontier.

**3. FUNCTION — three families + a full programmatic-head map.**
- Atlas: category-prediction (early MLP engine), induction (attention copy), layout; 4-model general.
- Selection census: 23/162 heads programmatic (predicate gain >=5%); the predicate LABEL predicts the
  causal specialization head-by-head (KEY_cap cluster L15-16 = within-capital discriminator +0.046 [NOT a capital-vs-lowercase gate; that is a static prior — §46 correction]; MATCH_same
  anti-self L3H8/L2H5 = induction necessity core, ind-drop 0.94/0.58). Selection ledger and function
  ledger cross-validate.
- Editing: three independent cross-validations of the selection ledger — knockout-vs-predicate,
  the capability dial, and the L3H8 sign-predicted steer (monotone induction control, CE flat).

**4. MEANING — the measured boundary: nameable SELECTION programs over spectral CONTENT dictionaries.**
- Layer-0 content: NOT class-nameable (3/576 coords; median class-R2 0.02); complete description IS the
  exact weight-derived spectrum. Archetype coords ~5x more class-aligned but still fail the ontology
  (2/144); spike codes fail.
- Higher-layer FUNCTIONAL content (L13 opener, L8 successor): nameable as a control-dial + an
  extractable table/predicate over a BOUNDED input set, NOT a generalizing law (opener type-blind &
  leaky; successor per-calibrated-element table, held-out elements fail; category dir steerable but
  not load-bearing). Boundary measured at layers 0/3/8/13.
- The induction MATCH predicate is the one fully meaning-verified functional claim (held-out 98-111%).

## Honest one-paragraph state
bilin18 is representationally exact, ~98-99.8% causally substitutable through PCA-bottlenecked
analytic interfaces (architecture-general; composed-fold fidelity down to ~94% at aggressive one-hop
truncation), functionally mapped into three families with a per-head selection
census whose predicate labels predict causal roles, and semantically it is a set of nameable selection
programs operating over graded, memorized, non-generalizing content dictionaries — with that
nameable/spectral boundary measured, not assumed. Every headline here has survived dedicated
adversarial review (three full passes; one caught a real lambda-scaling bug plus the retracted knob-
recovery numbers, and several framing retractions). Open: content nameability is a spectrum bounded
by function vs lexicality; scale transfer is out-of-ledger (not tested here; Pythia held).
