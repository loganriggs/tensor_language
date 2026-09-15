# Numeric downstream oriented bilinear decomposition V1

R590 tested the exact cached-payload response at MLP8/10/12/14 as a combined
background-cross term, a self term, and their joint response.  The combined
cross term sums two architecture-distinct products.  This experiment splits it
before new outcomes into `left_delta_right_background` and
`left_background_right_delta` while retaining `contrast_self` and
`joint_response` as references.

FIT is the already opened R582 FIT authority.  At each site, source-present and
exact cached-term-deleted trajectories define normalized states `x1` and `x0`
and `d=x1-x0`.  The four frozen response vectors are

* `D[(Ld)*(Rx0)]`;
* `D[(Lx0)*(Rd)]`;
* `D[(Ld)*(Rd)]`;
* their sum.

Their sum must reproduce the direct finite MLP response to relative squared
error at most `1e-10`.  Each vector is removed from the corresponding native
MLP write and scored with the unchanged R582/R590 successor, copy, surface,
conflict, cross-representation, and bootstrap rules.  Candidate order is site
8,10,12,14 and within site left-oriented, right-oriented, self, joint.  The
first FIT candidate passing all ordinary gates must also beat both frozen
active nulls.  Only then may the previously unopened SELECT authority run.

SELECT evaluates all four terms at the selected site under FIT scales and the
same two nulls.  A claim requires the selected oriented term itself to pass all
SELECT ordinary and null gates.  Reference self or joint success is recorded
but does not establish oriented interaction compression.  No coefficient,
fit, learned basis, centering, gain, subgroup, threshold change, gradient,
weight update, or quantization is allowed.  Maximum price is 632 forwards,
zero backwards, and zero fits; FINAL_TEST and OOD remain unopened.
