# Literal hop instruction at the final-layer boundary

The separate Q1/Q2 hop-control hypotheses fail. A single address perhead also
fails a best-case causal-response bound, and nativeonehop read aftergraphadvance
does not reproduce higherhop binding reads. Preserve these as closed proposals.

Test a different boundary: can prefix content be reused across hop requests,
with the literal hop embedding controlling the final layer? For inputs differing
only in their last hop token, postL2 states at all earlier positions coincide.
The exact residual expansion contains E(hop)/8 at the finalquery. Define direct
instruction change as [E(newhop)-E(oldhop)]/8 there, and computed change as the
complete donor-minus-recipient postL2 change minus this direct term.

Use fresh seeds33909/33910 with the existing random-layout generator, first8
worlds/population, all24 entities and all12 ordered distinct hop0..3 transitions:
4608 transitions on16 independent worlds. IID24-cycle/OODthree8-cycle graphs.
No new native outcomes opened before this registration. Native capability is
reported at every hop; errors remain in full-distribution and effect panels.

Arms: native recipient, direct instruction swap, computed-only diagnostic,
joint complete state swap, identity, and literal instruction removal (-E(old)/8
at finalquery). Recompute final RMS, all Q/K/V and residual normally. This is a
physical state intervention, not a frozen-gain path cut. Native full donor token
execution is the predictive target. All earlier output positions must remain
native because the changed token/state is final. The joint state swap must equal
every donor output. The computed-only arm is diagnostic, not an alternate
candidate to adopt if the direct hypothesis fails.

A mechanical: model-free controls, source-prefix identity, finite outputs;
export/native physical-state oracle for everyarm andjoint/donor token execution
max<=1e-9/relativeRMS<=1e-10, denominatorfloor1e-6. Identity and restored state.
Record overlapping direct/computed interaction without assuming additivity.

B full distribution: direct swap versus donor mean teacherKL<=1e-3 nats/token,
token99th percentile<=1e-2 andquerymeanKL<=1e-3 for everypopulation/orderedhoppair.
Report maximum KL as well. C causal prediction: direct-minus-recipient centered
querylogit vector versus donor-minus-recipient relativeRMS<=.01 in every such
group with denominatorfloor1e-6. These are quantitative full-vocabulary bars,
not just agreement on the correct answer.

D instruction erasure/reuse: for a fixedworld/queryentity, removing eachhop's
literal E(hop)/8 should make the resulting queryoutputs independent ofhop.
For eachpopulation/orderedhoppair, the centered difference between erasedoutputs
must have RMS<=.01 times the corresponding nativehop-difference RMS, using
denominatorfloor1e-6. This is the fixed removal prediction; do not weaken it to
average task accuracy. Report native and swapped task accuracy separately.

If B/C/D hold, one canonical-hop prefix plus an explicit literal instruction
adapter becomes a candidate reusable executor needing independent fresh program
validation and honest complete pricing. This run alone does not prove reduced
structural description. All387968 nativeconstants/background and contexts remain
charged; swapping direct E requires onlyrecipientprefix and known token embeddings,
while the joint/control oracle uses a donorcontext explicitly. No scale, field
expansion, head/rank selection or computed-only rescue afterfailure.

Managed GPU only,FP64,B8,1800s,256MiB per newtensor. Reuse exact_source_edit_reference
and shared metrics; no new compiler or large training run.
