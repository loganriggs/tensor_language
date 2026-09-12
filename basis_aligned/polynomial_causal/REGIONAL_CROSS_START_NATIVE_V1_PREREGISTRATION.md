# Independent-fit regional block: native correspondence

Use the frozen48geographic/new-spelling rows. Target is the common-quadratic block
from random-start arm1. Independent reference is native-aligned arm0, atoms11and2,
selected only by parent-reader weight alignment; retain that fit's own complete
16-function dictionary and eliminated private query writers. Do not refit readers
or use semantic outcomes to change the atom match. All causal source positions.

A: native attention replay<=1e-5; independent/target all-source block write error
<=.1relative. B:A and each of4families independent-block removal reduces regional
cue gap>=10%native, with>=4/6positive pairs. C:A/B and RMS difference of the
independent versus target signed regional removal effects<=.1relative in EACH
family. Report unrelated contrast effects and per-pair errors without filtering.
This tests stable native implementation of the weight-matched graph; no change
to earlier atom-level correspondence misses or the parent-dominance failure.

10bodyforwards48sequences6-14tokens,3batchedsuffix arms,180secondmanagedGPUlimit.
No fitting, all native gates/background/full suffix retained. Passing this screen
still does not close broader OOD, unrelated-behavior preservation or upstream
independent extraction. Literal independent source block stores6vectors versus
4with explicit sharing in the target; all private writer dependencies charged.
