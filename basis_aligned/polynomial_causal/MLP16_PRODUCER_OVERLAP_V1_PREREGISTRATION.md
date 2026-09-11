# MLP16 producer sharing between frozen QK and OV source features

No fitting or text. The fixed QK source frames come from
POSITION_SHARED_QK_SOURCE_V1_RESULT; the compact unembedding/MLP17 component
comes from JOINT_SHARED_READER_RANK16_V1_RESULT. Both cache hashes are checked.

For each attention17 head h, A_h is the transpose of its rank17 QK source frame.
The value-side source reader family is B_h=F O_h (1-lambda17) V_h, where F stacks
the compact component's shared reader and16partner readers. Use a raw orthonormal
row basis for B_h; require rank17. This concerns current values only. Layer0base
values are an explicitly separate branch, not folded into MLP16.

MLP16 quadratic producer g(x)=D[(Lx)*(Rx)] has coefficient Gram G=D K D^T,
K_ij=((li.lj)(ri.rj)+(li.rj)(ri.lj))/2. Compare reader FUNCTION spaces by
whitening A G A^T and B G B^T on supported spans (relative rank tolerance1e-10),
then taking singular values of the whitened A G B^T. Mean squared singular value
is the primary overlap score. No interpretation as statistical independence,
selectivity, full-path extraction or actual-state similarity.

Controls fixed before run:
- Raw identity metric, same families.
- All81 head pairings; primary uses nine matching head indices, cross-head mean
  reported descriptively without selecting a favorable pair.
- Sixteen common signed coordinate permutations (seeds1361..1376), applied to
  every A/B family together, preserving all raw geometric relationships but
  changing their orientation relative to G. No control selected after outputs.
- Trace-free producer metric G-trace(g)trace(g)^T/1152 as a diagnostic, not a
  replacement primary metric. Record radial coefficient-energy fraction.

Exact interface replay on32formal Gaussian pre-MLP16 residuals z and base states
x0: x=RMS(z), h16=z+g(x)+bias16, t=lambda17[0]h16+lambda17[1]x0, source=RMS(t).
The folded reader expression retains z, x0, bias16 and the exact shared source
RMS divisor. Native FP64 algebra with native epsilon; no FP32 bitwise claim.
This includes the local residual and norm interfaces but does not replace the
earlier model that produces z/x0. Full attention QK normalization/routing and
downstream model remain background.

Predictions:
- A: prior controls pass; all raw/function spans rank17; every overlap singular
  value in[0,1+1e-8]; producer metric versus four independent dense polynomial
  inner products agrees1e-8; local port replay1e-10; raw geometry preserved by
  control permutations1e-8; all finite. Native checkpoint weights only.
- B: mean matching-head folded overlap exceeds the MAXIMUM of16rotated-control
  mean overlaps, and exceeds their mean by at least0.05. A miss limits this pair
  of frozen feature families, not the existence of producer sharing elsewhere.
- C: at least6of9 matching heads have a largest function cosine>=0.95. This is
  only a candidate-shared-function screen, not a semantic promotion.

Report coefficient spectrum of G, all per-head singular values, all cross-head
means, controls, and full/traceless differences. Preserve original bars on any
miss; examine generic producer anisotropy, rank truncation and raw-space overlap
before calling it structural absence. Save G/A/B in /dev/shm for CPU audits.

Price:0model forwards,0text tokens; fixed weights and frozen features, no
optimization. Exact1152x1152producer Gram;4608x4608product Gram computed implicitly
from native factors, not a vocabulary tensor. No model compression adopted.
Managed lane1 only,1200second alarm, hash-bound sources and dryrun before enqueue.
