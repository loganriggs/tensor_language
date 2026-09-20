# Registered root-product comparison

Exact saved-program root covariance is positive definite, condition362.3,
with independent small-quadrature error1.1e-15. FP32 archive covariance drift
6.3e-8; output frame Gram is explicitly retained. Output rank4 can retain at
most99.778% centered root energy, but product structure can prevent attainment.

Fit root product widths4/6/8, Adam/Muon,.005/.03,seeds0/1,500steps:24CPUarms.
Fixed six-product bank, root readers on4bankcoordinates, analytic writer,
unit centered functional feature norm and penalty.001. Selection by penalized
exact root-function loss, never empirical errors. Correct final constant to
preserve original teacher Gaussian mean after root replacement.

Primary width4 predictions: centered root energy retained>=99%; full quartic
64token diagnostic error<=.19152331 (shared6baseline.18152331 plus.01); export
has24280coefficients10products and scalar-DAG replay<1e-10. Width6/8 are
capacity controls with prices24312/24344coefficients and12/14products. Original
shared6 is24296coefficients16products. Linear combinations cost coefficients
and additions, so fewer products do not automatically dominate storage.

root_product_fit.py already implements the exact metric objective and
analytic writer; gradient finite difference7.8e-11 and coefficient replay1.8e-16.
Execution/export runner remains next. No automatic further rate/seed sweep.
