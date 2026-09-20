# Shared quartic basis search

Extend the successful planted Tucker gauge test to shared quadratic-feature DAGs. Two fixed starting functions: the recovered planted width3DAG and first-native-target optimized width4DAG. Preserve each fitted function exactly via invertible changes of its quadratic-feature bank, normalize features in the isotropicGaussian quadratic norm, then minimize sum of output-vector norms for sharedrootpairs. This encourages fewer reusable productnodes rather than sparsity in individual output entries.

Adam/Muon,lr.005/.05,seeds0/1/2,3000steps:24searches. At each learnedbasis hardretain2/3/4/6pairs forplantedtarget or2/4/6/8/10fornative; refitrootcoefficients by exactGaussianweight leastsquares. Keep initialbasis/refit baseline, allseeds, functionpreservationerror andbasisconditionnumbers. No sampledoutputs train, no per-evaluation refit. Bankvalues plus outputcount timesretainedpairs andsupportmetadata arecharged.

Scope: fixedquadratic-feature span; it cannot discover a new intermediate polynomial outside that span or all repeated-input identities. Comparefunctionerror to originalteacher andbasis-preservationerror to startingstudent separately. A successful sparse localquartic representation is not a wholemodel orsemanticcircuit claim.
