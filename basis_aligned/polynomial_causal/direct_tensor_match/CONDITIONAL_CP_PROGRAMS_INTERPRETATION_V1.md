# Constructive Gaussian conditioning reduces linear cost

22 September 2026, 01:34 UTC. Explicit programs from two frozen CP512 parents. Exact covariance-weighted parent gradient Gram selects the input subspace; calibration mean/covariance defines the Gaussian law. No text labels fitted.

Write x=mu+S z, z standard Gaussian; retain eta=V^T z. Each affine factor is l_i(eta)+epsilon_i. Discarded epsilons are zero-mean Gaussian, independent of eta, with covariance c_ij. Then

$$
E\left[\prod_{i=1}^4(l_i+\epsilon_i)\mid\eta\right]
=\prod_i l_i+\sum_{i<j}c_{ij}\prod_{k\notin\{i,j\}}l_k
+c_{12}c_{34}+c_{13}c_{24}+c_{14}c_{23}.
$$

Project input once, evaluate four affine features per atom, cache six pair products, form the quartic from two cached pairs, and add covariance-weighted pairs. Fold constants into output bias. Seven variable products per atom suffice; omitting corrections is a different measured approximation.

| Input rank | Floats incl. fixed writer | Variable products | Parent Gaussian probe error, two seeds | Opened256doc value error | Old root1 response error |
|---|---:|---:|---:|---:|---:|
| Original CP512 |2,385,920|1,536|0|7.384/7.581%|9.723/11.781%|
|64|236,560|3,584|2.771/2.710%|8.715/8.830%|13.235/13.266%|
|128|441,360|3,584|1.832/1.745%|7.794/7.931%|11.368/12.180%|
|256|850,960|3,584|0.888/0.790%|7.314/7.528%|10.260/10.543%|

Rank256 coefficient multiplications fall from2,367,488 to830,464 and additions from2,365,424 to830,208, excluding common fixed writer equally. More variable products means no dominance under every simplicity metric. The smaller linear projection has not acquired a semantic interpretation merely by being smaller.

Exact derivative-energy upper bounds on parent-Gaussian error are0.925/0.833% at rank256; independent4096Gaussian probes estimate0.888/0.790%. The guarantee concerns each fitted parent under its Gaussian, NOT native weights or text effects. The finite probe estimate is a consistency check, not the proof.

Apparent native-value improvements at rank256 are exploratory, not statistically established. Root1 sensitivity errors remain11.190/11.675%, above10%. Small-output errors remain high. Conditioning a flawed parent does not recover missing native computations.

Corrections matter at rank64: plugging in mean discarded inputs yields4.291/4.075%parent error versus2.771/2.710%with corrections. At rank256:0.954/0.856%versus0.888/0.790%. Corrections are real charged computations.

Controls:15independent Gauss-Hermite integrations across5structures/3ranks agree<3.6e-15; fullrank exact. Six FP32exports replay FP64predictions on16384states with drift<4.6e-7. No GPU construction. Exact eigensubspaces use weights/covariance, not evaluation labels.

Next: rank256 native finite-removal, both seeds, openedFineWeb/stdlib panels, actual denominator/finalRMS/softcap. All-cell<=1.10parent-error retention and absolute10%fidelity separate. Rank selection followed opened-panel inspection; no untouched validation, semantic, OOD or composition claim.

Files: conditional_quartic_cp.py; check_conditional_quartic_cp.py; audit_conditional_cp_programs.py; CONDITIONAL_CP_PROGRAMS_V1.json; six CONDITIONAL_CP_SEED*_RANK*_V1.pt programs; CONDITIONAL_CP_REMOVAL_PLAN_V1.md.
