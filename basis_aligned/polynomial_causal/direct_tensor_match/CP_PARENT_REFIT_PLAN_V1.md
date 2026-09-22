Move retained CP directions after fixed-direction pruning

22 September 2026, 00:41 UTC. Compare a CP256 student with the frozen mixed CP512 parent from each start1001/1002. Initialize from the exact mixed-metric256-term pruning candidate. Both fitting directions and output coefficients may change; four normalized linear factors per term, no change in term count. Price768variable products and1202176floating coefficients, including the fixed16-output writer.

The target is the fitted parent, not the native quartic tensor. Match it exactly under the same shifted-Gaussian/coefficient mixture used for pruning, with each parent's fixed multiplier. This is a controlled test of frozen-direction restriction, not another loss-reweighting sweep. Parent error can now be evaluated exactly because the parent's coefficient tensor is represented by CP512.

Let G be the student Gram, X the parent/student cross Gram, and d_j=sqrt(G_jj). Preserve the pruning penalty by fitting normalized features:

$$
G'_{ij}=G_{ij}/(d_i d_j),\quad X'_{vj}=X_{vj}/d_j,
\quad C'=X'(G'+10^{-10}I)^{-1},\quad C_{vj}=C'_{vj}/d_j.
$$

Directions change G and d, so differentiate through the normalization as well as the Grams. Detach only the optimal readout for the envelope derivative. Parent energy is an exact constant and supplies the loss normalization. Fixed-scale input factors prevent artificial changes from factor scaling.

Muon, match_rms_adamw, initial rate0.1sqrt(4/1152), no weight decay,100steps with cosine decay to1%floor. Both archived starts receive the same schedule. Choose the best checkpoint by fitting objective only. No text labels, sensitivity weights, or pair responses enter fitting or checkpoint selection. Mean/covariance remain data-informed.

Pred_a_integrity: initialization prediction agrees with archived pruning<1e-4relative; exportedFP32 prediction drift<1e-4; finite gradients andfirst-backward allocatedmemory<20GB. Pred_b_parent: both Gaussian errors relative to fitted parent<=1%. Pred_c_retention: both starts' native text, root1 sensitivity and same-token response errors<=1.1times unpruned parent. Also report the unchanged absolute component10%criterion separately; retaining an imperfect parent is not completion.

Five structural controls compare loss, readout, teacher norm and gradients against explicit symmetric coefficient entries and exact degree-eight Gaussian quadrature. Max gradient discrepancy7.4e-15; readout discrepancy1.6e-14. Actual512teacher/256student/16output backward shape test passes on CPU. An initial test import/name/layout mistake was corrected before any control receipt or native execution; formulas and thresholds were unchanged.

Opposing outcomes: if moving directions restores fidelity at768products, frozen support was a material restriction and a cheaper candidate merits frozen finite-removal tests. If parent fidelity improves but native responses remain poor, local compression and native error compose unfavorably. If neither improves, do not infer an arbitrary-DAG lower bound from a100-step local search. Retain the strong unprunedCP and shared1088-product baselines.

Native intervention, OOD, composition and stable semantic identification remain outside this screen. Any promotion must test them explicitly. The exact target remains16fixed projections of the selectedMLP16→MLP17purequartic branch; other model paths and nonlinear operations have not been replaced.
