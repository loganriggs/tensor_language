# Single-layer shared orthogonal-square baseline

22 September2026,02:48UTC. Return to true native MLP17 quadratic tensor (not quartic parent):16fixed output readers, all1152input coordinates,4608native productchannels. T_v=sym(L17^T diag(reader_v^T D17) R17). Normalization outside this polynomial.

Restricted hypothesis: F_v(x)=sum_j c_vj (q_j^T x)^2 using ONE orthogonal inputbasis Q across alloutputs. Exact representation exists iff real symmetric T_v commute pairwise. This is much narrower than arbitrary nonorthogonal, overcomplete bilinear/square, Tucker/block-term or generalDAG representations. No claim of universal decomposition failure follows.

Eight fixedpairs(0,1),(2,3),...,(14,15), each matrix independently unitFrobenius normalized. Compute commutator norm c and s=sqrt(||A||op²+||B||op²). For any commuting approximants with joint error e, c<=2s e+e², hence e>=c/(sqrt(s²+c)+s). Divideby sqrt2 for pairrelativeerror. Floatingpoint numericalbound, not intervalcertification. Report inexpensive upperbound fromeither matrix's eigenbasis. Pairbounds cannotbe relabeled as fulltensororfunctionaltext error.

Controls: commuting, nearlycommuting, generic symmetric, sharednonorthogonalsquare representation (counterexampletooverbroadinterpretation), and blockstructuredforms. Check boundagainstconstructedcommutingapproximations; native quadratic contractions replay randominputvalues. No optimization, datafitting, codepatheffect or semanticadoption.
