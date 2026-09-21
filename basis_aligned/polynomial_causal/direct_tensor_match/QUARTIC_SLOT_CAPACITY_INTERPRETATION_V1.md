**The 384-product budget is too small for 10% error in the measured native contraction metric.**

Same16fixedreader purequartic target as exactfeaturepilot. Twoindependent256-tripleGaussian sketches constructexactfirst-slotmaps ofthefullysymmetrizedcoefficienttensor. Native2.67s,analytic/autogradreplay4.75e-15,FP32/64difference5.61e-7,tracechecks<2e-16. AllregisteredpredictionsPASS.

|Measurement|Best256-directionfloor|Best768-directionfloor|Necessarydirectionsfor10%|
|Sketch0|46.47%|18.82%|987|
|Sketch1|46.81%|18.93%|989|
|Pooled512triples|58.58%|23.46%|1044|

TheseareEXACT best-input-span lowerbounds forthedeclaredFINITEmultilinear-contractionmetric. TheGramisanunbiasedestimateofafullcoefficient-modeGram, butwehavenoconcentrationcertificateforits spectrum. Do notpresenttheseasexactrelativefullFrobeniuserrorfloors, ordinarytextpredictionfloors orlogitinterventionbounds.

The current32-feature/four-product architecture readsatmost256linearinputdirections, regardlessoptimizer. Itsinheritedspanmisses85.44/85.33%ofsketchenergyinrelative-normterms; exact-trainedspanmisses83.41/83.43%. Betterdirectionselectionhasroom,butcannotbeat46–47%ontheindividualsketchesor58.58%pooled atthisspanwidth.

Evena broader384-product division-free arithmeticDAG cannotreach10%inthismetric: itsquarticinputspanhasrankatmost768, withpooledfloor23.46%. Atleast522products arenecessary(1044/2), notsufficient. For thespecific32-feature hierarchy, at least17productsperquadraticfeature areneeded bytheinputspanbound; retainingthecurrentpairedrootcompilerwouldcostatleast32*17+256=800products. Thisisnotaconstructionthatguarantees10%, noragenerallowerboundof800forallDAGs.

**Why the general arithmetic-DAG bound holds**

Eliminateintermediateaddition/linear gates bysubstitution. Everyvariablemultiplicationgate canbewritten

$$
p_k(x)=\left(a_k^\top x+\alpha_{k0}+\sum_{j<k}\alpha_{kj}p_j(x)\right)
\left(b_k^\top x+\beta_{k0}+\sum_{j<k}\beta_{kj}p_j(x)\right).
$$

With $M$ distinctmultiplicationgates, set $B=[a_1,b_1,\ldots,a_M,b_M]$. Byinductionevery $p_k$ isapolynomialof $B^\top x$. Outputs mayalsohaveadirectaffine term, butthatcontributesnothingtothehomogeneousquarticpart. Consequentlythequarticpartfactors throughatmost $2M$ inputdirections. Thisallowsarbitrarysharing,skipconnections, affineconstants andcancellation; eachsharedproductischargedonce. Constantmultiplicationsareabsorbedinlinearmaps. Divisionandothernonpolynomialoperationsareoutsidethislemma.

For fixedotherinputslots $b,c,d$, define $A_{bcd}a=H(a,b,c,d)$. If a candidatequarticusesreader span $B$, everyrowofitsfirst-slotmapliesinthatspan. Thusitsresidualnormcannotbesmallerthan $\|A_{bcd}(I-P_B)\|_F$. Summingoverthesketch gives $G=\sum A_{bcd}^\top A_{bcd}$; thebest rank-$r$ space leaves tailenergy $\sum_{i>r}\lambda_i(G)$. Thisisthelowerboundusedabove. Itallowsevenmorefreedomthanarealquarticprogram,soitisnecessaryonly.

**Redteam: stable spectra do not mean stable features**

Although thetworank256floorvaluesareclose, aspaceoptimizedonsketch0misses83.24%onsketch1, versus46.81%forsketch1'sownoptimum (and83.20%inreverse). Theseareunstablefinite-sketchspaces, notidentifiedsemanticfeatures. Poolingconstraintsraisesfloorsto58.58%. Do notuseindividualsketchspectrumagreementasproofthatthepopulationGramhasconverged.

Scientificconsequence: a10%globalcoefficient-contractiontargetat384productsisnotareasonablecontinuingconstraint. Futurearchitecturesmustallowbroaderinputdependenceandbepricedagainsttheoriginalcomputation, whilemaintainingdistincttestsforcoefficientmatching, naturalbehavior andselectivemanipulation. Thisdoesnotvindicateanylargerarchitectureinadvance. Toyoptimizationfailuresremainaseparateissue. OrdinaryTucker/HTandgeneralDAGsarenotruledout, andtheconditionaltextapproximationsremainusefulonlyintheirvalidatedscope.

[Preregistered measurement](QUARTIC_SLOT_CAPACITY_PLAN_V1.md) · [Native sketches and transfers](QUARTIC_SLOT_CAPACITY_V1.json) · [Pooled gate-count consequence](QUARTIC_GATE_CAPACITY_AUDIT_V1.json) · [Exact-map controls](QUARTIC_SLOT_READER_CONTROLS_V1.json) · [Pooling code](audit_quartic_gate_capacity.py).
