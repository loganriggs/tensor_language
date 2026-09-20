# Shared six-source selector versus per-input feasibility ceiling

Target: a role-level reusable source combination, replacing per-input optimized source amplitudes. Sixsourceports are unchanged; native port generators/suffix remain required. Candidate price12coefficients total (six perrole), excluding generators. This tests reuse/selective intervention, not complete independent extraction.

Compute native float64 first derivatives for number plus8controls at all96sites. Replay old4-output sixsourcegradients<=1e-8 and a fixed signed finite difference(.001) relative<=1e-3. Fit only oppositepanel gradients for eachrole, holdcongruent gradients out of sharedfit. Bothpanels are openeddata, so this is a fit-held split, not freshprospectiveOOD. No finite outcome fitting.

Let n=-g_number, r=[0,0,1,1,1,0], b=n.r. Maximize worstsigned retention gamma with sign(b)n.a >=gamma|b| perfitrow, |g_control.a|<=.08|b| for all8controls, -1<=a_i<=1,0<=gamma<=2. Explicit tiny-reference floor1e-10 is reported. Common a acrossall24fitrows perrole. Independently solve same LP perinput as diagnostic ceiling, includingheldrows; oracle has noheld-outclaim. Primalfeasibility anddual objective checks1e-7 required. Planted independent feature recoversgamma1; perfectlycoupled number/control caps gamma.08, bothcertifiedCPU.

Native validation arms unitB,shared,oracle. Primarysharedprediction: all8congruent role/familycells retain>=80%aligned number effect versusunitB and maxof8control L2norms<=10%ofown numbernorm. Oraclegatesameall16cells. A passingoraclewithfailingshared localizes reuse difficulty; bothfailing may be finitecurvature or source-interface limitations, so do not infer impossibility fromnativefailure alone. LPduality certifiesonly the specifiedfirst-order linearprogram.

Counts12prefix24double suffix72gradient reverse28native suffix. Zero/nativeunitB replay included, signed control effects retained. Thischanges theobject fromcontext-specific selectors to sharedsourcefeatures; no rank-onlycompressionclaim.
