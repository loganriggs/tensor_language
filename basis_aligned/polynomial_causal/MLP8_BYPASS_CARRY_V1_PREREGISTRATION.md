# Direct residual carry versus MLP8 bypass response

Same32worlds and five-factor grids. Replay source/attention parent F00(native),
F11(fullmixedMLP8 removal) and F10(sourceedited withwholeA9nativeheld). Capturefull
vocabulary logits, native final raw residual and actual MLP8 output. Source edit
uses complete Qoh MLP8write atallpositions, notonlysemanticreadout.

Predict directcarry finalstate=rnative-gamma*Qoh MLP8write_at_readout,
gamma=product of actual block9..17 residuallambda0. FinalRMS, fullunembedding and
30tanh softcap are native. This is the exact real-arithmetic frozen-write transport
prediction, not an assumption that later writes are actually frozen by source edit.
Do not claim a native full-write-clamp identity from this calculation alone.

A instrument: native/source/bypass three-reader replayparent <=1e-3maxabs AND1e-5rel;
full-vocabulary native final-state decoder replay samebars. Sourcepatchincomingbitwise,
A9clampincomingmatchescapturedsourceeditedwritebitwise; firstVunchanged; finite/cleanup.
Counts192nativeforwards3072seq (6perworld),128decoderbatches2048states (4perworld),0fits,
600s cap. Existing affinecarry/nonlinearwritefalsifier andsource/attentionhookcontrols.
All545902902nativeparameters andsourcecounterfactuals retained, zero saving.

B task-effect fidelity: native-minus-carry prediction matches native-minus-bypass
within .10relative, BOTH Qoh correct-margin and centered Qoh three-answer vectors,
EVERYworld. Bypass effect norm>=.10native mixed norm in both; zero denominators fail.
C vocabulary fidelity: same .10relative for centered full-vocabulary Qoh effect,
EVERYworld. No answer-only success is a whole-output program. Report errors and
signed projections without fittedgains. If B/Cfail, retain later-computation response;
no reconstruction/rank/precision ladder. If Bpasses/Cfails, label task-local only.
