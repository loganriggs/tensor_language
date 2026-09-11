# Frozen branch suppression across four corpus domains

Test the original spectral parent1 branches, unchanged from the completed FineWeb screens. These are weight-discovered factors; no Pile-adapted factor or pending local graph fit is substituted. The original support-direction prediction remains failed. The separate FineWeb suppression/specificity screen passed, while input-gating specificity missed.

Freeze32documents each from Wikipedia(en), StackExchange, biomedical(PubMed Abstracts/Central), and legal/patent(FreeLaw/USPTO Backgrounds). Use the pinned local Pile10k Arrow source; exclude all2048exact document hashes and tokenprefixes in the earlier million-token panel, plus previous FineWeb endpointwindows and selected duplicates. Selection seed5401. The fixed32-per-domain coverage held after8090source rows visited. The selected split is Wikipedia32,StackExchange32,PubMed Abstracts24,PubMed Central8,USPTO24,FreeLaw8. Prefix lengths range190–513tokens; all actual target windows have128inputs and never touch padding.

Use the first family0 target after128tokens, nearest family1 target, and nearest target outside both families, with earlier-position ties. Families are the original frozen weight-token lists. No model outputs guided selection.384endpoints/128documents are frozen in the ROWS artifact. Stored prefix padding is only for source replay; targets lie below their recorded original length.

- A: exact source/row/domain counts, finite effects,50bodyforwards400sequences128inputs, native/branch0 physical logit replay relative error<=1e-5 and MLP input error<=1e-6.
- B conditional on A: native actual-target top20>=.5 in both target families, own removal mean CE<=-.02 in both, and each branch's pooled mean absolute control CE change<=.05.
- C conditional on A/B: both own-minus-other effects<=-.01 and both paired document-bootstrap95%upper bounds<0;2000resamples,seed5402.
- D conditional on A: in EACH domain, both own-family mean CE effects are negative and both branch control mean absolute changes<=.05. Per-domain native capability and complete effect matrices are reported. D is a mean-sign/preservation check, not a per-domain confidence guarantee.

Prospective follow-up on saved endpoint states: apply the existing exact amplitude/input-gating test with unchanged writer-sign orientation, standardized paired own-family suppression bar>=.25 and both paired-bootstrap95%lower bounds>0, using its existing seed4901. This controls the possibility that token-facing effects alone explain specificity. It requires no new model forward or factor fit.

Reuse the completed suppression runner in a separate immutable variant, adding domain reporting and source-length validation. Retain native background, bias, final RMS and tanh. The bank has5760coefficients; no independent producer or full circuit extraction is claimed. Queue after existing matched local fits, with peer order preserved.

This is a corpus-shift test relative to the FineWeb training corpus. Source labels and exact exclusions do not verify that documents are absent from training, historically untouched, or free of near-duplicates. Generalization across these domains would support a narrower signed-effect prediction; it would not complete the OOD, extraction, selective manipulation and composition program.
