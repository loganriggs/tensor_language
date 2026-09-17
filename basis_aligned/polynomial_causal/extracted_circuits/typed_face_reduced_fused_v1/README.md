# Fused reduced-cross executor

Load local_program.pt,context_program.pt,reentry_program.pt into keys local,
context,reentry. execute.execute takes residual7,donor_city7,token_ids,
recipient_token,donor_token,city,destination,strength=.5; returns block9 delta.
Two native vector inputs,114supported sequence tokens; later native model external.
24,153,091floating scalars;38,016native-state scalars atT32.

Exactly the reduced counterfactual: other-head attention/write cross terms omitted,
allnormalizers/otherbackground terms retained. Shared Down is applied once,
attention8 andhead8delta once each. Native and isolated replay gates pass.
CPU local median0.104to0.056seconds in boundedinterleavedcheck; no GPU/end-to-end
speedup claim. Composition remains failed; no full-circuit certification.
