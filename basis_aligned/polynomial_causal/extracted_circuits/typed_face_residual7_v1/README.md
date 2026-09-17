# Two-state regional local executor

Load local_program.pt,context_program.pt,reentry_program.pt into keys local,
context,reentry. execute.execute accepts residual7,donor_city7,token_ids,
recipient_token,donor_token,city,destination,strength=.5 and returns block9 delta.
Owns exact token initial states, block8 reentry, fullattention8 and MLP8 response.
Two supplied native vector arrays remain; no supplied rho8/postattn8. Later
suffix external.74sequence tokens supported; unknown tokens rejected.
24,060,931floats,38,016native state scalars atT32. Native and isolated gates pass.
No full-model, composition, or matched-effect simplicity certification.
