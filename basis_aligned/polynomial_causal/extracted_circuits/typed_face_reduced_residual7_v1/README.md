# Reduced-cross regional executor

Load local_program.pt,context_program.pt,reentry_program.pt into keys local,
context,reentry. execute.execute takes residual7,donor_city7,token_ids,
recipient_token,donor_token,city,destination,strength=.5 and returns block9 delta.
The only omission is other-head attention/background cross terms in the MLP8
response; full normalizers and otherbackground terms remain. This is distinct
from the complete-response residual7 package.

Native and isolated replay pass.24,153,091floats,114supported sequence tokens,
two supplied native arrays and external suffix. The token extension is compiled
from weights,not fitted. Fresh prediction/selectivity: HEAD2_MLP8_CROSS_FRESH_V1.
No token-only,newendpoint/corpus,donor-free or composition certification.
