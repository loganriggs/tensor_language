pred_a priorwrite/pre/baseline relative<=1e-5 and partition<=1e-6.
pred_b city-only removal target AND distractor contrasts error<=.1 each4cells.
pred_c city-specific removal >=.8 own and <=.2 other fullcomponent contrast norm.
Null: contextual/query dependence distributes cue influence beyond city source positions.
12bodybatches96rows21-24tokens6suffixarms180sec,~25MB; frozen mixedpackage.

Both96paired rows fixed. Source city positions determined by one-token target and distractor factorial changes. Independent city removal is a read-site intervention; it does not remove all descendants of that city. Compare signed contrast vectors, not just aggregate amplitude. Preserve failures.
