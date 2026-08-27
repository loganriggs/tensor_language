
## Marker hygiene (2026-08-27)

`[QUEUED]` markers on items 1-3 were stale: all three completed `exit=0`, are
written up in the ledger, and their scripts are no longer in the tree. A wake
that trusted the markers would have re-queued finished work. Rungs 4-8 remain
genuinely open, and none of them has a script yet — pulling from this file means
*writing* an experiment (with predictions registered in the docstring before it
runs), not just appending a path to a queue.
