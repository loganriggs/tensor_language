
## Marker hygiene (2026-08-27)

`[QUEUED]` markers on items 1-3 were stale: all three completed `exit=0`, are
written up in the ledger, and their scripts are no longer in the tree. A wake
that trusted the markers would have re-queued finished work. Rungs 4-8 remain
genuinely open, and none of them has a script yet — pulling from this file means
*writing* an experiment (with predictions registered in the docstring before it
runs), not just appending a path to a queue.

## Rungs 4-8 are not actually described here (2026-08-29)

The note above says "Rungs 4-8 remain genuinely open", but **this file has never contained their
descriptions** — checked against every revision in `git log -- BENCHMARK_BACKLOG.md`, including the
earliest, which is byte-identical to the current text apart from this addendum. So the wake prompt's
selection rung 4, "an open BENCHMARK_BACKLOG rung (4-8)", **cannot be acted on from this file**: there is
nothing here to pull.

Recording rather than inventing rungs to fill the gap. A wake that reaches rung 4 should fall through to
rung 2 (second-class confirmation) and say so in the ledger, which is what 2026-08-29T08:5xZ did.
