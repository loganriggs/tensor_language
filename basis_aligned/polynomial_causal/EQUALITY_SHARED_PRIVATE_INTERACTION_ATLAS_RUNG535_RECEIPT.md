# Rung 535 CPU interaction-atlas receipt

**Completed:** 2026-09-03 13:23 UTC

Using the frozen rung-534 per-document sufficient statistics, compute

$$
I_{S,R}=E_{\rm native}-E_S-E_R.
$$

This is the exact two-way causal interaction between the shared score and target remainder. No model was loaded and
no new outcome was opened. The factorial identity closes exactly.

Across code, all six token-group/background cells have the same interaction-mean sign in both document halves. The
interaction is negative and has root-mean-square size 13.3%--35.6% of the native target effect. Across natural text,
three of six cells keep the same mean sign across halves; the copy-positive sign changes with the source background.
Thus the interaction is substantial and code-stable, but the present coordinates do not yet define a
corpus-independent composition rule.

- source: `219309b3b6761504c630687615378d06d713bc9225a03ae16afb45a8b351cb6c`;
- result: `2ec547ac04fe01e407cffde6c7b20e890b191e9867f6d8af4e509d303788fc19`;
- new model forwards: `0`.
