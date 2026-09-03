# Rung 528 managed smoke v2 terminal receipt

**Completed:** 2026-09-03 11:02 UTC

**Status:** valid managed operational smoke; full discovery becomes eligible

The corrected v2 wrapper remained sealed through the enqueue helper's CPU preflight. The managed runner logged its
start at `11:02:09`; the result file was created at `11:02:17`; and the runner logged exit 0 at `11:02:17`. This
resolves the provenance failure in the first wrapper.

All registered smoke checks passed on four documents:

- direct native and analytical native logits and post-MLP12 boundaries were bit-exact;
- inserting each action's own complete boundary transition reproduced its logits exactly;
- effective boundary, embedding skip, and attention first-value states had maximum absolute error zero;
- factor reconstruction error was `4.42e-14`;
- all four score transitions were live, with minimum boundary-change RMS `8.336`;
- both continuation patches were live, with minimum write-change RMS `1.128`;
- all 22 forwards, 22 boundary captures, 16 overrides, two write captures, and 16 write patches reconciled exactly;
- peak GPU memory was `6,330,523,136` bytes; and
- no task or circuit effect was retained.

The v2 artifact SHA-256 is `436d98a2c5f66fc8fdedf1143d2cd4d145e73134bd076708085614262cd83374`.
It matches the invalid first artifact byte-for-byte because the computation is deterministic. The content hash alone
does not establish managed provenance; the sealed preflight check, file birth time, runner start, and exit ledger do.

The full runner is now implemented with the preregistered conditional stages. Its final pre-outcome hashes are:

- preregistration: `8e8bdb6af3f0ede2a86a07fa75f86bcefc58e6d8c9214169d5bc8de4f759ad77`;
- runner: `69e728bae2b67fcdc30beebbdc0e65981646d6dbfe474743e37d46e22cd89427`;
- tests: `1cbbe85b9fcdd6d4ab3dd3bb9eeed9b0e6bfbf77af970e66830cc5845b649f0e`.

Fifteen CPU tests, the static gate, full fast suite, and GPU-free dry run pass. The unconditional discovery price is
`1,984` forwards; the fully conditional ceiling is `11,485`. The runner retains only aggregate task/circuit sums, not
tokens, logits, boundaries, or hidden states. Full GPU discovery is eligible after this unit is committed and pushed.
