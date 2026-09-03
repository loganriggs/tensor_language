# Rung 534 managed smoke receipt

**Completed:** 2026-09-03 13:16 UTC

**Managed runner exit:** 0

**Scientific outcomes opened:** no

The source-frozen smoke executed one four-document natural-text batch through one direct native forward and all
seven registered score arms in both donor backgrounds: `1 + 7 * 2 = 15` forwards. It exercised the same support
accumulation and physical score replacement path as the full run, but did not compute cross-entropy losses.

All structural checks passed:

- direct and analytical native logits were bit-exact (`max difference = 0`);
- the replayed target product was exact (`max difference = 0`);
- factor replay error was `4.3824e-14`;
- float32 `P_target = S + R` recomposition error was `7.4506e-09`, below the preregistered `2e-6` bound;
- the smallest donor removal and target replacement root-mean-square changes were `9.2833` and `3.3845`;
- no intended edit was zero;
- peak allocated GPU memory was `3,159,521,280` bytes.

Frozen hashes:

- core: `fdfb3b0ba8a7a5639cb75677e26e33e24b346f6bd7f45de20f40a70090ab5e88`;
- smoke wrapper: `8c9d043da7054894bd1a5e5cd74fa8d4370e38d37a09c722d863eefe9235d8ed`;
- smoke log: `d662a5be1a213e6219c8d379c0a1e29a4ef396f0c14fe1cf047055952ca7127d`.

The full 1,440-forward registered run is therefore authorized through the managed runner. This receipt supports
only instrument validity; it says nothing yet about whether the private correction is autonomous.
