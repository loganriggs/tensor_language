# Query/key normalisation: the 2x2

Depth 2 width 128. Parameter-identical within each family whether the norm is on or off.

| family | query/key norm | seeds | held CE | induction |
|---|---|---|---|---|
| foldable (ours) | on | 3 | 4.64630 | -0.0034 |
| foldable (ours) | OFF | 0 | -- | -- |
| conventional | on | 2 | 4.60262 | +0.1356 |
| conventional | OFF | 3 | 4.43920 | +1.1235 |

```
{
  "foldable_ce_change": null,
  "foldable_induction_change": null,
  "conventional_ce_change": -0.16342166666666635,
  "conventional_induction_change": 0.9878873471860532,
  "sign_convention": "negative CE change = removing the norm HELPS"
}
```

## Predictions, scored

```
{}
```
