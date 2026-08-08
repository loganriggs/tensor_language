# Training grid — claim a cell by editing this file and pushing BEFORE starting

Status: `unclaimed` / `local:running` / `scale:running` / `done` (+ CE)
Head dim fixed at 16, so heads = width/16. Vocab 4096 primary.
Fresh single-epoch protocol; 3 seeds per cell.

## Primary grid (depths 1-2)

| depth | width | heads | owner | status |
|---|---|---|---|---|
| 1 | 32 | 2 | local | unclaimed |
| 1 | 64 | 4 | local | unclaimed |
| 1 | 128 | 8 | local | unclaimed |
| 1 | 256 | 16 | scale | unclaimed |
| 2 | 32 | 2 | local | unclaimed |
| 2 | 64 | 4 | local | unclaimed |
| 2 | 128 | 8 | local | unclaimed |
| 2 | 256 | 16 | scale | unclaimed |

## Depth ladder (after the primary grid has first results)

| depth | width | owner | status |
|---|---|---|---|
| 3 | 64 | scale | unclaimed |
| 3 | 128 | scale | unclaimed |
| 3 | 256 | scale | unclaimed |
| 4 | 64 | scale | unclaimed |
| 4 | 128 | scale | unclaimed |
| 4 | 256 | scale | unclaimed |

## Baselines (matched-optimizer, per width — REQUIRED before quoting any cost)

| kind | widths | owner | status |
|---|---|---|---|
| bigram table (closed form, no training) | n/a | local | unclaimed |
| unigram floor | n/a | local | unclaimed |
| same-size softmax+GELU transformer | 32-256 | local | unclaimed |

The softmax/GELU baseline answers the first question a reviewer will ask:
what does the no-softmax bilinear architecture cost in prediction quality,
at each size? If the gap grows with size, the fold's tractability is being
bought with capability and that must be reported alongside every result.

## Vocab check

| vocab | owner | status |
|---|---|---|
| 8192 at width 128 depth 2 | local | unclaimed |

Answers whether conclusions are vocab-size artifacts.
