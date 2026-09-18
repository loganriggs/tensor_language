# One residual6 input: complete city-removal write

Load attention7.pt, readers.pt and head8.pt with torch.load(weights_only=True)
into a dictionary keyed by their stems. From this directory, import execute and call
`execute.execute(program, residual6, token_ids, city, destination)`.

Only PyTorch and these files are required. Certified interface: CPU float32
residual6[1,T,1152], integer token_ids[1,T], city index, Boolean destination[T].
The result is a float64 attention8 removal delta. Unsupported tokens and batches
larger than one fail. The frozen token vocabulary has 396 entries.

Five MLP7 readers supply both queries, both keys and current value. Attention7
is generated with all nine heads. Mixed8 RMS is approximated from non-MLP7 sources;
MLP7 remains fully present in each numerator. No supplied queries or RMS scalars.
Native blocks0–6 and MLP8 plus later layers remain external.

Price: 23,326,342 FP32 values (93,305,368 bytes), plus 36,864 native input floats
at T32. Both weight and state counts increase over the earlier three-input export.
This is not token-only extraction or a compression win. Fresh prediction remains
untested for this version; independent source composition previously failed.
Consult manifest evidence links for local, installed and isolated test receipts.
