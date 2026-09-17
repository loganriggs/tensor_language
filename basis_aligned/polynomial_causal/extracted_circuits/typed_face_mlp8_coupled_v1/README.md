# Coupled head8/MLP8 regional edit

`execute.execute(program, current8, donor_city8, post_attention8, recipient_token,
donor_token, city, destination, strength=.5)` returns the block9 input change.
Load `program.pt` with PyTorch. All tensors must be on the same device. FP32
native-state inputs; internal MLP response algebra uses FP64 and FP32 RMS epsilon.

This folder contains code and all static weights, with no model/repository imports.
Three supplied native arrays remain; no token-only or full-suffix extraction.
All native and isolated precision gates pass on40opened fixtures. About17million
floating scalars; no storage reduction. Independent causal composition fails.
See manifest for exact price and linked numerical receipts.
