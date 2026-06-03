#!/usr/bin/env python3
"""One-time: stream DSIR-Pile, GPT2-tokenize (mod VOCAB), cache a flat int16 token
tensor for fast random-window sampling (no overfitting -> 'more steps' is meaningful)."""
import sys, time, torch
from train_sweep import DSIRPileStreaming, VOCAB_SIZE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000_000
OUT = "data/pile_tokens.pt"
buf, total, t0 = [], 0, time.time()
for ex in DSIRPileStreaming(n_ctx=1024):
    buf.append(ex["input_ids"].to(torch.int16))
    total += ex["input_ids"].numel()
    if total % (10 * 1024) < 1024:
        pass
    if len(buf) % 2000 == 0:
        print(f"  {total/1e6:.1f}M tokens  ({time.time()-t0:.0f}s)", flush=True)
    if total >= N:
        break
data = torch.cat(buf)[:N].contiguous()
torch.save(data, OUT)
print(f"saved {OUT}  shape={tuple(data.shape)} dtype={data.dtype} vocab={VOCAB_SIZE} ({time.time()-t0:.0f}s)")
