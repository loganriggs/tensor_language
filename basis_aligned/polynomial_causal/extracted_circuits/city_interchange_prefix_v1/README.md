# Conditional city interchange from two native prefixes

Copy all files in this directory. Requires PyTorch, CPU float32 native inputs and
batch1. Load attention7.pt, readers.pt, head8.pt with torch.load(weights_only=True)
into program keyed by filename stem. Import execute, then call:

```python
execute.execute(program, recipient_residual6, recipient_tokens,
                donor_residual6, donor_tokens, city, destination)
```

Recipient prefix ends at last destination; donor prefix ends at city. Token IDs
match each prefix. Destination is a Boolean mask of the full output length. Output
is float64 attention8 delta, exactly zero outside the edited positions. Unknown
tokens and unsupported prefix shapes fail. Tables cover385tokens from the frozen
fresh panel. No native query or RMS inputs; mixed8 RMS uses the stated approximation.

For T32, city12 and destinations13..30: recipient31tokens + donor13tokens,
50,688native floats versus73,728for two full sequences. Two logical native contexts
remain. Shared weights:23,300,998FP32values /93,203,992bytes. Weight price unchanged;
no matched-effect random-component simplicity advantage or runtime speedup claimed.
Native blocks0–6 and recipient MLP8/later model remain external.

Fresh prediction and selective interchange pass; independent composition remains
failed/unestablished. This is not full token replacement or a whole-head donation.
The installed receipt contains a stale copied scope sentence: its linked scope
correction states the true inputs without changing any measurements or gates.
