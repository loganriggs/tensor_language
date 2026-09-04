"""Guard for ops/fastload.py.

The first version of fastload shipped a bug that this file's first version could not have caught: it
constructed the model on the `meta` device, and `jacclust.tt_model.Rotary.inv_freq` is a PLAIN ATTRIBUTE
(not a registered buffer), so it stayed meta, was invisible to `state_dict()` and `named_buffers()`, and the
first forward died. The check compared state_dicts and passed. **So the load-bearing assertion here is the
FORWARD one**: two models that produce identical logits on real tokens are identical in the way that matters.

Loads the 546M checkpoint several times, so it is slow (~10 s); run it deliberately, not in a fast suite:
    /venv/main/bin/python -m pytest -q ops/test_fastload.py
"""
import sys

sys.path.insert(0, "/workspace/tensor_language/basis_aligned/bilinear_quotient/ops")


def test_fast_loader_is_identical_including_a_real_forward():
    import fastload
    ok, n, dev = fastload.verify_identical(verbose=True)
    assert ok, f"fastload diverged from load_model(); max forward deviation {dev}"
    assert dev == 0.0, f"forward outputs differ by {dev}"
    assert n >= 230, f"only {n} tensors compared; the sweep should reach 236 (218 state_dict + buffers + attrs)"


def test_the_sweep_actually_covers_plain_attributes():
    """The specific blind spot that produced the original bug: a tensor set as a plain attribute in
    __init__, invisible to state_dict() and named_buffers(). If _all_tensors ever stops covering those,
    this module's verification is worthless again."""
    import fastload
    _cfg, _blob, R = fastload._paths()
    m = R.load_model()
    keys = fastload._all_tensors(m)
    attrs = [k for k in keys if k.startswith("attr:")]
    assert attrs, "the plain-attribute sweep found nothing; it is not doing its job"
    assert any("inv_freq" in k for k in attrs), f"inv_freq not covered by the sweep: {attrs[:5]}"


def test_fast_loader_is_actually_faster_end_to_end():
    """Not a correctness property, but the module's whole reason to exist. Measured end to end (load +
    CUDA transfer + first forward), because mmap defers page faults and a CPU-only timing would flatter it."""
    import time
    import torch
    import fastload
    import mlp_in_situ_usage_rank_map_probe as R
    if not torch.cuda.is_available():
        return
    tok = (torch.arange(1, 65).unsqueeze(0) % 50000).cuda()

    def endtoend(loader):
        t = time.time()
        m = loader().to("cuda").eval()
        with torch.no_grad():
            m(tok, tok)
        torch.cuda.synchronize()
        dt = time.time() - t
        del m
        torch.cuda.empty_cache()
        return dt

    slow, fast = endtoend(R.load_model), endtoend(fastload.load_model_fast)
    assert fast < slow, f"fastload is not faster end to end: {fast:.2f}s vs {slow:.2f}s"
