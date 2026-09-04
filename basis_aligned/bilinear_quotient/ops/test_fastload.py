"""Guard for ops/fastload.py: the fast loader must stay bit-identical to the existing one.

Loads the 546M checkpoint twice, so it is slow (~4 s) and GPU-free; run it deliberately, not in a fast suite:
    /venv/main/bin/python -m pytest -q ops/test_fastload.py
"""
import sys

sys.path.insert(0, "/workspace/tensor_language/basis_aligned/bilinear_quotient/ops")


def test_fast_loader_is_bit_identical():
    import fastload
    ok, n = fastload.verify_identical(verbose=True)
    assert ok, "fastload.load_model_fast() diverged from load_model()"
    assert n >= 200, f"only {n} tensors compared; the checkpoint has 218"


def test_fast_loader_is_actually_faster():
    """Not a correctness property, but the module's entire reason to exist -- if it stops being
    faster (a torch upgrade making mmap or assign a no-op), the right move is to delete it."""
    import time
    import fastload
    import mlp_in_situ_usage_rank_map_probe as R
    t = time.time(); R.load_model(); slow = time.time() - t
    t = time.time(); fastload.load_model_fast(); fast = time.time() - t
    assert fast < slow, f"fastload is not faster: {fast:.2f}s vs {slow:.2f}s"
