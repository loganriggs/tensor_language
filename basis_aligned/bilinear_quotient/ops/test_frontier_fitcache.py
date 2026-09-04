"""Guard for ops/frontier_fitcache.py -- the verifier must SEE a difference, not merely report a count."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import frontier_fitcache as FC


def _stack():
    """The real shape: S maps a key to a TUPLE containing a dict of tensors."""
    return {
        "a10L": ("attnd", 10, torch.arange(6.0).reshape(2, 3),
                 {2: torch.ones(2, 2), 7: torch.zeros(2, 2)}, torch.eye(2)),
        "m0E": ("tableres", 0, torch.full((3, 3), 2.0), torch.ones(3, 1), torch.eye(3)),
    }


def test_identical_stacks_verify():
    ok, n, dev, where = FC.verify_stack(_stack(), _stack())
    assert ok and dev == 0.0 and where is None, (ok, dev, where)
    assert n == 7, n          # 4 in a10L (one + two in the LW dict + P), 3 in m0E


def test_a_difference_THREE_LEVELS_DOWN_is_caught():
    """The fastload failure mode: a verifier that cannot reach the changed tensor."""
    a, b = _stack(), _stack()
    b["a10L"][3][7][1, 1] = 99.0          # inside a dict, inside a tuple, inside the stack
    ok, n, dev, where = FC.verify_stack(a, b)
    assert not ok, "verifier missed a planted difference three levels down"
    assert dev == 99.0, dev
    assert "a10L" in where and "[3]" in where, where


def test_structural_difference_is_a_mismatch_not_a_skip():
    a, b = _stack(), _stack()
    del b["a10L"][3][7]                    # one fewer tensor
    ok, n, dev, where = FC.verify_stack(a, b)
    assert not ok and where.startswith("tensor count"), (ok, where)


def test_roundtrip_through_disk_is_exact():
    key = FC.stack_key("test", "roundtrip", 1)
    s = _stack()
    FC.save_stack(key, s, ["a0", "m0E"], ["a0", "m0E", "a10L"])
    got = FC.load_stack(key)
    assert got is not None
    S2, cfg, o2 = got
    ok, n, dev, _ = FC.verify_stack(s, S2)
    assert ok and dev == 0.0, (ok, dev)
    assert cfg == ["a0", "m0E"] and o2 == ["a0", "m0E", "a10L"]
    FC.cache_path(key).unlink()


def test_key_is_stable_across_processes():
    assert FC.stack_key("a", 1) == FC.stack_key("a", 1)
    assert FC.stack_key("a", 1) != FC.stack_key("a", 2)


def test_saving_matches_the_logged_figure():
    assert FC.expected_saving(15) == 1275.0     # 15 rungs x (90 - 5) s
