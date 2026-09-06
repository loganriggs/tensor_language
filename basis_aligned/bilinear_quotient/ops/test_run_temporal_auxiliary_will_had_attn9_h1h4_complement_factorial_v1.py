import run_temporal_auxiliary_will_had_attn9_h1h4_complement_factorial_v1 as runner


def test_frozen_head_partition():
    assert runner.H1H4 == (1, 4)
    assert runner.COMPLEMENT == (0, 2, 3, 5, 6, 7, 8)
    assert set(runner.H1H4).isdisjoint(runner.COMPLEMENT)
    assert set(runner.H1H4) | set(runner.COMPLEMENT) == set(range(9))


def test_authority_and_parent_coverage():
    rows, parent = runner.load_closure()
    assert len(rows) == 128
    assert parent["terminal"] == "screen"


def test_dryrun_price_shape():
    rows, _parent = runner.load_closure()
    assert len(runner.chunks(rows)) == 4
    assert 3 * len(runner.chunks(rows)) == 12
