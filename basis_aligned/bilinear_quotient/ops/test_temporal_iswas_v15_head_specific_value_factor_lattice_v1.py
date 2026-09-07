import run_temporal_iswas_v15_head_specific_value_factor_lattice_v1 as lattice


def test_six_bits_define_complete_unique_lattice():
    assert len(lattice.BITS) == len(set(lattice.BITS)) == 6
    assert lattice.VALUE_BITS == ((8, 1), (9, 1), (9, 4), (11, 3))
    assert len({lattice.arm_name(mask) for mask in range(64)}) == 64
    assert lattice.arm_name(0) == "none"
    assert lattice.arm_name(63).split("+") == list(lattice.BITS)


def test_exact_price_accounts_for_every_forward():
    assert lattice.EXACT_FORWARDS == 8 + 1 + 64
