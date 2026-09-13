"""Explicit projected-input interface for a conditional mixed attention write."""
from additive_head_raw_ports_v1 import from_projections, additive
from joint_attention_mixed_ports_v1 import decompose
from joint_attention_three_group_v1 import execute as three_group


def execute(projection_corners, norm_corners, change_inner_product,
            first_values, mixture, compact=False):
    """Three corners N/C/R; norm=mean(raw**2)+eps; cross=mean(deltaC*deltaR).

    Input projection arrays are the five Q1/K1/Q2/K2/V raw maps per corner.
    This interface consumes the cross scalar; it does not generate it from projections.
    Returns the final-query 128-dimensional mixed write before output projection.
    """
    native, child, remainder = projection_corners
    rn, rc, rr = norm_corners
    ra = rc+rr-rn+2*change_inner_product
    if not bool((ra > 0).all()):
        raise ValueError('Nonpositive additive residual norm; inconsistent input interface')
    ports = [from_projections(p, r, first_values, mixture)
             for p, r in zip(projection_corners, norm_corners)]
    ports.append(from_projections(additive(native, child, remainder), ra, first_values, mixture))
    if compact:
        return three_group(*ports)
    parts = decompose(*ports)
    return parts['cross']+parts['defect']
