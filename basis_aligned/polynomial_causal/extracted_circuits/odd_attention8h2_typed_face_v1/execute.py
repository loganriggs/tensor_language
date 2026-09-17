"""Closed routing/inherited face at a declared attention-factor boundary.

No model, row lookup, fitted coefficients, or native suffix is included.
"""
import torch

MASKS = (1, 4, 5)
REQUIRED_CORNERS = (0, 1, 4, 5)


def write_terms(routing_recipient, routing_donor, current_value,
                inherited_recipient, inherited_donor, mixture,
                output_weight, source_mask, destination_mask):
    """Return three B,T,D writes; routing B,T,S, values B,S,H, W D,H.

    The current value is held at recipient. Masks are B,S and B,T.
    Values are pre-mixture native head values. The destination mask applies
    before block9 reentry; it does not mask the later head9 output.
    """
    if routing_recipient.shape != routing_donor.shape:
        raise ValueError("Routing shapes must agree")
    if current_value.shape != inherited_recipient.shape or current_value.shape != inherited_donor.shape:
        raise ValueError("Value shapes must agree")
    b, t, s = routing_recipient.shape
    if current_value.shape[:2] != (b, s) or source_mask.shape != (b, s) or destination_mask.shape != (b, t):
        raise ValueError("Declared source/destination dimensions do not agree")
    v0 = (1 - mixture) * current_value + mixture * inherited_recipient
    dv = mixture * (inherited_donor - inherited_recipient)
    dr = routing_donor - routing_recipient
    def project(route, value):
        channels = (route * source_mask[:, None, :]) @ value
        return (channels @ output_weight.T) * destination_mask[..., None]
    return {1: project(dr, v0), 4: project(routing_recipient, dv), 5: project(dr, dv)}


def execute(*args, **kwargs):
    """Return the exact selected-face state delta; suffix execution is external."""
    terms = write_terms(*args, **kwargs)
    return terms[1] + terms[4] + terms[5]


def behavioral_face(corners):
    """Closed Mobius face telescopes after ANY identical downstream function.

    Input corner measurements must have identical backgrounds and readouts.
    This is not permission to sum independently propagated state atoms.
    """
    return corners[5] - corners[0]
