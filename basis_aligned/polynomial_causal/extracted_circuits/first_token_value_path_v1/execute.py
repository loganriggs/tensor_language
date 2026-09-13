"""Conditional first-value input path. No activation lookup table or learned prompt fit."""
import torch
import torch.nn.functional as F

def token_readings(token_ids, embedding_weight, program):
    """[T] token ids -> [T,4] shared first-layer value readings.

    Embedding matrix is an explicit external weight dependency. The native
    block0 initial-state normalization/reentry is retained; float32 arithmetic
    precedes the double-precision folded map.
    """
    x0 = F.rms_norm(embedding_weight[token_ids].float(), (1152,))
    lam = program['block0_lambdas'].to(x0)
    x = F.rms_norm(lam[0] * x0 + lam[1] * x0, (1152,))
    return x.double() @ program['first_value_map'].to(x.device).T


def donor_field(recipient_ids, donor_ids, embedding_weight, program,
                gamma8, q7_readings, rms8_squared, gamma9, rms9,
                source_mask):
    """Return [T] scalar write at the declared head9 writer.

    All gamma/norm/Q inputs are recipient native contextual ports. gamma8
    and gamma9 are the specified joint-QK routing matrices; gamma9 is the
    previously selected even component, not an arbitrary full head pattern.
    source_mask [T] acts at head9 sources j, after attention8 source transport.
    Inputs must be aligned equal-length donor/recipient sequences.
    """
    assert recipient_ids.shape == donor_ids.shape
    df = token_readings(donor_ids, embedding_weight, program) - token_readings(recipient_ids, embedding_weight, program)
    dh = gamma8.double() @ df
    eig = program['eigenvalues'].to(dh.device)
    value = 2 * (q7_readings.double() * eig * dh).sum(-1) / rms8_squared.double()
    value = value * source_mask.to(value)
    return gamma9.double() @ (program['head9_gain'].to(value) * value / rms9.double())
