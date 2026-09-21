"""Evaluate source-only differences of exported shared bilinear programs."""
def product_source_delta(e,n,dm,context_only=False):
    left=((n@e['Pn'])@e['Tn']) if 'Pn' in e else n@e['A']
    right=((dm@e['Pm'])@e['Tm']) if 'Pm' in e else dm@e['B']
    phi=(left-e['base_left_mean'])*right if 'base_left_mean' in e and not context_only else left*right
    if 'group_writers' in e:
        groups=e['group_writers'].shape[1]
        out=phi.reshape(len(phi),groups,-1).sum(-1)@e['group_writers'].T+(phi@e['correction_left'])@e['correction_writers'].T
    elif 'output_basis' in e:
        out=(phi@e['output_core'].T)@e['output_basis'].T
    else:
        out=phi@e['reduced_writers'].T
    if 'centered_correction_left' in e:
        # For context_only, n already is n_recipient - calibration_mean_n.
        correction_left=left if context_only else left-e['left_mean']
        out=out+((correction_left*right)@e['centered_correction_left'])@e['centered_correction_writers'].T
    if 'linear_m' in e and not context_only:out=out+dm@e['linear_m']
    if 'linear_right_reader' in e and not context_only:out=out+(dm@e['linear_right_reader'])@e['linear_writer'].T
    return out
