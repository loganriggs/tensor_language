"""Per-coordinate FP32 rounding envelope for a four-cell FP64 state projection."""
# BQGATE: LIBRARY
import command_response_modes as C


def audit(ideal,actual,cells):
    import torch
    if actual.dtype!=torch.float32 or ideal.dtype!=torch.float64:raise ValueError('FP64 ideal and FP32 deployed state required')
    expected=ideal.float();up=torch.nextafter(expected,torch.full_like(expected,float('inf'))).double()
    down=torch.nextafter(expected,torch.full_like(expected,float('-inf'))).double()
    envelope=.5*torch.maximum(up-expected.double(),expected.double()-down)
    errors=(actual.double()-ideal).abs()
    im=C.modes({ab:ideal[i] for i,ab in enumerate(cells)});am=C.modes({ab:actual[i].double() for i,ab in enumerate(cells)})
    mode_envelope=envelope.sum(0)/4
    slack=8*torch.finfo(torch.float64).eps*sum(x.abs() for x in im.values())
    mode_errors={uv:(am[uv]-im[uv]).abs() for uv in C.CELLS}
    finite=bool(ideal.isfinite().all() and actual.isfinite().all() and envelope.isfinite().all())
    correct_cast=torch.equal(actual,expected)
    pointwise=bool((errors<=envelope).all())
    modes=all(bool((e<=mode_envelope+slack).all()) for e in mode_errors.values())
    return {'passed':finite and correct_cast and pointwise and modes,'correct_nearest_cast':correct_cast,
            'pointwise_envelope_pass':pointwise,'mode_envelope_pass':modes,
            'max_rounding_error':float(errors.max()),'max_pointwise_envelope':float(envelope.max()),
            'max_mode_error':max(float(e.max()) for e in mode_errors.values()),'max_mode_envelope':float(mode_envelope.max()),
            'max_slack':float(slack.max())}


def controls():
    import torch
    import mixed_command_state_intervention as S
    cells=('00','10','01','11')
    base=torch.tensor([1024.3,-4096.2,8192.1],dtype=torch.float32)
    rows=torch.stack([base+(-1)**int(ab[0])*.11111+(-1)**int(ab[1])*.22222+(-1)**(int(ab[0])+int(ab[1]))*.33333 for ab in cells])
    ideal=S.remove_mixed(rows.double(),cells);actual=ideal.float();good=audit(ideal,actual,cells)
    bad=actual.clone();bad[0,0]=torch.nextafter(bad[0,0],torch.tensor(float('inf')))
    negative=audit(ideal,bad,cells)
    exact=torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]],dtype=torch.float64)
    checks={'scale_fixture_rounding':good['passed'],'old_absolute_bar_live':good['max_mode_error']>1e-5,
        'one_ulp_corruption_rejected':not negative['passed'],'exact_representable':audit(exact,exact.float(),cells)['passed']}
    return {'passed':all(checks.values()),'checks':checks,'fixture':good}
