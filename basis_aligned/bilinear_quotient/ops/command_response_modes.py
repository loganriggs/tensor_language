"""Dtype-preserving balanced two-command response modes, keyed by explicit bits."""
# BQGATE: LIBRARY
CELLS=('00','01','10','11')


def modes(cells):
    if set(cells)!=set(CELLS):raise ValueError('four binary cells required')
    shape=cells['00'].shape
    if any(x.shape!=shape for x in cells.values()):raise ValueError('cell shapes differ')
    return {uv:sum((-1)**(int(uv[0])*int(ab[0])+int(uv[1])*int(ab[1]))*cells[ab] for ab in CELLS)/4 for uv in CELLS}


def reconstruct(coefficients):
    return {ab:sum((-1)**(int(uv[0])*int(ab[0])+int(uv[1])*int(ab[1]))*coefficients[uv] for uv in CELLS) for ab in CELLS}


def controls():
    import torch
    constants={ab:torch.tensor([1.,-2.,3.],dtype=torch.float64) for ab in CELLS}
    fixture={ab:torch.tensor([1+2*int(ab[0]),3*int(ab[1]),5*int(ab[0])*int(ab[1])],dtype=torch.float64) for ab in CELLS}
    constant=modes(constants);coeff=modes(fixture);replay=reconstruct(coeff)
    flip=modes({ab:fixture[str(1-int(ab[0]))+ab[1]] for ab in CELLS})
    b=torch.tensor([.3,-.7,1.1],dtype=torch.float64)
    lhs=sum((x-b).square().sum() for x in fixture.values())
    rhs=sum((x-coeff['00']).square().sum() for x in fixture.values())+4*(b-coeff['00']).square().sum()
    checks={'constant_only':all(torch.equal(constant[ab],torch.zeros_like(b)) for ab in CELLS if ab!='00'),
            'live_modes':all(float(coeff[ab].norm())>0 for ab in CELLS),
            'reconstruction':all(torch.equal(replay[ab],fixture[ab]) for ab in CELLS),
            'parseval':torch.equal(sum(x.square().sum() for x in fixture.values()),4*sum(x.square().sum() for x in coeff.values())),
            'bit_flip_sign':all(torch.equal(flip[uv],(-1)**int(uv[0])*coeff[uv]) for uv in CELLS),
            'best_constant_identity':torch.allclose(lhs,rhs,atol=1e-12,rtol=1e-12)}
    checks={k:bool(v) for k,v in checks.items()};return {'passed':all(checks.values()),'checks':checks}
