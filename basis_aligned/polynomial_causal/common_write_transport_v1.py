"""Role-matched transport of a two-position residual write, with causal guards."""
import torch


def transport(component,donor,recipient):
    dp=donor['common_positions'];rp=recipient['common_positions']
    assert len(dp)==len(rp)==2
    assert min(dp)>=donor['interaction_start'] and min(rp)>=recipient['interaction_start']
    assert donor['foil']==recipient['foil']
    assert len(donor['rows'])==len(recipient['rows'])==component.shape[0]
    for a,b in zip(donor['rows'],recipient['rows'],strict=True):
        assert a['factors']==b['factors']
        assert [a['ids'][i] for i in dp]==[b['ids'][i] for i in rp]
    assert component.shape[1]==donor['length']
    other=[i for i in range(donor['length']) if i not in dp]
    assert component[:,other].count_nonzero()==0
    output=component.new_zeros((component.shape[0],recipient['length'],component.shape[2]))
    output[:,rp]=component[:,dp]
    return output
