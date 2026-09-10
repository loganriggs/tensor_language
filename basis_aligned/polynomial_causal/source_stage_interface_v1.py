"""Information-stage interface; first mature noun roles may differ by layout."""
import numpy as np


def validate_pair(original, fronted):
    assert original['layout']=='original' and fronted['layout']=='fronted_pp'
    assert fronted['world_id']=='fronted_pp:'+original['world_id']
    assert original['foil']==fronted['foil']
    for world in (original,fronted):
        assert world['interaction_start']==world['length']-3
        assert max(world['object_position'],world['category_position'])==world['interaction_start']
        assert world['common_positions']==[world['length']-2,world['length']-1]
        assert len(world['rows'])==32
    for a,b in zip(original['rows'],fronted['rows']):
        assert a['ids'][-2:]==b['ids'][-2:]
    return True


def controls():
    rng=np.random.default_rng(9100905)
    source=np.zeros((32,10,5));source[:,-3:]=rng.normal(size=(32,3,5))
    mapped=np.zeros((32,11,5));mapped[:,-3:]=source[:,-3:]
    restored=np.zeros_like(source);restored[:,-3:]=mapped[:,-3:]
    assert np.array_equal(restored,source) and np.count_nonzero(mapped[:,:-3])==0
    recipient=rng.normal(size=mapped.shape);component=np.zeros_like(mapped)
    component[:,-3:]=rng.normal(size=component[:,-3:].shape)
    swapped=recipient-component+mapped
    assert np.array_equal(swapped[:,:-3],recipient[:,:-3])
    # A gate square can interact with no bilinear product at all.
    # z(a,b)=a*b from two gated scalar identity maps.
    z=np.array([[0.,0.],[0.,1.]])
    assert z[1,1]-z[1,0]-z[0,1]+z[0,0]==1
    return {'passed':True,'stage_roundtrip_bitwise':True,'recipient_prefix_preserved':True,
            'linear_gated_chain_interaction':1.,
            'scope':'Timing-aligned source ports, not equal noun roles or a semantic feature certificate. Gate interaction does not identify a native bilinear product.'}
