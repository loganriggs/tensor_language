import pytest

import unordered_token_pair_router_contract as router


def test_signature_is_direction_invariant_and_ordered():
    forward=router.signature((1,2,3,4),(5,2,7,4),3)
    reverse=router.signature((5,2,7,4),(1,2,3,4),3)
    assert forward==reverse==((1,5),(3,7))


def test_fit_routes_known_targets_and_defaults_unknown_off():
    a1=((1,5),); a2=((2,6),); p=((1,2),)
    fitted=router.fit((a1,a2,p), (0,1,2))
    assert router.predict(fitted,(a1,a2,p,((9,10),),()))==(0,1,2,2,2)


def test_fit_rejects_target_control_collision():
    with pytest.raises(router.TokenRouterError): router.fit((((1,2),),((1,2),)),(0,2))
