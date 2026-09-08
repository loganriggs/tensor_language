import torch

import entry12_gram_router_contract as router


def states():
    off=torch.zeros(3,2,4)
    a=torch.arange(24,dtype=torch.float32).reshape(3,2,4)+1
    b=torch.flip(a,(-1,))*.5
    return off,{"A1":a,"A2":b}


def test_gram_features_are_simultaneous_sign_invariant():
    off,experts=states(); features=router.gram_features(torch,off,experts,(0,1,1))
    reversed_experts={name:-value for name,value in experts.items()}
    reversed_features=router.gram_features(torch,off,reversed_experts,(0,1,1))
    assert features.shape==(3,6)
    assert torch.allclose(features,reversed_features)


def test_six_feature_centroid_fit_predicts_training_example():
    features=torch.tensor([[3.,0.,0.,3.,0.,0.],[2.,0.,0.,2.,0.,0.],
                           [0.,3.,0.,0.,3.,0.],[0.,2.,0.,0.,2.,0.],
                           [0.,0.,3.,0.,0.,3.],[0.,0.,2.,0.,0.,2.]])
    labels=torch.tensor([0,0,1,1,2,2])
    fitted=router.fit(torch,features,labels)
    prediction,scores=router.predict(torch,features,fitted)
    assert scores.shape==(6,3)
    assert torch.equal(prediction,labels)
