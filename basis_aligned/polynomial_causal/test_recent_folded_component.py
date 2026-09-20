import torch
import torch.nn.functional as F
from jacclust.tt_model import GPT,GPTConfig
from recent_folded_component import extract,execute


def test_independent_recent_executor_matches_native_modules_and_cross_terms():
    torch.manual_seed(628)
    model=GPT(GPTConfig(vocab_size=17,n_layer=18,n_embd=8,n_head=2,
        expansion_factor=2,bilinear=True,bilinear_attn=True,squared_attn=True))
    with torch.no_grad():
        for parameter in model.parameters():parameter.normal_(0,.15)
    component=dict(readers=torch.randn(8,3),coefficients=torch.tensor([1.,-2.,3.]),
        residual_writer=torch.randn(8),vocabulary_writer=torch.randn(17))
    w=extract(model,component);b16,b17=model.transformer.h[16:18]
    for length in (3,7):
        h=torch.randn(2,length,8);x0=torch.randn_like(h);v1=torch.randn(2,length,2,4)
        for scale in (0.,.5,1.,1.5):
            write=b16.mlp(F.rms_norm(h,(8,)))
            live=b17.lambdas[0]*(h+scale*write)+b17.lambdas[1]*x0
            attn,_=b17.attn(F.rms_norm(live,(8,)),v1)
            h17=live+attn
            alpha=(F.rms_norm(h17,(8,))@component['readers']).square()@component['coefficients']
            actual=execute(w,h,x0,v1,scale)
            torch.testing.assert_close(actual['h17'],h17,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(actual['alpha'],alpha,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(actual['terms'].sum(-1),actual['alpha'])
