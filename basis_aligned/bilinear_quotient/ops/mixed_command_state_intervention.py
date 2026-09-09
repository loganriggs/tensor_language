"""Remove the joint two-bit residual-state contrast at one registered boundary."""
# BQGATE: LIBRARY
from contextlib import contextmanager
import command_response_modes as C


def remove_mixed(batch, cells):
    import torch
    if len(cells)!=4 or set(cells)!=set(C.CELLS) or batch.shape[0]!=4:raise ValueError('one aligned command cube required')
    m=C.modes({ab:batch[i].double() for i,ab in enumerate(cells)})['11']
    return (batch.double()-torch.stack([(-1)**(int(ab[0])+int(ab[1]))*m for ab in cells])).to(batch.dtype)


@contextmanager
def at_boundary(model, layer, cells, audit=None):
    def replace(_module,_args,out):
        state,first=out
        changed=remove_mixed(state,cells)
        if audit is not None:audit.append((state.detach().clone(),changed.detach().clone()))
        return changed,first
    handle=model.transformer.h[layer].register_forward_hook(replace)
    try:yield
    finally:handle.remove()


def controls():
    import torch
    cells=('00','10','01','11')
    base=torch.tensor([1.,-2.,3.],dtype=torch.float64)
    rows=torch.stack([base+(-1)**int(ab[0])*base*2+(-1)**int(ab[1])*base*3+(-1)**(int(ab[0])+int(ab[1]))*base*4 for ab in cells])
    before=C.modes({ab:rows[i] for i,ab in enumerate(cells)})
    changed=remove_mixed(rows,cells);after=C.modes({ab:changed[i] for i,ab in enumerate(cells)})
    checks={'mixed_zero':torch.equal(after['11'],torch.zeros_like(base)),
        'others_preserved':all(torch.equal(before[uv],after[uv]) for uv in ('00','10','01')),
        'live':float((changed-rows).norm())>1,
        'idempotent':torch.equal(changed,remove_mixed(changed,cells)),
        'cell_order':torch.equal(remove_mixed(rows.flip(0),cells[::-1]),changed.flip(0))}
    try:remove_mixed(rows,('00',)*4);checks['bad_cells_rejected']=False
    except ValueError:checks['bad_cells_rejected']=True
    import torch.nn.functional as F
    from jacclust.tt_model import GPT,GPTConfig
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60911)
        model=GPT(GPTConfig(vocab_size=8,n_layer=3,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        for block in model.transformer.h:block.attn.c_proj.weight.data.normal_(std=.1)
        model.lm_head.weight.data.normal_(std=.1)
        tokens=torch.randint(8,(4,7))
    def forward(manual=False):
        x=F.rms_norm(model.transformer.wte(tokens),(16,));x0=x;first=None
        for layer,block in enumerate(model.transformer.h):
            x,first=block(x,first,x0)
            if manual and layer==1:x=remove_mixed(x,cells)
        return model.lm_head(F.rms_norm(x,(16,))),first
    with torch.inference_mode():
        native,first=forward();oracle,_=forward(manual=True);audit=[]
        with at_boundary(model,1,cells,audit):actual,edited_first=forward()
        checks['native_hook_oracle']=torch.equal(actual,oracle) and len(audit)==1
        checks['first_value_unchanged']=torch.equal(first,edited_first)
        checks['native_intervention_live']=float((actual-native).norm())>1e-6
        try:
            with at_boundary(model,1,cells):raise RuntimeError('planted cleanup check')
        except RuntimeError:pass
        checks['restored']=torch.equal(native,forward()[0]) and not model.transformer.h[1]._forward_hooks
    return {'passed':all(checks.values()),'checks':checks}
