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
    return {'passed':all(checks.values()),'checks':checks}
