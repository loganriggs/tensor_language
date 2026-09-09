"""Token-derived context and exact sparse source/query updates for one suffix.

All prefix and reader coefficients remain explicit. Only the generic final O/head
fold removes constants; the source-edit algorithm is intervention infrastructure.
"""
import copy
import torch
from torch import nn


@torch.no_grad()
def make_edits(model,tokens,masks,heads):
    from join_contribution_context_reference import contributions
    from join_value_producer_reference import producer_writes,KINDS
    writes=contributions(model,tokens,masks,heads)
    parts,_=producer_writes(model,tokens,masks,heads)
    empty=torch.zeros_like(writes[0]);out={}
    for mask in range(8):
        out['disjoint_'+str(mask)]=-sum((writes[j] for j in range(3) if mask&(1<<j)),empty)
        out['overlap_'+str(mask)]=-sum((parts[1][k] for j,k in enumerate(KINDS) if mask&(1<<j)),empty)
    out['half_joint']=-.5*sum(writes.values())
    return out


def export_package(program,config):
    state={k:v.detach().cpu().clone() for k,v in program.state_dict().items()}
    fixed={k:v.detach().cpu().clone() for k,v in program.named_buffers() if k not in state}
    return {'program_type':'exact_source_edit_v1','config':config,'dtype':'float64',
            'n_ctx':program.background.layers[-1].mask.shape[0],'state_dict':state,'fixed_buffers':fixed}


def load_package(package):
    from deep_model import DeepModel
    assert package['program_type']=='exact_source_edit_v1' and package['dtype']=='float64'
    cfg=package['config']
    model=DeepModel(cfg['vocab'],cfg['d_model'],cfg['n_head'],cfg['spec'],package['n_ctx'],norm=cfg['norm']).double().eval()
    program=SourceEditProgram(model)
    program.load_state_dict(package['state_dict'],strict=True)
    buffers=dict(program.named_buffers())
    expected={k for k in buffers if k not in package['state_dict']}
    assert expected==set(package['fixed_buffers'])
    for k,v in package['fixed_buffers'].items():
        assert buffers[k].shape==v.shape
        buffers[k].copy_(v)
    return program.eval()


class SourceEditProgram(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.background=copy.deepcopy(model)
        layer=self.background.layers[-1]
        assert layer.attention=='bilinear' and layer.residual=='lerp' and layer.scale==.5
        assert isinstance(self.background.head,nn.Linear) and self.background.head.bias is None and layer.o.bias is None
        self.register_buffer('folded',(model.head.weight@model.layers[-1].o.weight).detach().clone())
        del layer.o

    def features(self,x,positions):
        layer=self.background.layers[-1];h=layer.norm(x);out={}
        for name in ('q1','q2','k1','k2'):
            z=getattr(layer,name)(h).reshape(len(x),x.shape[1],layer.n_head,layer.d_head)
            a,b=z.chunk(2,dim=-1);rotated=torch.cat((-b,a),-1)
            out[name]=z*layer.rotary.cos_cached[:,positions]+rotated*layer.rotary.sin_cached[:,positions]
        out['v']=layer.v(h).reshape(len(x),x.shape[1],layer.n_head,layer.d_head)
        return out

    def aggregate(self,q,k,qpos,spos):
        layer=self.background.layers[-1]
        a=torch.einsum('bthd,bshd->bhts',q['q1'],k['k1'])
        b=torch.einsum('bthd,bshd->bhts',q['q2'],k['k2'])
        p=a*b/(layer.d_head**2)*layer.mask[qpos[:,None],spos[None,:]]
        return torch.einsum('bhts,bshd->bthd',p,k['v']).flatten(-2)

    @torch.no_grad()
    def prepare(self,tokens):
        x=self.background.embed(tokens)
        for layer in self.background.layers[:-1]:x=layer(x)
        positions=torch.arange(tokens.shape[1],device=tokens.device)
        f=self.features(x,positions);z=self.aggregate(f,f,positions,positions)
        logits=.5*self.background.head(x)+.5*(z@self.folded.T)
        return {'x':x,'features':f,'z':z,'logits':logits,'positions':positions}

    @torch.no_grad()
    def edit(self,context,delta):
        x=context['x'];assert delta.shape==x.shape
        selected=torch.where(delta.ne(0).any((0,2)))[0]
        if not len(selected):return context['logits'].clone()
        positions=context['positions'];base=context['features']
        changed=self.features(x[:,selected]+delta[:,selected],selected)
        old={k:v[:,selected] for k,v in base.items()}
        # Source messages for all original queries; changed queries are replaced below.
        z=context['z']+self.aggregate(base,changed,positions,selected)-self.aggregate(base,old,positions,selected)
        installed={k:v.clone() for k,v in base.items()}
        for k in installed:installed[k][:,selected]=changed[k]
        z[:,selected]=self.aggregate(changed,installed,selected,positions)
        return context['logits']+.5*self.background.head(delta)+.5*((z-context['z'])@self.folded.T)

    def independent_constant_count(self):
        return sum(p.numel() for p in self.parameters())+self.folded.numel()


def controls():
    from deep_model import DeepModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21908)
        model=DeepModel(8,16,4,['attn']*4,16,norm='rms').double().eval()
        tokens=torch.randint(8,(2,12))
        delta=torch.zeros(2,12,16,dtype=torch.float64)
        delta[:,[1,5,9]]=.3*torch.randn(2,3,16,dtype=torch.float64)
    program=SourceEditProgram(model);checks={};errors={}
    with torch.inference_mode():
        context=program.prepare(tokens);native=model(tokens)
        checks['token_prepare']=bool(torch.allclose(native,context['logits'],atol=1e-9,rtol=1e-9))
        for name,edit in (('empty',delta*0),('disjoint',delta),('overlapping_sum',delta+delta.roll(1,-1)*.7)):
            actual=program.edit(context,edit);expected=model.head(model.layers[-1](context['x']+edit))
            maximum=float((actual-expected).abs().max())
            relative=float((actual-expected).square().mean().sqrt())/max(float(expected.square().mean().sqrt()),1e-6)
            checks[name]=maximum<=1e-9 and relative<=1e-10
            errors[name]={'max_abs':maximum,'relative_rms':relative}
        layer=model.layers[-1];positions=torch.tensor([1,5,9]);features=program.features(context['x'][:,positions],positions)
        for k in ('q1','q2','k1','k2'):
            full=layer.rotary(getattr(layer,k)(layer.norm(context['x'])).reshape(2,12,4,4))
            # Projection row counts differ, so use the registered FP64 tolerance,
            # plus a live wrong-position control rather than demand bitwise GEMM.
            wanted=full[:,positions]
            checks['absolute_rope_'+k]=bool(torch.allclose(features[k],wanted,atol=1e-9,rtol=1e-10))
            wrong=layer.rotary(getattr(layer,k)(layer.norm(context['x'][:,positions])).reshape(2,3,4,4))
            checks['wrong_position_rejected_'+k]=float((wrong-wanted).abs().max())>1e-6
        checks['context_not_mutated']=bool(torch.equal(program.edit(context,delta*0),context['logits']))
        checks['final_O_absent']=not hasattr(program.background.layers[-1],'o')
        checks['constant_count']=program.independent_constant_count()==sum(p.numel() for p in model.parameters())-16*16+8*16
        loaded=load_package(export_package(program,{'vocab':8,'d_model':16,'n_head':4,'spec':['attn']*4,'norm':'rms'}))
        checks['package_roundtrip']=bool(torch.allclose(loaded.edit(loaded.prepare(tokens),delta),program.edit(context,delta),atol=1e-9,rtol=1e-10))
    return {'passed':all(checks.values()),'checks':checks,'errors':errors}
