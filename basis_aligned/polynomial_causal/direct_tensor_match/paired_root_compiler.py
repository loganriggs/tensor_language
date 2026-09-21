"""Reuse the checked real symmetric-pair compiler across adjacent root forms.
Regular well-conditioned pencils use shared products; unsupported pairs retain
separate spectral forms. No fitting or approximate merge is performed.
"""
import torch,math
from quadratic_pair_blocks import compile_pair,products


def compile_pair_rotated(A,B):
    failures=[]
    for angle in [0.,math.pi/8,math.pi/4,3*math.pi/8]:
        c,s=math.cos(angle),math.sin(angle);rotation=A.new_tensor([[c,s],[-s,c]])
        try:
            pair=compile_pair(c*A+s*B,-s*A+c*B)
            pair['product_weights']=pair['product_weights']@rotation
            T=pair['input_transform'];i,j,kind=pair['product_indices'];u,v=T[:,i],T[:,j];left=torch.where((kind==1)[None,:],u+v,u);right=torch.where((kind==1)[None,:],u-v,v)
            raw=torch.einsum('pk,ip,jp->kij',pair['product_weights'],left,right);rebuilt=(raw+raw.transpose(-1,-2))/2
            errors=[float((a-b).norm()/a.norm().clamp_min(1e-30)) for a,b in zip([A,B],rebuilt)]
            if max(errors)>1e-10:raise ValueError('Original-basis replay failed')
            pair['diagnostics'].update(output_rotation=angle,original_matrix_replay=errors)
            return pair
        except ValueError as error:failures.append(str(error))
    raise ValueError('All fixed pencil bases failed: '+repr(failures))


def compile_program(source,allow_rotations=False):
    E=source['root_eigenvectors'].double();values=source['root_eigenvalues'].double()
    if len(values)%2:raise ValueError('An even number of forms is required')
    forms=(E*values[:,None,:])@E.transpose(-1,-2);pairs=[];diagnostics=[]
    for index in range(0,len(forms),2):
        try:
            pair=(compile_pair_rotated(forms[index],forms[index+1]) if allow_rotations else compile_pair(forms[index],forms[index+1]));pairs.append(dict(kind='pair',input_transform=pair['input_transform'],product_indices=pair['product_indices'],product_weights=pair['product_weights']));diagnostics.append(dict(first=index,status='shared',**pair['diagnostics']))
        except ValueError as error:
            pairs.append(dict(kind='spectral',vectors=E[index:index+2],values=values[index:index+2]));diagnostics.append(dict(first=index,status='fallback',reason=str(error)))
    return dict(U=source['U'],V=source['V'],writer=source['writer'],pairs=pairs),diagnostics


def evaluate(program,x):
    U,V=program['U'],program['V'];m,k,_=U.shape
    q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),m,k).sum(2);pieces=[]
    for pair in program['pairs']:
        if pair['kind']=='pair':pieces.append(products(q@pair['input_transform'],pair['product_indices'])@pair['product_weights'])
        else:pieces.append((torch.einsum('ni,gij->ngj',q,pair['vectors']).square()*pair['values']).sum(-1))
    return torch.cat(pieces,1)@program['writer'].T


def cast(program,dtype):
    if torch.is_tensor(program):return program.to(dtype=dtype) if program.is_floating_point() else program
    if isinstance(program,dict):return {k:cast(v,dtype) for k,v in program.items()}
    if isinstance(program,list):return [cast(v,dtype) for v in program]
    return program


def price(program):
    m,k,d=program['U'].shape;v,r=program['writer'].shape;coeff=program['U'].numel()+program['V'].numel()+program['writer'].numel();adds=2*m*k*(d-1)+m*(k-1)+v*(r-1);count=m*k;indices=0
    for pair in program['pairs']:
        if pair['kind']=='pair':
            coeff+=pair['input_transform'].numel()+pair['product_weights'].numel();indices+=pair['product_indices'].numel();count+=m;adds+=m*(m-1)+2*(m-1)+2*int((pair['product_indices'][2]==1).sum())
        else:coeff+=pair['vectors'].numel()+pair['values'].numel();count+=2*m;adds+=2*m*(m-1)+2*(m-1)
    return dict(products=count,stored_coefficients=coeff,stored_integer_indices=indices,additions=adds,fp32_parameter_and_index_bytes=4*coeff+8*indices)


def controls():
    from shared_root_block_compiler import evaluate as reference
    torch.manual_seed(5800);torch.set_default_dtype(torch.float64);rows=[]
    for name in ['generic','squares','shared_inputs','repeated_eigenvalue','singular_base']:
        U,V=torch.randn(4,2,6),torch.randn(4,2,6);E=torch.stack([torch.linalg.qr(torch.randn(4,4)).Q for _ in range(4)]);values=torch.randn(4,4)
        if name=='squares':V=U.clone()
        if name=='shared_inputs':U[1]=U[0]
        if name=='repeated_eigenvalue':E[1]=E[0];values[1]=2*values[0]
        if name=='singular_base':E[1]=E[0];values[1]=-values[0]
        source=dict(U=U,V=V,writer=torch.randn(3,4),root_eigenvectors=E,root_eigenvalues=values);program,diagnostics=compile_program(source);x=torch.randn(203,6);y=reference(source,x);err=float((evaluate(program,x)-y).norm()/y.norm());assert err<1e-10
        if name=='singular_base':assert diagnostics[0]['status']=='fallback'
        rows.append(dict(family=name,replay=err,shared_pairs=sum(r['status']=='shared' for r in diagnostics),price=price(program)))
    return rows
if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);r=controls();Path(__file__).with_name('PAIRED_ROOT_CONTROLS_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
