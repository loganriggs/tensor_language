"""Exact conditional contraction from producer value ports to final attention write."""
import torch

def compile_path(routes,downstream):
    # routes[N,J,H,T]; downstream[N,T,A,O], includes recipient rho and all downstream gates.
    return torch.einsum('njht,ntao->njhao',routes,downstream)

def execute_path(operator,source_delta):
    return torch.einsum('njhao,njha->no',operator,source_delta)

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.manual_seed(61207);torch.set_num_threads(2)
    routes=torch.randn(3,3,9,7,dtype=torch.float64);down=torch.randn(3,7,2,11,dtype=torch.float64);z=torch.randn(3,3,9,2,dtype=torch.float64)
    K=compile_path(routes,down);direct=torch.einsum('njht,njha,ntao->no',routes,z,down);compiled=execute_path(K,z)
    error=float((direct-compiled).norm()/direct.norm());assert error<1e-12
    z2=torch.randn_like(z);sumerror=float((execute_path(K,z+z2)-execute_path(K,z)-execute_path(K,z2)).norm()/execute_path(K,z+z2).norm());assert sumerror<1e-12
    result=dict(relative_contraction_error=error,relative_composition_error=sumerror,scope='Exact linear map over independent first-value ports with conditional routes/downstream fixed; not new model output or semantic evidence.')
    out=Path(__file__).with_name('CONDITIONAL_TOKEN_PATH_V1_CONTROL.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
