"""Capture and absolute-reset complete attention/MLP writes for mediation tests."""

from __future__ import annotations


class ModuleWriteError(ValueError): pass


def prefix_hybrid(background, source, semantic_positions):
    if background.shape != source.shape or background.ndim != 3: raise ModuleWriteError("writes must be equal [batch,token,width] tensors")
    if len(semantic_positions) != background.shape[0]: raise ModuleWriteError("semantic-position count mismatch")
    result=background.clone()
    for row,stop in enumerate(semantic_positions):
        stop=int(stop)
        if not 0<=stop<background.shape[1]: raise ModuleWriteError("semantic position out of range")
        result[row,:stop+1]=source[row,:stop+1].to(result)
    return result


def capture_writes(sites, execute):
    captured={}; handles=[]
    for name,(module,kind) in sites.items():
        def hook(_module,_args,output,name=name,kind=kind):
            value=output[0] if kind=="attention" else output
            captured.setdefault(name,[]).append(value.detach().clone())
        handles.append(module.register_forward_hook(hook))
    try: output=execute()
    finally:
        for handle in handles: handle.remove()
    if set(captured)!=set(sites) or any(len(values)!=1 for values in captured.values()): raise ModuleWriteError("each module must execute exactly once")
    return output,{name:values[0] for name,values in captured.items()}


def execute_with_writes(sites, absolute, execute):
    if set(absolute)!=set(sites): raise ModuleWriteError("absolute writes must match selected sites")
    calls={name:0 for name in sites}; handles=[]
    for name,(module,kind) in sites.items():
        def hook(_module,_args,output,name=name,kind=kind):
            native=output[0] if kind=="attention" else output; changed=absolute[name]
            if changed.shape!=native.shape: raise ModuleWriteError("absolute write shape mismatch")
            calls[name]+=1; changed=changed.to(native)
            return (changed,)+tuple(output[1:]) if kind=="attention" else changed
        handles.append(module.register_forward_hook(hook))
    try: result=execute()
    finally:
        for handle in handles: handle.remove()
    if any(count!=1 for count in calls.values()): raise ModuleWriteError("each selected module must execute exactly once")
    return result


def hybrid_bank(off,on,names,semantic_positions):
    names=tuple(names)
    if not names or len(names)!=len(set(names)) or any(name not in off or name not in on for name in names): raise ModuleWriteError("invalid mediator names")
    return {name:prefix_hybrid(off[name],on[name],semantic_positions) for name in names}
