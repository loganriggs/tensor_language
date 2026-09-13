"""Explicit parent/child field selection, with no accidental double counting."""
from pathlib import Path
import importlib.util

spec=importlib.util.spec_from_file_location('crossfirst_hierarchy_executor',Path(__file__).with_name('execute.py'))
executor=importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)


def split_fields(token_ids, mlp7_input, attention8_input, attention9_input,
                 rms8_squared, rms9, weights):
    child=executor.field(token_ids,mlp7_input,attention8_input,attention9_input,
                         rms8_squared,rms9,weights)
    p=weights['routing']
    parent=(executor._routing.routing(attention9_input,p,1)
            @ (attention9_input.double()@p['current_value_reader'])[...,None])[...,0]
    return {'parent':parent,'child':child,'remainder':parent-child}


def select_field(parts, components):
    """Sum disjoint hierarchy nodes; callers choose the intervention strength."""
    names=tuple(components)
    if len(set(names))!=len(names):
        raise ValueError('A component may be selected only once.')
    if not names or not set(names)<= {'parent','child','remainder'}:
        raise ValueError('Select parent, child, remainder, or child plus remainder.')
    if 'parent' in names and len(names)>1:
        raise ValueError('Parent already contains child and remainder; select disjoint nodes.')
    return sum(parts[name] for name in names)
