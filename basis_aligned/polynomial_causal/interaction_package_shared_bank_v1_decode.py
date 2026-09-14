"""Reconstruct original program structures from an exact shared tensor bank."""
def decode(tree, bank):
    kind, value = tree
    if kind == 'tensor':
        return bank[value].clone()
    if kind == 'dict':
        return {key: decode(child, bank) for key, child in value}
    if kind == 'list':
        return [decode(child, bank) for child in value]
    if kind == 'tuple':
        return tuple(decode(child, bank) for child in value)
    if kind == 'literal':
        return value
    raise ValueError(kind)
