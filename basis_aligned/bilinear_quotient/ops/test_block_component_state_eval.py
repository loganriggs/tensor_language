import block_component_state_eval as component


def test_subset_inventory_and_ids():
    assert len(component.subsets()) == 8
    assert component.arm_id(()) == "empty"
    assert component.arm_id(component.COMPONENTS) == "entry+attention+mlp"


def test_factorial_accounting_is_exact_for_additive_values():
    weights = {"entry": 3.0, "attention": 2.0, "mlp": 1.0}
    values = {
        subset: sum(weights[name] for name in subset)
        for subset in component.subsets()
    }
    accounting = component.factorial_accounting(values)
    assert accounting["shapley"] == weights
    assert accounting["efficiency_error"] == 0.0


def test_assemble_uses_native_block_order():
    class Piece:
        def __init__(self, text):
            self.text = text
        def __add__(self, other):
            return Piece(f"({self.text}+{other.text})")
    base = {name: Piece(name[0]) for name in component.COMPONENTS}
    changed = {name: Piece(name[0].upper()) for name in component.COMPONENTS}
    result = component.assemble(base, changed, ("attention",))
    assert result.text == "((e+A)+m)"
