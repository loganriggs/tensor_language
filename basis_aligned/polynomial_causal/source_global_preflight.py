#!/usr/bin/env python3
"""Small dependency-free undefined-global preflight for expensive experiment files."""

from __future__ import annotations

import ast
import builtins
import symtable
from pathlib import Path


IMPLICIT_GLOBALS = {
    "__file__", "__name__", "__package__", "__spec__", "__loader__",
    "__cached__", "__builtins__",
}


def target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(target_names(child) for child in node.elts))
    return set()


def module_definitions(tree: ast.Module) -> set[str]:
    defined = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                defined.update(target_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            defined.update(target_names(node.target))
    return defined


def undefined_global_names(path: Path) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    defined = module_definitions(tree)
    referenced = set()

    def walk(table: symtable.SymbolTable) -> None:
        for symbol in table.get_symbols():
            if symbol.is_referenced() and symbol.is_global():
                referenced.add(symbol.get_name())
        for child in table.get_children():
            walk(child)

    walk(symtable.symtable(source, str(path), "exec"))
    allowed = defined | set(dir(builtins)) | IMPLICIT_GLOBALS
    return sorted(referenced - allowed)


def require_defined_globals(paths: list[Path]) -> None:
    failures = {
        str(path): missing for path in paths
        if (missing := undefined_global_names(path))
    }
    if failures:
        raise RuntimeError(f"undefined global preflight failed: {failures}")
