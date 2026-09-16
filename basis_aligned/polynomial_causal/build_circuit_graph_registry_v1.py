#!/usr/bin/env python3
"""Build an auditable registry of extracted sparse-circuit packages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
POLY = Path(__file__).resolve().parent
PACKAGES = POLY / "extracted_circuits"
JSON_OUT = POLY / "CIRCUIT_GRAPH_REGISTRY_V1.json"
MARKDOWN_OUT = POLY / "CIRCUIT_GRAPH_REGISTRY_V1.md"
TRAITS = ("ood_prediction", "extraction", "selective_removal", "composition_reuse")
PORT_KEYS = ("dynamic_input_ports", "input_ports", "activation_ports", "public_data_inputs",
             "external_inputs", "inputs", "ports")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _ports(manifest: dict) -> tuple[str | None, list[str]]:
    for key in PORT_KEYS:
        values = _list(manifest.get(key))
        if values:
            return key, values
    return None, []


def _dependencies(manifest: dict, package_names: set[str]) -> list[str]:
    values = []
    dependency = manifest.get("dependency")
    if isinstance(dependency, str):
        values.append(dependency)
    values.extend(_list(manifest.get("dependencies")))
    return sorted({value for value in values if value in package_names})


def _verify(package: Path) -> dict:
    verifier = package / "verify_export.py"
    if not verifier.is_file():
        return {"status": "not_available", "path": None, "terminal": None}
    try:
        completed = subprocess.run(
            [sys.executable, str(verifier)], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "failed", "path": _display(verifier), "terminal": None}
    terminal = None
    if completed.returncode == 0:
        try:
            terminal = json.loads(completed.stdout)["terminal"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return {
        "status": "passed" if completed.returncode == 0 and terminal else "failed",
        "path": _display(verifier),
        "terminal": terminal,
    }


def _cycles(records: list[dict]) -> list[list[str]]:
    graph = {record["package"]: record["dependencies"] for record in records}
    active = []
    complete = set()
    found = set()

    def visit(node):
        if node in active:
            cycle = active[active.index(node):] + [node]
            rotations = [tuple(cycle[index:-1] + cycle[:index] + [cycle[index]])
                         for index in range(len(cycle) - 1)]
            found.add(min(rotations))
            return
        if node in complete:
            return
        active.append(node)
        for child in graph.get(node, []):
            visit(child)
        active.pop()
        complete.add(node)

    for node in sorted(graph):
        visit(node)
    return [list(cycle) for cycle in sorted(found)]


def build_registry(packages: Path = PACKAGES, run_verifiers: bool = True) -> dict:
    package_dirs = sorted((path for path in packages.iterdir() if path.is_dir()), key=lambda path: path.name)
    package_names = {path.name for path in package_dirs}
    records = []
    for package in package_dirs:
        manifest_path = package / "manifest.json"
        readme_path = package / "README.md"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        port_key, ports = _ports(manifest)
        declared_traits = manifest.get("four_traits") if isinstance(manifest.get("four_traits"), dict) else {}
        traits = {trait: declared_traits.get(trait) if trait in declared_traits else None for trait in TRAITS}
        verifier = _verify(package) if run_verifiers else {
            "status": "available_not_run" if (package / "verify_export.py").is_file() else "not_available",
            "path": _display(package / "verify_export.py") if (package / "verify_export.py").is_file() else None,
            "terminal": None,
        }
        all_traits = all(traits[trait] is True for trait in TRAITS)
        if all_traits and verifier["status"] == "passed":
            maturity = "four_trait_verified"
        elif all_traits:
            maturity = "four_trait_declared_unverified"
        else:
            maturity = "partial_or_unassessed"
        gaps = []
        if not manifest_path.is_file():
            gaps.append("manifest")
        else:
            if not manifest.get("schema"):
                gaps.append("schema")
            if not ports:
                gaps.append("declared_inputs")
            for trait in TRAITS:
                if traits[trait] is not True:
                    gaps.append(trait)
            if all_traits and verifier["status"] != "passed":
                gaps.append("passing_verifier")
        external_activations = manifest.get("external_activation_inputs")
        if external_activations is None:
            external_activations = manifest.get("external_native_activation_inputs")
        nodes = _list(manifest.get("nodes")) or _list(manifest.get("internal_native_ports"))
        records.append({
            "package": package.name,
            "relative_directory": _display(package),
            "manifest": _display(manifest_path) if manifest_path.is_file() else None,
            "manifest_sha256": sha(manifest_path) if manifest_path.is_file() else None,
            "readme": _display(readme_path) if readme_path.is_file() else None,
            "schema": manifest.get("schema"),
            "nodes": nodes,
            "node_count": len(nodes) if nodes else None,
            "declared_input_field": port_key,
            "declared_inputs": ports,
            "declared_input_count": len(ports) if ports else None,
            "external_activation_inputs": external_activations,
            "deterministic_context_inputs": manifest.get("deterministic_context_inputs"),
            "learned_parameters": manifest.get("learned_parameters"),
            "dependencies": _dependencies(manifest, package_names),
            "four_traits": traits,
            "verification": verifier,
            "maturity": maturity,
            "gaps": gaps,
            "scope": manifest.get("scope"),
        })
    cycles = _cycles(records)
    counts = {
        "packages": len(records),
        "with_manifest": sum(record["manifest"] is not None for record in records),
        "with_declared_inputs": sum(record["declared_inputs"] != [] for record in records),
        "four_trait_verified": sum(record["maturity"] == "four_trait_verified" for record in records),
        "four_trait_declared_unverified": sum(record["maturity"] == "four_trait_declared_unverified" for record in records),
        "partial_or_unassessed": sum(record["maturity"] == "partial_or_unassessed" for record in records),
        "explicit_dependency_edges": sum(len(record["dependencies"]) for record in records),
        "dependency_cycles": len(cycles),
    }
    return {
        "schema": "circuit_graph_registry_v1",
        "counts": counts,
        "four_trait_order": list(TRAITS),
        "verified_four_trait_packages": [record["package"] for record in records
                                         if record["maturity"] == "four_trait_verified"],
        "dependency_cycles": cycles,
        "packages": records,
        "interpretation": "Missing trait metadata is unknown, not false. Verified means all four traits are declared and the package's hash/evidence verifier passed during this build.",
    }


def render_markdown(registry: dict) -> str:
    counts = registry["counts"]
    lines = [
        "# Circuit graph registry V1", "",
        "This is a generated inventory of extracted circuit boundaries. Missing evidence is recorded as unknown, not as a failed scientific claim.", "",
        f"Packages: **{counts['packages']}**; manifests: **{counts['with_manifest']}**; declared input boundaries: **{counts['with_declared_inputs']}**; verified four-trait circuits: **{counts['four_trait_verified']}**.", "",
        "| package | maturity | inputs | external activations | nodes | next gap |", "|---|---:|---:|---:|---:|---|",
    ]
    for record in registry["packages"]:
        gaps = ", ".join(record["gaps"][:3]) + ("…" if len(record["gaps"]) > 3 else "")
        lines.append("| {package} | {maturity} | {inputs} | {activations} | {nodes} | {gaps} |".format(
            package=record["package"], maturity=record["maturity"],
            inputs=record["declared_input_count"] if record["declared_input_count"] is not None else "?",
            activations=record["external_activation_inputs"] if record["external_activation_inputs"] is not None else "?",
            nodes=record["node_count"] if record["node_count"] is not None else "?", gaps=gaps or "none",
        ))
    lines.extend(["", "Explicit dependency cycles: **{}**.".format(counts["dependency_cycles"]), ""])
    return "\n".join(lines)


def main():
    registry = build_registry()
    JSON_OUT.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    MARKDOWN_OUT.write_text(render_markdown(registry))
    print(json.dumps({"terminal": "circuit_graph_registry_built", **registry["counts"],
                      "verified": registry["verified_four_trait_packages"]}, indent=2, sort_keys=True))
    if registry["counts"]["dependency_cycles"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
