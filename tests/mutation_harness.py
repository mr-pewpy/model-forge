#!/usr/bin/env python3
"""Mutation-testing harness for the pure core modules (quality gate 7).

Why a custom harness: mutmut's test-to-mutant coverage mapping does not
correlate under this repo's multi-package layout (``model_forge`` is imported
via pytest.ini ``pythonpath = model-forge`` while mutmut swaps code under
``mutants/``), so it reports "no test case for any mutant" (see
docs/quality-gates.md gate 7). This harness sidesteps that entirely: it
mutates the REAL pure modules in place (backup/restore) with AST-level
operator swaps, runs the suite per mutant, and records killed/survived.

A mutant is KILLED when the suite exits non-zero (a test caught the change)
and SURVIVES when the suite still passes (the test suite does not pin that
behaviour). The surviving count is the honest gate-7 number; it is never
reported as 0 without a real run.

Scope: the pure, network-free modules where mutation is meaningful and the
suite runs in seconds — aliases.py, registry.py, schemas.py. A full-tree run
is a multi-hour CI job (tools.py/facade.py/broker.py need engine fixtures).

Usage:
    .venv/bin/python tests/mutation_harness.py [--json]

Parses cleanly to a killed/survived/total report; ``--json`` emits the rows.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGETS = [
    REPO / "model-forge" / "model_forge" / "aliases.py",
    REPO / "model-forge" / "model_forge" / "registry.py",
    REPO / "model-forge" / "model_forge" / "schemas.py",
]
PYTEST = REPO / ".venv" / "bin" / "pytest"

# AST node attribute -> list of replacement literals. Each swap produces one
# mutant. These are the classic first-order mutation operators: comparison /
# boolean / arithmetic operator swaps and constant flips.
_CMP_SWAPS = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}
_BOOL_SWAPS = {ast.And: ast.Or, ast.Or: ast.And}
_BINOP_SWAPS = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
}


def _const_replacements(node: ast.Constant):
    """Yield replacement constants for a literal (bool flip, int +/-1)."""
    v = node.value
    if isinstance(v, bool):
        yield ast.Constant(value=not v)
    elif isinstance(v, int) and not isinstance(v, bool):
        yield ast.Constant(value=v + 1)
        yield ast.Constant(value=v - 1)
    elif isinstance(v, str) and v == "":
        yield ast.Constant(value="x")


class _Mutator(ast.NodeTransformer):
    """Produce one mutant at a time: collect (node, replacement) candidates
    first, then apply the Nth on a fresh tree copy."""

    def __init__(self):
        self.candidates: list[tuple[str, ast.AST, object]] = []

    def collect(self, tree: ast.AST):
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                op = node.ops[0]
                if type(op) in _CMP_SWAPS:
                    self.candidates.append(("cmp", node, _CMP_SWAPS[type(op)]))
            elif isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_SWAPS:
                self.candidates.append(("bool", node, _BOOL_SWAPS[type(node.op)]))
            elif isinstance(node, ast.BinOp) and type(node.op) in _BINOP_SWAPS:
                self.candidates.append(("binop", node, _BINOP_SWAPS[type(node.op)]))
            elif isinstance(node, ast.Constant):
                for rep in _const_replacements(node):
                    self.candidates.append(("const", node, rep))
        return self.candidates


class _ApplyOne(ast.NodeTransformer):
    def __init__(self, kind: str, target: ast.AST, replacement):
        self.kind, self.target, self.replacement = kind, target, replacement
        self.applied = False

    def generic_visit(self, node):
        if node is self.target and not self.applied:
            self.applied = True
            if self.kind == "cmp":
                node.ops[0] = self.replacement()
            elif self.kind in ("bool", "binop"):
                node.op = self.replacement()
            elif self.kind == "const":
                return self.replacement
            return node
        return super().generic_visit(node)


def _apply_mutant(source: str, kind: str, target: ast.AST, replacement) -> str:
    tree = ast.parse(source)
    # Re-find the same node position in the fresh tree by index of the walk.
    mut = _Mutator()
    candidates = mut.collect(tree)
    for k, n, r in candidates:
        if k == kind and type(r) is type(replacement) and n.lineno == getattr(target, "lineno", -1) \
                and n.col_offset == getattr(target, "col_offset", -1):
            _ApplyOne(kind, n, r).visit(tree)
            break
    return ast.unparse(tree)


def _run_suite() -> bool:
    """True when the suite passes (mutant survived), False when it fails."""
    r = subprocess.run([str(PYTEST), "-q", "-x", "tests/"],
                       cwd=str(REPO), capture_output=True, text=True)
    return r.returncode == 0


def run_module(path: Path) -> dict:
    source = path.read_text()
    tree = ast.parse(source)
    mut = _Mutator()
    candidates = mut.collect(tree)
    backup = source
    killed = survived = 0
    rows = []
    for i, (kind, node, replacement) in enumerate(candidates):
        try:
            mutated = _apply_mutant(source, kind, node, replacement)
        except Exception:
            continue
        if mutated == source:
            continue
        path.write_text(mutated)
        try:
            ok = _run_suite()
        finally:
            path.write_text(backup)
        verdict = "survived" if ok else "killed"
        if ok:
            survived += 1
        else:
            killed += 1
        rows.append({"mutant": i, "file": path.name, "line": getattr(node, "lineno", "?"),
                     "kind": kind, "verdict": verdict})
    return {"file": path.name, "mutants": len(rows), "killed": killed,
            "survived": survived, "rows": rows}


def main() -> int:
    as_json = "--json" in sys.argv
    totals = {"killed": 0, "survived": 0, "mutants": 0}
    all_rows = []
    for path in TARGETS:
        if not path.exists():
            continue
        res = run_module(path)
        totals["killed"] += res["killed"]
        totals["survived"] += res["survived"]
        totals["mutants"] += res["mutants"]
        all_rows.extend(res["rows"])
        if not as_json:
            print(f"{res['file']}: {res['killed']} killed / {res['survived']} "
                  f"survived / {res['mutants']} total")
    if as_json:
        print(json.dumps({"totals": totals, "rows": all_rows}, indent=2))
    else:
        print(f"\nTOTAL: {totals['killed']} killed / {totals['survived']} survived"
              f" / {totals['mutants']} total")
        print(f"mutation score (killed/total): "
              f"{(totals['killed'] / totals['mutants'] * 100) if totals['mutants'] else 0:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
