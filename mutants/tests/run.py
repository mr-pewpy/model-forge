"""Test runner for Model Forge.

Canonical path: when pytest is importable, delegate to it (the full suite,
including fixture-based tests, runs exactly as ``pytest -q`` from the repo
root). No-pytest fallback: discover and call the plain ``test_*`` functions
that take no fixtures (the original registry/planner/credential tests).

Usage: python3 tests/run.py
"""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
import traceback
from pathlib import Path


def _has_pytest() -> bool:
    try:
        import pytest  # noqa: F401
        return True
    except ImportError:
        return False


def _run_pytest() -> int:
    """Delegate to pytest from the repo root (where pytest.ini lives)."""
    repo_root = Path(__file__).resolve().parent.parent
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-q"], cwd=str(repo_root))


def _run_stdlib() -> int:
    """No-pytest fallback: call the fixture-free test_* functions."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "model-forge"))
    fails = skipped = 0
    for path in sorted(here.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except ImportError as exc:
            # a test module needs pytest (or another dep) we lack — skip it
            # with a note rather than count it as a failure
            print(f"SKIP {path.stem}: {exc}")
            skipped += 1
            continue
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            if len(inspect.signature(fn).parameters) > 0:
                # takes fixtures (monkeypatch/tmp_path) — pytest-only
                print(f"SKIP {path.stem}::{name} (needs pytest fixtures)")
                skipped += 1
                continue
            try:
                fn()
                print(f"PASS {path.stem}::{name}")
            except Exception:
                fails += 1
                print(f"FAIL {path.stem}::{name}")
                traceback.print_exc()
    print(f"\n{fails} failure(s), {skipped} skipped (pytest-only)")
    return 1 if fails else 0


def main() -> int:
    if _has_pytest():
        return _run_pytest()
    return _run_stdlib()


if __name__ == "__main__":
    sys.exit(main())