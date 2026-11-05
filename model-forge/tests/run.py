"""Stdlib test runner (no pytest needed): discovers test_* functions in this
dir and runs them. Usage: python3 tests/run.py"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))  # plugin root: `model_forge` package
    fails = 0
    for path in sorted(here.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(mod, name)()
                print(f"PASS {path.stem}::{name}")
            except Exception:
                fails += 1
                print(f"FAIL {path.stem}::{name}")
                traceback.print_exc()
    print(f"\n{fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
