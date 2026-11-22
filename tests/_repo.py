"""Repo-root resolution robust to mutmut's ``mutants/`` copy (quality gate 7).

mutmut copies only ``model-forge/`` and ``tests/`` into ``mutants/`` and runs
the suite there; the sibling packages (``model-forge-providers``/``-image``/
``-video``), ``docs/`` and ``model-forge-desktop`` live only at the REAL repo
root. Resolving the root by walking up to the ancestor that actually contains
the siblings keeps the suite runnable under mutation testing, so gate 7 can be
measured at all (it was previously "not measured" because the copy broke the
path-anchored loads).

``repo_root()`` returns the nearest ancestor of this file that holds
``model-forge-providers`` (the real repo root both normally and under
``mutants/``), falling back to ``parent.parent`` if no ancestor matches.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "model-forge-providers").is_dir():
            return cand
    return here.parent.parent
