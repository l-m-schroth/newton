# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


class ChronoGroundTruthError(RuntimeError):
    pass


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "chrono").is_dir() and (parent / "newton").is_dir():
            return parent
    return start


def _default_chrono_build_dir(repo_root: Path) -> Path:
    return repo_root / "chrono" / "build"


def _default_chrono_lib_dir(repo_root: Path) -> Path:
    return _default_chrono_build_dir(repo_root) / "lib"


def _default_bin_path(repo_root: Path) -> Path:
    # Default out-of-source build location for the CLI (see chrono_gt/README.md).
    return repo_root / "newton" / "newton" / "tests" / "tires" / "chrono_gt" / "build" / "newton_chrono_gt"


def _binary_from_env() -> Path | None:
    raw = os.getenv("NEWTON_CHRONO_GT_BIN", "").strip()
    return Path(raw) if raw else None


def chrono_gt_binary() -> Path:
    """Return path to the Chrono ground-truth CLI binary.

    Resolution order:
    1) `NEWTON_CHRONO_GT_BIN`
    2) `newton/newton/tests/tires/chrono_gt/build/newton_chrono_gt`
    """
    env_bin = _binary_from_env()
    if env_bin is not None:
        return env_bin

    repo_root = _find_repo_root(Path(__file__).resolve())
    return _default_bin_path(repo_root)


def _build_instructions(repo_root: Path) -> str:
    chrono_dir = repo_root / "chrono" / "build" / "cmake"
    src_dir = repo_root / "newton" / "newton" / "tests" / "tires" / "chrono_gt"
    build_dir = src_dir / "build"
    return (
        "Build the Chrono ground-truth CLI:\n"
        f"  cmake -S {src_dir} -B {build_dir} -DChrono_DIR={chrono_dir}\n"
        f"  cmake --build {build_dir} -j\n"
        "\n"
        "Or set `NEWTON_CHRONO_GT_BIN` to an existing binary."
    )


def run_chrono_gt(
    request: dict[str, Any],
    *,
    timeout_s: float = 30.0,
    bin_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run Chrono ground-truth CLI and return parsed JSON response."""
    repo_root = _find_repo_root(Path(__file__).resolve())
    exe = Path(bin_path) if bin_path is not None else chrono_gt_binary()

    if not exe.exists():
        raise ChronoGroundTruthError(f"Missing Chrono GT binary at '{exe}'.\n\n{_build_instructions(repo_root)}")

    # Ensure Chrono shared libs are discoverable at runtime (local build tree).
    chrono_lib = _default_chrono_lib_dir(repo_root)
    env = os.environ.copy()
    if chrono_lib.is_dir():
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{chrono_lib}:{ld_path}" if ld_path else str(chrono_lib)

    proc = subprocess.run(
        [str(exe)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout_s,
        check=False,
    )

    if proc.returncode != 0:
        raise ChronoGroundTruthError(
            "Chrono GT CLI failed.\n"
            f"  cmd: {request.get('cmd')}\n"
            f"  returncode: {proc.returncode}\n"
            f"  stdout: {proc.stdout}\n"
            f"  stderr: {proc.stderr}"
        )

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ChronoGroundTruthError(
            "Chrono GT CLI returned non-JSON output.\n"
            f"  error: {e}\n"
            f"  stdout: {proc.stdout}\n"
            f"  stderr: {proc.stderr}"
        ) from e


