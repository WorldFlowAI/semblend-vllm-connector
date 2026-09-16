"""Arm the compatibility guards against real vLLM types, and say which ran.

The connector falls back to an in-repo shim for ``KVConnectorBase_V1`` when
vLLM is absent, so a suite run on a machine without vLLM checks the startup
gate against this repo's own stand-ins — exactly the version drift the gate
exists to catch. This conftest resolves the real vLLM types when the platform
allows it, and prints one line in the header of every run saying whether the
real-type guards were armed or skipped. ``tests/README-compat.md`` records what
runs where.

Two ways in, in descending fidelity:

1. An installed vLLM wheel. Everything imports; this is what CI on Linux does.
2. A read-only source checkout named by ``SEMBLEND_VLLM_SOURCE``. Only the
   pure-Python modules resolve, which is enough for the type guards and is the
   only option on a machine that cannot install the wheel.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from semblend_vllm_connector.capture_writer import close_live_writers

SOURCE_ENV = "SEMBLEND_VLLM_SOURCE"
REQUIRE_ENV = "SEMBLEND_REQUIRE_VLLM_GUARDS"

# The vLLM modules the connector's startup gate imports that are pure Python in
# 0.29 and therefore loadable from a source checkout. Everything else vLLM
# exposes (the connector base, the config tree, kv_cache_utils) pulls in the
# compiled extension or the engine runtime and is reachable only from a real
# installation; see README-compat.md.
REAL_TYPE_MODULES = ("vllm.v1.kv_cache_interface",)


@dataclass(frozen=True)
class VllmTypes:
    """The real vLLM modules the guards may use, and their provenance."""

    origin: str
    modules: dict[str, types.ModuleType] = field(default_factory=dict)
    # Compromises made to get the import through, reported in the header so a
    # reader is never told "real types" without being told what it took.
    notes: tuple[str, ...] = ()
    # Set only when nothing could be loaded; doubles as the skip reason.
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.reason is None


def _import_all(names: tuple[str, ...]) -> dict[str, types.ModuleType]:
    return {name: importlib.import_module(name) for name in names}


def _installed_vllm() -> VllmTypes | None:
    """Real installation, imported the way production imports it."""
    try:
        modules = _import_all(REAL_TYPE_MODULES)
    except Exception:
        return None
    version = getattr(sys.modules.get("vllm"), "__version__", "unknown version")
    return VllmTypes(origin=f"installed vllm {version}", modules=modules)


def _source_root() -> Path | None:
    """A vLLM source checkout to read the types out of, if one is configured."""
    configured = os.environ.get(SOURCE_ENV)
    if not configured:
        return None
    root = Path(configured).expanduser()
    if not (root / "vllm" / "v1" / "kv_cache_interface.py").is_file():
        return None
    return root


def _fill_missing_torch_symbols() -> tuple[str, ...]:
    """Fill in torch names vLLM 0.29 imports that an older local torch lacks.

    ``vllm.utils.torch_utils`` does ``from torch.library import Library,
    infer_schema`` at module scope, and ``infer_schema`` arrived in a torch
    newer than some platforms can install. The dataclasses the guards use never
    call it, so the name is filled in with a function that raises: the import
    proceeds, and anything that genuinely needs the real symbol fails loudly
    rather than running against a stand-in.
    """
    try:
        import torch
        import torch.library
    except Exception as exc:  # torch itself is optional for most of this suite
        raise RuntimeError(f"torch is unavailable ({exc!r})") from exc

    if hasattr(torch.library, "infer_schema"):
        return ()

    def _infer_schema_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "torch.library.infer_schema is filled in by tests/conftest.py for "
            "import only; this torch predates the real symbol"
        )

    torch.library.infer_schema = _infer_schema_unavailable
    return (
        f"torch {torch.__version__} predates torch.library.infer_schema, filled in for import only",
    )


def _undo_torch_fill(notes: tuple[str, ...]) -> None:
    """Drop the import-only fill when the import failed anyway.

    Nothing else in the session should find a torch symbol that exists only to
    raise, least of all code that probes for it with ``hasattr``.
    """
    if not notes:
        return
    import torch.library

    if hasattr(torch.library, "infer_schema"):
        delattr(torch.library, "infer_schema")


def _register_source_package(root: Path) -> None:
    """Expose the checkout's submodules without running ``vllm/__init__.py``.

    That module imports the compiled extension and the engine, neither of which
    exists in a source tree. A package object whose ``__path__`` is the tree
    lets the normal import machinery reach the pure-Python submodules beneath
    it, so the classes that come back are vLLM's own and not a stand-in.
    """
    package = types.ModuleType("vllm")
    package.__path__ = [str(root / "vllm")]
    sys.modules["vllm"] = package


def _source_vllm(root: Path) -> VllmTypes:
    try:
        notes = _fill_missing_torch_symbols()
    except RuntimeError as exc:
        return VllmTypes(origin=str(root), reason=str(exc))
    _register_source_package(root)
    try:
        modules = _import_all(REAL_TYPE_MODULES)
    except Exception as exc:
        # Leave no half-registered package behind: the rest of the suite relies
        # on vLLM being absent so the connector takes its documented fallback.
        for name in [n for n in sys.modules if n == "vllm" or n.startswith("vllm.")]:
            del sys.modules[name]
        _undo_torch_fill(notes)
        return VllmTypes(
            origin=str(root),
            reason=f"source tree {root} could not be imported: {exc!r}",
        )
    return VllmTypes(origin=f"source tree {root}", modules=modules, notes=notes)


def _probe_vllm_types() -> VllmTypes:
    installed = _installed_vllm()
    if installed is not None:
        return installed
    root = _source_root()
    if root is None:
        return VllmTypes(
            origin="none",
            reason=(f"vLLM is not installed and no usable source tree was named by ${SOURCE_ENV}"),
        )
    return _source_vllm(root)


# Probed once per session, at conftest import: the modules go into sys.modules
# either way, so repeating the probe could only produce a different answer by
# mutating global import state mid-run.
VLLM_TYPES = _probe_vllm_types()


def pytest_configure(config) -> None:
    """Let a job demand the real types rather than trust a header line.

    A run that means to check vLLM compatibility and silently fell back to the
    shim is a green run that proves nothing, so make the fallback fatal when
    the caller says the guards are the point of the run.
    """
    if VLLM_TYPES.available or not os.environ.get(REQUIRE_ENV):
        return
    raise pytest.UsageError(
        f"${REQUIRE_ENV} is set, but the real vLLM types could not be loaded: {VLLM_TYPES.reason}"
    )


def pytest_report_header(config) -> list[str]:
    """One line at the top of every run: did the real-type guards arm?"""
    if not VLLM_TYPES.available:
        return [
            f"real-vLLM type guards: SKIPPED ({VLLM_TYPES.reason})",
            "  the compatibility guards run against this repo's shim only; "
            "see tests/README-compat.md",
        ]
    loaded = ", ".join(sorted(VLLM_TYPES.modules))
    return [
        f"real-vLLM type guards: ACTIVE via {VLLM_TYPES.origin} [{loaded}]",
        *[f"  note: {note}" for note in VLLM_TYPES.notes],
    ]


@pytest.fixture(autouse=True)
def _close_capture_writers():
    """Stop every capture writer a test left running, before the next test.

    A connector starts its writer on the first captured layer and stops it in
    ``shutdown``; a test that never calls shutdown would otherwise leave a
    thread and a queue of host tensors alive for the rest of the session. The
    close is bounded, so a test that deliberately wedges a store costs the
    writer's close timeout here and does not hang the run.
    """
    yield
    close_live_writers()


@pytest.fixture(scope="session")
def vllm_types() -> VllmTypes:
    """The real vLLM modules; skips loudly when none could be loaded."""
    if not VLLM_TYPES.available:
        pytest.skip(f"real vLLM types unavailable: {VLLM_TYPES.reason}")
    return VLLM_TYPES


@pytest.fixture(scope="session")
def kv_cache_interface(vllm_types: VllmTypes) -> types.ModuleType:
    """vLLM's own ``vllm.v1.kv_cache_interface``: specs, groups, KVCacheConfig."""
    return vllm_types.modules["vllm.v1.kv_cache_interface"]


@pytest.fixture(scope="session")
def vllm_module_source(vllm_types: VllmTypes):
    """Locate the file backing a vLLM module without importing it.

    ``find_spec`` resolves the origin of a submodule by importing only its
    parent packages, which is the one way to inspect modules whose own imports
    need the full engine runtime (``vllm.v1.core.kv_cache_utils``).
    """

    def _locate(module_name: str) -> Path:
        spec = importlib.util.find_spec(module_name)
        if spec is None or not spec.origin:
            pytest.skip(f"{module_name} has no resolvable source file")
        return Path(spec.origin)

    return _locate
