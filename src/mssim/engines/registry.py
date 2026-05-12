"""
csbench.engines.registry
=========================
Central registry that maps short engine names to concrete engine classes
and provides a factory for instantiating them from settings dicts.

Supported short names
---------------------
``"tn"``      → :class:`~csbench.engines.quimb_engine.QuimbEngine`
``"mpstab"``  → :class:`~csbench.engines.mpstab_engine.MPStabEngine`
``"sv"``      → :class:`~csbench.engines.statevector_engine.StatevectorEngine`
``"all"``     → all of the above

Examples
--------
>>> from csbench.engines.registry import build_engines
>>> engines = build_engines(["tn", "sv"], max_bond_dimension=32)
"""

from __future__ import annotations

from typing import Any

from .abstract import BenchmarkEngine

# Lazy imports so that missing optional dependencies only fail when the
# specific engine is actually requested.
_ENGINE_REGISTRY: dict[str, str] = {
    "tn":     "mssim.engines.quimb_engine:QuimbEngine",
    "mpstab": "mssim.engines.mpstab_engine:MPStabEngine",
    "sv":     "mssim.engines.statevector_engine:StatevectorEngine",
}

ALL_ENGINE_KEYS = list(_ENGINE_REGISTRY.keys())


def _import_engine(key: str) -> type[BenchmarkEngine]:
    """Dynamically import and return the engine class for *key*."""
    if key not in _ENGINE_REGISTRY:
        raise KeyError(
            f"Unknown engine '{key}'. "
            f"Available: {sorted(_ENGINE_REGISTRY.keys())} or 'all'."
        )
    module_path, class_name = _ENGINE_REGISTRY[key].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_engines(
    keys: list[str],
    max_bond_dimension: int | None = None,
    **extra_kwargs: Any,
) -> list[BenchmarkEngine]:
    """
    Instantiate and return a list of engines.

    Parameters
    ----------
    keys:
        List of engine short names, e.g. ``["tn", "sv"]``.
        Pass ``["all"]`` to get every registered engine.
    max_bond_dimension:
        Forwarded to MPS-based engines (``tn``, ``mpstab``).
        Ignored by exact engines.
    **extra_kwargs:
        Additional keyword arguments forwarded to every engine constructor
        (engines silently ignore unknown fields via ``dataclass`` defaults).

    Returns
    -------
    list[BenchmarkEngine]
    """
    if keys == ["all"]:
        keys = ALL_ENGINE_KEYS

    engines: list[BenchmarkEngine] = []
    for key in keys:
        cls = _import_engine(key)
        # Only pass max_bond_dimension to engines that declare it
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        if "max_bond_dimension" in field_names and max_bond_dimension is not None:
            kwargs["max_bond_dimension"] = max_bond_dimension
        kwargs.update(
            {k: v for k, v in extra_kwargs.items() if k in field_names}
        )
        engines.append(cls(**kwargs))

    return engines
