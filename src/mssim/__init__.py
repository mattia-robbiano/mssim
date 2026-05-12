"""
mssim — Multi System Simulator
==============================
A quantum circuit classical simulation package supporting multiple simulation engines.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Core API
from .circuits.model import CircuitModel
from .circuits.library import build_circuit
from .engines.registry import build_engines
from .executor import Executor

__all__ = [
    "CircuitModel",
    "build_circuit",
    "build_engines",
    "Executor",
    "__version__",
]

