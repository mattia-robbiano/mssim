"""
csbench.engines.abstract
========================
Abstract base class that every simulation engine must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class BenchmarkEngine(ABC):
    """
    Base class for all simulation engines.

    Every concrete engine receives a QASM string (the canonical circuit
    representation), a flat list/array of numeric parameters (for
    parameterised gates, in the order they appear in the circuit), and a
    Pauli observable expressed as a plain list of single-qubit labels
    (e.g. ``["Z", "I", "Z", "I"]``).

    Returns
    -------
    tuple[float, float, float | None]
        ``(expectation_value, elapsed_seconds, fidelity_or_None)``
    """

    # ------------------------------------------------------------------
    # Subclasses may add their own fields via dataclass inheritance.
    # ------------------------------------------------------------------

    @abstractmethod
    def expectation_value(
        self,
        qasm_circuit: str,
        parameters: Sequence[float],
        observable: Sequence[str],
    ) -> tuple[float, float, float | None]:
        """
        Compute ⟨ψ|O|ψ⟩ for the parametrised circuit.

        Parameters
        ----------
        qasm_circuit:
            OpenQASM 2.0 string representing the (possibly parametrised)
            circuit skeleton.
        parameters:
            Flat sequence of real-valued gate parameters, in the order the
            parametrised gates appear in *qasm_circuit*.
        observable:
            Pauli operator as a list of single-qubit labels, length == nqubits.
            Allowed labels: ``"I"``, ``"X"``, ``"Y"``, ``"Z"``.

        Returns
        -------
        expval : float
            The expectation value ⟨O⟩.
        elapsed : float
            Wall-clock seconds for the simulation kernel (excluding I/O and
            any one-time setup that is amortised across runs).
        fidelity : float or None
            Truncation / approximation fidelity where applicable; ``None``
            for exact engines.
        """

    # ------------------------------------------------------------------
    # Optional: engines may override for per-engine warm-up / teardown.
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Called once before any ``expectation_value`` call in a batch."""

    def teardown(self) -> None:
        """Called once after all ``expectation_value`` calls in a batch."""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable engine identifier (class name by default)."""
        return type(self).__name__
