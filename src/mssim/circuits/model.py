"""
mssim.circuits.model
=======================
``CircuitModel`` — the single object passed.

It holds:
- the circuit definition in a canonical format (QASM string)
- the observable (Pauli string list)
- a parameter sampler (returns a fresh random parameter vector each call)
- all circuit-specific metadata (n_qubits, depth, circuit family name, …)
"""


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np


@dataclass
class CircuitModel:
    """
    Object containing all the needed information on the model. The core is the qasm object, passed from the circuit builder and the observable.
    Also parameters name, number of qubit, depth

    Parameters
    ----------
    name : str
        Circuit family name (e.g. ``"kicked_ising"``).
    n_qubits : int
        Number of qubits.
    depth : int
        Circuit depth (meaning is family-specific).
    qasm : str
        OpenQASM 2.0 string with numeric gate parameters *as placeholders*
        that will be replaced by :meth:`sample_parameters`.
        The convention is that parametrised gates are generated with
        concrete numeric literals so that the QASM string is valid on its own;
        calling :meth:`sample_parameters` returns a new flat parameter vector
        to be bound at runtime.
    observable : list[str]
        Pauli operator as a list of single-qubit labels (length == n_qubits).
        E.g. ``["Z", "I", "Z"]``.
    n_params : int
        Number of free (random) parameters in the circuit.
    parameter_sampler : Callable[[], list[float]]
        Zero-argument callable that returns a fresh flat parameter list of
        length ``n_params``.  Defaults to uniform sampling in ``[0, 2π)``.
    metadata : dict[str, Any]
        Free-form dict for circuit-family-specific extra info (e.g. coupling
        map, Hamiltonian coefficients, …).
    """

    name: str
    n_qubits: int
    depth: int
    qasm: str
    observable: list[str]
    n_params: int
    parameter_sampler: Callable[[], list[float]] = field(default=None,repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        if len(self.observable) != self.n_qubits:
            raise ValueError(f"Observable length ({len(self.observable)}) must equal "f"n_qubits ({self.n_qubits}).")
        
        if self.parameter_sampler is None:
            n = self.n_params
            self.parameter_sampler = lambda: list(np.random.uniform(0, 2 * np.pi, size=n))


    def sample_parameters(self) -> list[float]:
        return self.parameter_sampler()


    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "observable": self.observable,
            "n_params": self.n_params,
            "metadata": self.metadata,
        }


    def __repr__(self) -> str:
        return (
            f"CircuitModel(name={self.name!r}, n_qubits={self.n_qubits}, "
            f"depth={self.depth}, n_params={self.n_params})"
        )
