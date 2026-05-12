"""
csbench.engines.quimb_engine
=============================
Engine wrapper for Quimb's CircuitMPS tensor-network simulator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from qibo import Circuit
from qibo.gates.abstract import ParametrizedGate
from quimb.tensor import CircuitMPS
from quimb import pauli

from .abstract import BenchmarkEngine


# ---------------------------------------------------------------------------
# Qibo → Quimb gate name map
# ---------------------------------------------------------------------------

GATE_MAP: dict[str, str] = {
    "h":        "H",
    "x":        "X",
    "y":        "Y",
    "z":        "Z",
    "s":        "S",
    "t":        "T",
    "rx":       "RX",
    "ry":       "RY",
    "rz":       "RZ",
    "u3":       "U3",
    "cx":       "CX",
    "cnot":     "CNOT",
    "cy":       "CY",
    "cz":       "CZ",
    "iswap":    "ISWAP",
    "swap":     "SWAP",
    "ccx":      "CCX",
    "ccy":      "CCY",
    "ccz":      "CCZ",
    "toffoli":  "TOFFOLI",
    "cswap":    "CSWAP",
    "fredkin":  "FREDKIN",
    "fsim":     "fsim",
    "measure":  "measure",
}


def _qibo_to_quimb(nqubits: int, qibo_circ: Circuit, **circuit_kwargs) -> CircuitMPS:
    """
    Convert a Qibo ``Circuit`` to a Quimb ``CircuitMPS``.

    Measurement gates are silently skipped.  Any unrecognised gate raises
    ``ValueError``.

    Parameters
    ----------
    nqubits:
        Number of qubits (passed separately so the function is pure).
    qibo_circ:
        A *fully parametrised* Qibo circuit (parameters already bound).
    **circuit_kwargs:
        Extra keyword arguments forwarded to ``CircuitMPS.__init__``
        (e.g. ``max_bond``).

    Returns
    -------
    CircuitMPS
        Quimb circuit ready for contraction.
    """
    circ = CircuitMPS(nqubits, **circuit_kwargs)

    for gate in qibo_circ.queue:
        gate_name: str | None = getattr(gate, "name", None)
        quimb_name: str | None = GATE_MAP.get(gate_name, None)  # type: ignore[arg-type]

        if quimb_name == "measure":
            continue
        if quimb_name is None:
            raise ValueError(
                f"Gate '{gate_name}' is not supported by the Quimb backend. "
                f"Supported gates: {sorted(GATE_MAP.keys())}"
            )

        params = tuple(getattr(gate, "parameters", ()))
        qubits = tuple(getattr(gate, "qubits", ()))
        is_parametrised = isinstance(gate, ParametrizedGate) and getattr(
            gate, "trainable", True
        )

        if is_parametrised:
            circ.apply_gate(quimb_name, *params, *qubits, parametrized=True)
        else:
            circ.apply_gate(quimb_name, *params, *qubits)

    return circ


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class QuimbEngine(BenchmarkEngine):
    """
    Simulation engine based on Quimb's MPS circuit simulator.

    Parameters
    ----------
    max_bond_dimension:
        Maximum MPS bond dimension.  ``None`` means no truncation.
    contraction_optimizer:
        Contraction path optimiser passed to ``.contract()``.
    """

    max_bond_dimension: int | None = None
    contraction_optimizer: str = "auto-hq"

    # ------------------------------------------------------------------

    def expectation_value(
        self,
        qasm_circuit: str,
        parameters: Sequence[float],
        observable: Sequence[str],
    ) -> tuple[float, float, float | None]:
        """
        Compute ⟨O⟩ via Quimb MPS contraction.

        The circuit is built *twice*: once inside the timed region to
        measure the cost of construction + contraction together, and once
        outside (no-truncation reference) to obtain the fidelity estimate.

        Returns
        -------
        expval : float
        elapsed : float   (seconds, MPS construction + contraction)
        fidelity : float  (Quimb's built-in fidelity estimate)
        """
        circuit = Circuit.from_qasm(qasm_circuit)
        circuit.set_parameters(list(parameters))

        # ----- Timed region -----------------------------------------------
        t0 = time.perf_counter()

        mps_circ = _qibo_to_quimb(
            nqubits=circuit.nqubits,
            qibo_circ=circuit,
            max_bond=self.max_bond_dimension,
        )
        psi_ket = mps_circ.psi
        norm = psi_ket.norm(squared=True).real

        # Build |ψ_op⟩ = O|ψ⟩ for the non-identity sites only
        non_identity = {
            i: op.upper()
            for i, op in enumerate(observable)
            if op.upper() != "I"
        }
        psi_op = psi_ket.copy()
        for site, label in non_identity.items():
            psi_op.gate_(pauli(label), site)

        expval = (
            (psi_ket.H & psi_op)
            .contract(optimize=self.contraction_optimizer)
            .real
            / norm
        )
        elapsed = time.perf_counter() - t0
        # ------------------------------------------------------------------

        # Fidelity is estimated out-of-timing (needs a fresh MPS build)
        mps_for_fidelity = _qibo_to_quimb(
            nqubits=circuit.nqubits,
            qibo_circ=circuit,
            max_bond=self.max_bond_dimension,
        )
        fidelity = mps_for_fidelity.fidelity_estimate()

        return float(expval), elapsed, float(fidelity)

    @property
    def name(self) -> str:
        return f"quimb_mps(chi={self.max_bond_dimension})"
