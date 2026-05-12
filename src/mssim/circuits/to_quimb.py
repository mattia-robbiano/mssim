"""
csbench.converters.to_quimb
=============================
Utilities for converting Qibo circuits into Quimb tensor-network circuits.

The authoritative gate name map lives here so that both the converter and
the :class:`~csbench.engines.quimb_engine.QuimbEngine` can share it.
"""

from __future__ import annotations

from typing import Sequence

from qibo import Circuit
from qibo.gates.abstract import ParametrizedGate
from quimb.tensor import CircuitMPS
from quimb.tensor import Circuit as QuimbCircuit

# ---------------------------------------------------------------------------
# Gate name map  (Qibo name → Quimb name)
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
    "measure":  "measure",   # will be skipped
}


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------


def qibo_to_quimb(
    qibo_circ: Circuit,
    *,
    use_mps: bool = True,
    **circuit_kwargs,
) -> CircuitMPS | QuimbCircuit:
    """
    Convert a *fully parametrised* Qibo ``Circuit`` to a Quimb circuit.

    Measurement gates are silently skipped.

    Parameters
    ----------
    qibo_circ:
        Qibo circuit with parameters already bound.
    use_mps:
        If ``True`` (default) returns a ``CircuitMPS``; otherwise a full
        statevector ``Circuit``.
    **circuit_kwargs:
        Extra keyword arguments forwarded to the Quimb circuit constructor,
        e.g. ``max_bond=32``.

    Returns
    -------
    CircuitMPS or quimb.tensor.Circuit

    Raises
    ------
    ValueError
        If the circuit contains a gate not listed in :data:`GATE_MAP`.
    """
    nqubits = qibo_circ.nqubits
    circ_cls = CircuitMPS if use_mps else QuimbCircuit
    circ: CircuitMPS | QuimbCircuit = circ_cls(nqubits, **circuit_kwargs)

    for gate in qibo_circ.queue:
        gate_name: str | None = getattr(gate, "name", None)
        quimb_name: str | None = GATE_MAP.get(gate_name, None)  # type: ignore[arg-type]

        if quimb_name == "measure":
            continue
        if quimb_name is None:
            raise ValueError(
                f"Gate '{gate_name}' is not supported by the Quimb backend.\n"
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


def qasm_to_quimb(
    qasm_circuit: str,
    parameters: Sequence[float],
    *,
    use_mps: bool = True,
    **circuit_kwargs,
) -> CircuitMPS | QuimbCircuit:
    """
    Parse an OpenQASM 2.0 string, bind parameters, and return a Quimb circuit.

    This is a convenience wrapper that combines :func:`~csbench.converters.to_qibo.qasm_to_qibo_bound`
    and :func:`qibo_to_quimb` into a single call.

    Parameters
    ----------
    qasm_circuit:
        OpenQASM 2.0 string.
    parameters:
        Flat list of numeric gate parameters.
    use_mps:
        If ``True`` returns ``CircuitMPS``; otherwise a full ``Circuit``.
    **circuit_kwargs:
        Forwarded to the Quimb circuit constructor.

    Returns
    -------
    CircuitMPS or quimb.tensor.Circuit
    """
    from .to_qibo import qasm_to_qibo_bound

    qibo_circ = qasm_to_qibo_bound(qasm_circuit, parameters)
    return qibo_to_quimb(qibo_circ, use_mps=use_mps, **circuit_kwargs)
