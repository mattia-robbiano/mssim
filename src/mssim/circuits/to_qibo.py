"""
csbench.converters.to_qibo
===========================
Utilities for loading / converting circuits into Qibo's native format.

Qibo can parse OpenQASM 2.0 directly via ``Circuit.from_qasm()``, so
this module is intentionally thin — it provides a uniform interface and
a parameter-binding helper consistent with the rest of the converters.
"""

from __future__ import annotations

from typing import Sequence

from qibo import Circuit


def qasm_to_qibo(qasm_circuit: str) -> Circuit:
    """
    Parse an OpenQASM 2.0 string into a Qibo ``Circuit``.

    Parameters
    ----------
    qasm_circuit:
        Valid OpenQASM 2.0 string.  Parameterised gates should use numeric
        literals or Qibo-compatible symbolic expressions.

    Returns
    -------
    Circuit
        Unbound Qibo circuit (parameters not yet set).
    """
    return Circuit.from_qasm(qasm_circuit)


def bind_parameters_qibo(circuit: Circuit, parameters: Sequence[float]) -> Circuit:
    """
    Bind a flat list of numeric parameters to a Qibo circuit *in-place*.

    Parameters are assigned in the order that parametrised gates appear
    in ``circuit.queue``.

    Parameters
    ----------
    circuit:
        A Qibo circuit whose parametrised gates will be updated.
    parameters:
        Flat sequence of floats.

    Returns
    -------
    Circuit
        The *same* circuit object (mutation is in-place) for chaining.
    """
    circuit.set_parameters(list(parameters))
    return circuit


def qasm_to_qibo_bound(qasm_circuit: str, parameters: Sequence[float]) -> Circuit:
    """
    Convenience wrapper: parse QASM and immediately bind parameters.

    Parameters
    ----------
    qasm_circuit:
        OpenQASM 2.0 string.
    parameters:
        Flat parameter list.

    Returns
    -------
    Circuit
        Fully bound Qibo circuit ready for simulation.
    """
    circ = qasm_to_qibo(qasm_circuit)
    return bind_parameters_qibo(circ, parameters)
