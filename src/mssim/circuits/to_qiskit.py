"""
csbench.converters.to_qiskit
==============================
Utilities for loading OpenQASM 2.0 circuits into Qiskit and binding
parameters.

Qiskit's ``QuantumCircuit.from_qasm_str()`` handles standard QASM natively.
Parametrised circuits must use Qiskit's ``ParameterVector`` or
``Parameter`` objects — circuits generated with numeric literals are
already bound and need no extra step.
"""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def qasm_to_qiskit(qasm_circuit: str) -> QuantumCircuit:
    """
    Parse an OpenQASM 2.0 string into a Qiskit ``QuantumCircuit``.

    Parameters
    ----------
    qasm_circuit:
        Valid OpenQASM 2.0 string.

    Returns
    -------
    QuantumCircuit
    """
    return QuantumCircuit.from_qasm_str(qasm_circuit)


def bind_parameters_qiskit(
    qc: QuantumCircuit,
    parameters: Sequence[float],
) -> QuantumCircuit:
    """
    Bind free ``Parameter`` / ``ParameterVector`` symbols to numeric values.

    Parameters are matched to ``qc.parameters`` in *sorted* order (Qiskit's
    default ordering — alphabetical by parameter name).  If the circuit has
    no free parameters the circuit is returned unchanged.

    Parameters
    ----------
    qc:
        Qiskit circuit, possibly with free parameters.
    parameters:
        Numeric values in the same sorted order as ``qc.parameters``.

    Returns
    -------
    QuantumCircuit
        A new (bound) circuit; the original is not mutated.

    Raises
    ------
    ValueError
        If the number of supplied values does not match the number of free
        parameters.
    """
    if not qc.parameters:
        return qc

    if len(parameters) != len(qc.parameters):
        raise ValueError(
            f"Circuit has {len(qc.parameters)} free parameters but "
            f"{len(parameters)} values were supplied."
        )

    param_dict = dict(zip(qc.parameters, parameters))
    return qc.assign_parameters(param_dict)


def qasm_to_qiskit_bound(
    qasm_circuit: str,
    parameters: Sequence[float],
) -> QuantumCircuit:
    """
    Parse QASM and immediately bind parameters.

    Parameters
    ----------
    qasm_circuit:
        OpenQASM 2.0 string.
    parameters:
        Flat numeric parameter list.

    Returns
    -------
    QuantumCircuit
        Fully bound Qiskit circuit, with measurements removed.
    """
    qc = qasm_to_qiskit(qasm_circuit)
    qc = bind_parameters_qiskit(qc, parameters)
    qc.remove_final_measurements(inplace=True)
    return qc


def build_parametrised_qiskit_circuit(
    qasm_circuit: str,
    param_name: str = "θ",
) -> tuple[QuantumCircuit, ParameterVector]:
    """
    Build a Qiskit circuit where every gate parameter is replaced by a
    symbolic ``ParameterVector`` entry.

    Useful when you want to bind many different parameter sets to the same
    compiled circuit object (avoids re-parsing QASM each time).

    .. warning::
        This function modifies the circuit structure and assumes all numeric
        gate parameters should become free symbols.  Use with care on
        circuits that mix fixed and trainable parameters.

    Parameters
    ----------
    qasm_circuit:
        OpenQASM 2.0 string.
    param_name:
        Base name for the ``ParameterVector``.

    Returns
    -------
    (QuantumCircuit, ParameterVector)
        The symbolic circuit and the vector whose entries correspond to the
        free parameters (in gate-appearance order).
    """
    qc = qasm_to_qiskit(qasm_circuit)

    # Count existing free parameters; if none, the circuit is already fully
    # numeric — nothing to do.
    if not qc.parameters:
        return qc, ParameterVector(param_name, 0)

    pv = ParameterVector(param_name, len(qc.parameters))
    bound = qc.assign_parameters(dict(zip(qc.parameters, pv)))
    return bound, pv
