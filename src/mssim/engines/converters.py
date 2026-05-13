"""
mssim.converters
================
Unified utilities for converting OpenQASM 2.0 circuits into Qibo, 
Quimb (Tensor Networks), and Qiskit formats.
"""

from __future__ import annotations
from typing import Any, Callable, Sequence

from qibo import Circuit
from qibo.gates.abstract import ParametrizedGate
from quimb.tensor import CircuitMPS
from quimb.tensor import Circuit as QuimbCircuit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def qasm_to_qibo(qasm_circuit: str) -> Circuit:
    """Parse an OpenQASM 2.0 string into a Qibo Circuit."""
    return Circuit.from_qasm(qasm_circuit)

def bind_parameters_qibo(circuit: Circuit, parameters: Sequence[float]) -> Circuit:
    """
    Bind parameters to a Qibo circuit. 
    Fixes the ValueError when parameters is empty but QASM contains numeric literals.
    """
    # If parameters is empty, assume the circuit is already 
    # functionally bound via QASM numeric literals.
    if parameters and len(parameters) > 0:
        circuit.set_parameters(list(parameters))
    return circuit

def qasm_to_qibo_bound(qasm_circuit: str, parameters: Sequence[float]) -> Circuit:
    """Parse QASM and immediately bind parameters."""
    circ = qasm_to_qibo(qasm_circuit)
    return bind_parameters_qibo(circ, parameters)


GATE_MAP: dict[str, str] = {
    "h": "H", "x": "X", "y": "Y", "z": "Z", "s": "S", "t": "T",
    "rx": "RX", "ry": "RY", "rz": "RZ", "u3": "U3",
    "cx": "CX", "cnot": "CNOT", "cy": "CY", "cz": "CZ",
    "iswap": "ISWAP", "swap": "SWAP", "ccx": "CCX", "ccy": "CCY",
    "ccz": "CCZ", "toffoli": "TOFFOLI", "cswap": "CSWAP",
    "fredkin": "FREDKIN", "fsim": "fsim", "measure": "measure",
}

def qibo_to_quimb(
    qibo_circ: Circuit,
    *,
    use_mps: bool = True,
    **circuit_kwargs,
) -> CircuitMPS | QuimbCircuit:
    """Convert a bound Qibo Circuit to a Quimb TN circuit."""
    nqubits = qibo_circ.nqubits
    circ_cls = CircuitMPS if use_mps else QuimbCircuit
    circ: CircuitMPS | QuimbCircuit = circ_cls(nqubits, **circuit_kwargs)

    for gate in qibo_circ.queue:
        gate_name: str | None = getattr(gate, "name", None)
        quimb_name: str | None = GATE_MAP.get(gate_name, None)

        if quimb_name == "measure" or quimb_name is None:
            continue

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
    """Convenience wrapper: QASM -> Qibo (Bound) -> Quimb."""
    qibo_circ = qasm_to_qibo_bound(qasm_circuit, parameters)
    return qibo_to_quimb(qibo_circ, use_mps=use_mps, **circuit_kwargs)


def qasm_to_qiskit(qasm_circuit: str) -> QuantumCircuit:
    """Parse QASM into a Qiskit QuantumCircuit."""
    return QuantumCircuit.from_qasm_str(qasm_circuit)

def bind_parameters_qiskit(
    qc: QuantumCircuit,
    parameters: Sequence[float],
) -> QuantumCircuit:
    """Bind Parameter symbols to numeric values."""
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
    """Parse QASM and immediately bind parameters, removing measurements."""
    qc = qasm_to_qiskit(qasm_circuit)
    qc = bind_parameters_qiskit(qc, parameters)
    qc.remove_final_measurements(inplace=True)
    return qc

def build_parametrised_qiskit_circuit(
    qasm_circuit: str,
    param_name: str = "θ",
) -> tuple[QuantumCircuit, ParameterVector]:
    """Convert numeric QASM gates into symbolic ParameterVector entries."""
    qc = qasm_to_qiskit(qasm_circuit)
    if not qc.parameters:
        return qc, ParameterVector(param_name, 0)

    pv = ParameterVector(param_name, len(qc.parameters))
    bound = qc.assign_parameters(dict(zip(qc.parameters, pv)))
    return bound, pv