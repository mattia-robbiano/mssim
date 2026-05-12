"""
csbench.engines.statevector_engine
====================================
Engine wrapper for Qiskit's ``StatevectorSimulator`` (exact simulation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from .abstract import BenchmarkEngine


@dataclass
class StatevectorEngine(BenchmarkEngine):
    """
    Exact statevector engine implemented with Qiskit.

    This engine accepts the standard QASM string + parameters convention.
    Because Qiskit's parametrised-circuit binding requires ``ParameterVector``
    objects, we re-parse the QASM and bind via positional order.

    Notes
    -----
    *  Fidelity is always ``1.0`` (exact engine, no approximation).
    *  Qubit ordering: Qiskit uses little-endian qubit ordering internally;
       the observable string is interpreted in the same big-endian (qubit-0
       first) convention as the other engines.  The conversion is handled
       automatically via ``SparsePauliOp``.
    """

    # No extra parameters needed for the exact engine.

    # ------------------------------------------------------------------

    def expectation_value(
        self,
        qasm_circuit: str,
        parameters: Sequence[float],
        observable: Sequence[str],
    ) -> tuple[float, float, None]:
        """
        Compute ⟨O⟩ with Qiskit statevector simulation.

        Parameters
        ----------
        qasm_circuit:
            OpenQASM 2.0 string.  Parametrised gates must use Qiskit's
            ``ParameterVector``-compatible syntax **or** be fully numeric
            (i.e. no symbolic parameters); if symbolic parameters are
            present they will be bound by positional index.
        parameters:
            Gate parameter values in gate-appearance order.
        observable:
            Pauli string, qubit-0 first (big-endian).

        Returns
        -------
        expval : float
        elapsed : float
        fidelity : None  (exact engine)
        """
        qc = QuantumCircuit.from_qasm_str(qasm_circuit)

        # Bind parameters if the circuit has any free symbols
        if qc.parameters:
            if len(parameters) != len(qc.parameters):
                raise ValueError(
                    f"Circuit has {len(qc.parameters)} free parameters but "
                    f"{len(parameters)} values were supplied."
                )
            param_dict = dict(zip(qc.parameters, parameters))
            qc = qc.assign_parameters(param_dict)

        # Remove any measurement gates — statevector sim cannot handle them
        qc.remove_final_measurements(inplace=True)

        # Build the Pauli observable.
        # SparsePauliOp uses *little-endian* (qubit-0 is rightmost character).
        # Our observable list is big-endian (index 0 → qubit 0), so reverse.
        pauli_str = "".join(reversed(observable))
        op = SparsePauliOp(pauli_str)

        # ----- Timed region -----------------------------------------------
        t0 = time.perf_counter()
        sv = Statevector(qc)
        expval = sv.expectation_value(op).real
        elapsed = time.perf_counter() - t0
        # ------------------------------------------------------------------

        return float(expval), elapsed, None

    @property
    def name(self) -> str:
        return "statevector"
