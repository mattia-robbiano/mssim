import time
from dataclasses import dataclass
from typing import Sequence

from qiskit.quantum_info import Statevector, SparsePauliOp
from .abstract import BenchmarkEngine
from mssim.engines.converters import qasm_to_qiskit_bound

@dataclass
class StatevectorEngine(BenchmarkEngine):

    def expectation_value(
        self,
        qasm_circuit: str,
        parameters: Sequence[float],
        observable: str,
    ) -> tuple[float, float, None]:
        
        qc = qasm_to_qiskit_bound(qasm_circuit, parameters)

        # pauli_str = "".join(reversed(observable)).upper()
        op = SparsePauliOp(observable.upper())

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