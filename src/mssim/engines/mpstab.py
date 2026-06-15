import time
from dataclasses import dataclass

from qibo import Circuit
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import Z
from mpstab import HSMPO
from mssim.engines.abstract import BenchmarkEngine


@dataclass
class MPStabEngine(BenchmarkEngine):
    max_bond_dimension: int | None = None

    def expectation_value(
        self,
        qasm_circuit: str,
        observable: str,
    ) -> tuple[float, float, float]:

        circuit = Circuit.from_qasm(qasm_circuit)
        mpstab_hsmpo = HSMPO(circuit, max_bond_dimension=self.max_bond_dimension)

        if observable.upper() == "MAGNETIZATION":
            num_qubits = circuit.nqubits
            observable = SymbolicHamiltonian(sum(Z(i) for i in range(num_qubits)))
        else:
            # Match the qubit-ordering convention of qiskit
            observable = observable[::-1]

        t0 = time.perf_counter()
        expval = mpstab_hsmpo.expectation(observable=observable)
        elapsed = time.perf_counter() - t0

        fidelity = mpstab_hsmpo.truncation_fidelity(replacement_probability=0.0,)

        return float(expval), elapsed, float(fidelity)


    @property
    def name(self) -> str:
        return f"mpstab"

        
