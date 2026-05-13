import time
from dataclasses import dataclass
from typing import Sequence

from mpstab.evolutors.hsmpo import HSMPO
from .abstract import BenchmarkEngine
from mssim.engines.converters import qasm_to_qibo_bound  # <--- Nuova importazione

@dataclass
class MPStabEngine(BenchmarkEngine):
    max_bond_dimension: int | None = None

    def expectation_value(
        self,
        qasm_circuit: str,
        parameters: Sequence[float],
        observable: Sequence[str],
    ) -> tuple[float, float, float | None]:
        
        # Usa il convertitore centralizzato
        circuit = qasm_to_qibo_bound(qasm_circuit, parameters)

        surrogate = HSMPO(circuit, max_bond_dimension=self.max_bond_dimension)

        t0 = time.perf_counter()
        expval = surrogate.expectation(observable=observable, return_fidelity=False)
        elapsed = time.perf_counter() - t0

        fidelity = surrogate.truncation_fidelity(replacement_probability=0.0)

        return float(expval), elapsed, float(fidelity)

    @property
    def name(self) -> str:
        return f"mpstab(chi={self.max_bond_dimension})"