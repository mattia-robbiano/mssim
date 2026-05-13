import time
from dataclasses import dataclass
from typing import Sequence
from quimb import pauli

from .abstract import BenchmarkEngine
from mssim.engines.converters import qasm_to_quimb

@dataclass
class QuimbEngine(BenchmarkEngine):
    max_bond_dimension: int | None = None
    contraction_optimizer: str = "auto-hq"

    def expectation_value(
        self,
        qasm_circuit: str,
        parameters: Sequence[float],
        observable: Sequence[str],
    ) -> tuple[float, float, float | None]:
        
        # ----- Timed region -----------------------------------------------
        t0 = time.perf_counter()

        # Usa il convertitore centralizzato
        mps_circ = qasm_to_quimb(
            qasm_circuit, 
            parameters, 
            max_bond=self.max_bond_dimension
        )
        
        psi_ket = mps_circ.psi
        norm = psi_ket.norm(squared=True).real

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

        # Stima della fedeltà (richiede una nuova istanza)
        mps_for_fidelity = qasm_to_quimb(
            qasm_circuit, 
            parameters, 
            max_bond=self.max_bond_dimension
        )
        fidelity = mps_for_fidelity.fidelity_estimate()

        return float(expval), elapsed, float(fidelity)

    @property
    def name(self) -> str:
        return f"quimb_mps(chi={self.max_bond_dimension})"