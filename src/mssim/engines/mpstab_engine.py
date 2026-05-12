"""
csbench.engines.mpstab_engine
==============================
Engine wrapper for the MPStab / HSMPO framework (Qibo circuits).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from qibo import Circuit

from mpstab.evolutors.hsmpo import HSMPO

from .abstract import BenchmarkEngine


@dataclass
class MPStabEngine(BenchmarkEngine):
    """
    Simulation engine based on the MPStab Hybrid Stabiliser MPO (HSMPO).

    Parameters
    ----------
    max_bond_dimension:
        Maximum bond dimension for the MPO truncation.
        ``None`` means no truncation (exact up to floating-point).
    """

    max_bond_dimension: int | None = None

    # ------------------------------------------------------------------

    def expectation_value(
        self,
        qasm_circuit: str,
        parameters: Sequence[float],
        observable: Sequence[str],
    ) -> tuple[float, float, float | None]:
        """
        Compute ⟨O⟩ via the HSMPO surrogate.

        Returns
        -------
        expval : float
        elapsed : float   (seconds, HSMPO contraction only)
        fidelity : float  (truncation fidelity at zero replacement probability)
        """
        circuit = Circuit.from_qasm(qasm_circuit)
        circuit.set_parameters(list(parameters))

        surrogate = HSMPO(circuit, max_bond_dimension=self.max_bond_dimension)

        t0 = time.perf_counter()
        expval = surrogate.expectation(observable=observable, return_fidelity=False)
        elapsed = time.perf_counter() - t0

        fidelity = surrogate.truncation_fidelity(replacement_probability=0.0)

        return float(expval), elapsed, float(fidelity)

    @property
    def name(self) -> str:
        return f"mpstab(chi={self.max_bond_dimension})"
