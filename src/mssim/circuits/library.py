"""
circuits.library
=========================
Factory functions for named circuit families.

Each function returns a :class:`~csbench.circuits.model.CircuitModel` that is
ready to be passed to the executor.

Supported families
------------------
- ``random_clifford``  — random single-/two-qubit Clifford brick-layer circuit
- ``random_rx``        — brick-layer of random Rx rotations + CNOT entangling
- ``ising``            — transverse-field Ising Trotter circuit
- ``qaoa``             — QAOA circuit on a 1-D ring with ZZ + X mixer
- ``hardware_efficient`` — hardware-efficient ansatz (Ry + CZ)

All circuits are expressed in OpenQASM 2.0 with *numeric* gate parameters
(the QASM string is valid stand-alone) and expose a ``parameter_sampler``
that draws a new parameter vector from the appropriate distribution.

Adding a new family
-------------------
1.  Write a function ``build_<name>(n_qubits, depth, **kwargs) -> CircuitModel``.
2.  Register it in :data:`CIRCUIT_REGISTRY` at the bottom of this file.
3.  Done — the factory :func:`build_circuit` will pick it up automatically.
"""

from __future__ import annotations

import textwrap
from typing import Any, Callable

import numpy as np

from .model import CircuitModel


# ---------------------------------------------------------------------------
# QASM helpers
# ---------------------------------------------------------------------------


def _qasm_header(n_qubits: int) -> str:
    return textwrap.dedent(f"""\
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[{n_qubits}];
        creg c[{n_qubits}];
    """)


def _uniform_sampler(n: int) -> Callable[[], list[float]]:
    """Return a sampler of *n* angles drawn uniformly from [0, 2π)."""
    return lambda: list(np.random.uniform(0.0, 2 * np.pi, size=n))


# ---------------------------------------------------------------------------
# Random Rx+CNOT brick-layer
# ---------------------------------------------------------------------------


def build_random_rx(
    n_qubits: int,
    depth: int,
    observable: list[str] | None = None,
    **kwargs: Any,
) -> CircuitModel:
    """
    Brick-layer circuit: alternating layers of random Rx rotations and a
    staircase of CNOT gates.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    depth:
        Number of (Rx-layer + CNOT-layer) repetitions.
    observable:
        Pauli observable.  Defaults to all-Z.
    """
    if observable is None:
        observable = ["Z"] * n_qubits

    lines = [_qasm_header(n_qubits)]
    n_params = n_qubits * depth  # one Rx per qubit per depth layer
    angles = np.zeros(n_params)  # placeholder; will be replaced each run

    param_idx = 0
    for _ in range(depth):
        for q in range(n_qubits):
            lines.append(f"rx({angles[param_idx]:.6f}) q[{q}];")
            param_idx += 1
        # Nearest-neighbour CNOT staircase
        for q in range(0, n_qubits - 1, 2):
            lines.append(f"cx q[{q}], q[{q + 1}];")
        for q in range(1, n_qubits - 1, 2):
            lines.append(f"cx q[{q}], q[{q + 1}];")

    return CircuitModel(
        name="random_rx",
        n_qubits=n_qubits,
        depth=depth,
        qasm="\n".join(lines),
        observable=observable,
        n_params=n_params,
        parameter_sampler=_uniform_sampler(n_params),
        metadata={"entangler": "cx"},
    )


# ---------------------------------------------------------------------------
# Transverse-field Ising Trotter circuit
# ---------------------------------------------------------------------------


def build_ising(
    n_qubits: int,
    depth: int,
    J: float = 1.0,
    h: float = 0.5,
    dt: float = 0.1,
    observable: list[str] | None = None,
    **kwargs: Any,
) -> CircuitModel:
    """
    First-order Trotter circuit for the transverse-field Ising model
    H = -J Σ ZZ - h Σ X.

    Parameters
    ----------
    J, h, dt:
        Coupling, field, and Trotter step.  These are *fixed* (not random).
    depth:
        Number of Trotter steps.
    observable:
        Defaults to Z on qubit 0.
    """
    if observable is None:
        observable = ["Z" if i == 0 else "I" for i in range(n_qubits)]

    lines = [_qasm_header(n_qubits)]
    # Initial |+⟩^n state
    for q in range(n_qubits):
        lines.append(f"h q[{q}];")

    zz_angle = 2 * J * dt
    x_angle = 2 * h * dt

    for _ in range(depth):
        # ZZ interactions (even bonds)
        for q in range(0, n_qubits - 1, 2):
            lines.append(f"cx q[{q}], q[{q + 1}];")
            lines.append(f"rz({zz_angle:.6f}) q[{q + 1}];")
            lines.append(f"cx q[{q}], q[{q + 1}];")
        # ZZ interactions (odd bonds)
        for q in range(1, n_qubits - 1, 2):
            lines.append(f"cx q[{q}], q[{q + 1}];")
            lines.append(f"rz({zz_angle:.6f}) q[{q + 1}];")
            lines.append(f"cx q[{q}], q[{q + 1}];")
        # X field
        for q in range(n_qubits):
            lines.append(f"rx({x_angle:.6f}) q[{q}];")

    # No free parameters — return a constant sampler
    return CircuitModel(
        name="ising",
        n_qubits=n_qubits,
        depth=depth,
        qasm="\n".join(lines),
        observable=observable,
        n_params=0,
        parameter_sampler=lambda: [],
        metadata={"J": J, "h": h, "dt": dt},
    )


# ---------------------------------------------------------------------------
# Hardware-efficient ansatz (Ry + CZ)
# ---------------------------------------------------------------------------


def build_hardware_efficient(
    n_qubits: int,
    depth: int,
    observable: list[str] | None = None,
    **kwargs: Any,
) -> CircuitModel:
    """
    Hardware-efficient ansatz: layers of Ry rotations interleaved with
    CZ entanglers (alternating even/odd bonds).

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    depth:
        Number of Ry+CZ layers.
    observable:
        Defaults to all-Z.
    """
    if observable is None:
        observable = ["Z"] * n_qubits

    lines = [_qasm_header(n_qubits)]
    n_params = n_qubits * (depth + 1)  # +1 for the final Ry layer
    angles = np.zeros(n_params)

    param_idx = 0
    for d in range(depth):
        for q in range(n_qubits):
            lines.append(f"ry({angles[param_idx]:.6f}) q[{q}];")
            param_idx += 1
        offset = d % 2
        for q in range(offset, n_qubits - 1, 2):
            lines.append(f"cz q[{q}], q[{q + 1}];")
    # Final rotation layer
    for q in range(n_qubits):
        lines.append(f"ry({angles[param_idx]:.6f}) q[{q}];")
        param_idx += 1

    return CircuitModel(
        name="hardware_efficient",
        n_qubits=n_qubits,
        depth=depth,
        qasm="\n".join(lines),
        observable=observable,
        n_params=n_params,
        parameter_sampler=_uniform_sampler(n_params),
        metadata={"entangler": "cz"},
    )


# ---------------------------------------------------------------------------
# QAOA on 1-D ring (ZZ cost + X mixer)
# ---------------------------------------------------------------------------


def build_qaoa(
    n_qubits: int,
    depth: int,
    observable: list[str] | None = None,
    **kwargs: Any,
) -> CircuitModel:
    """
    QAOA ansatz for the 1-D ring Max-Cut problem.

    ``depth`` QAOA layers, each with one ``γ`` (cost) and one ``β`` (mixer)
    parameter.  Total free parameters: ``2 * depth``.

    Parameters
    ----------
    n_qubits:
        Number of qubits / vertices.
    depth:
        Number of QAOA layers (``p`` in the standard notation).
    observable:
        Defaults to ZZ on the first edge.
    """
    if observable is None:
        obs = ["I"] * n_qubits
        obs[0] = "Z"
        if n_qubits > 1:
            obs[1] = "Z"
        observable = obs

    lines = [_qasm_header(n_qubits)]
    n_params = 2 * depth  # [γ_0, β_0, γ_1, β_1, …]
    angles = np.zeros(n_params)

    # Initial |+⟩^n
    for q in range(n_qubits):
        lines.append(f"h q[{q}];")

    for p in range(depth):
        gamma = angles[2 * p]
        beta = angles[2 * p + 1]
        # Cost layer: ZZ on ring edges
        for q in range(n_qubits):
            nxt = (q + 1) % n_qubits
            lines.append(f"cx q[{q}], q[{nxt}];")
            lines.append(f"rz({2 * gamma:.6f}) q[{nxt}];")
            lines.append(f"cx q[{q}], q[{nxt}];")
        # Mixer layer: X rotations
        for q in range(n_qubits):
            lines.append(f"rx({2 * beta:.6f}) q[{q}];")

    return CircuitModel(
        name="qaoa",
        n_qubits=n_qubits,
        depth=depth,
        qasm="\n".join(lines),
        observable=observable,
        n_params=n_params,
        parameter_sampler=_uniform_sampler(n_params),
        metadata={"topology": "ring", "problem": "max_cut"},
    )


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

CIRCUIT_REGISTRY: dict[str, Any] = {
    "random_rx":          build_random_rx,
    "ising":              build_ising,
    "hardware_efficient": build_hardware_efficient,
    "qaoa":               build_qaoa,
}


def build_circuit(
    name: str,
    n_qubits: int,
    depth: int,
    **kwargs: Any,
) -> CircuitModel:
    """
    Instantiate a :class:`~csbench.circuits.model.CircuitModel` by family name.

    Parameters
    ----------
    name:
        Circuit family name (must be a key in :data:`CIRCUIT_REGISTRY`).
    n_qubits:
        Number of qubits.
    depth:
        Circuit depth (semantics depend on the family).
    **kwargs:
        Family-specific extra arguments (e.g. ``J``, ``h`` for ``"ising"``).

    Returns
    -------
    CircuitModel

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in CIRCUIT_REGISTRY:
        raise KeyError(
            f"Unknown circuit family '{name}'. "
            f"Available: {sorted(CIRCUIT_REGISTRY.keys())}"
        )
    return CIRCUIT_REGISTRY[name](n_qubits=n_qubits, depth=depth, **kwargs)
