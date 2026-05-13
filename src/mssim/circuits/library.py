"""
circuits.library
=========================
Factory functions for named circuit families.

Each function returns a :class:`~mssim.circuits.model.CircuitModel` that is
ready to be passed to the executor.

Supported families
------------------
- ``kicked_ising``
- ``ising``
- ``hardware_efficient``

All circuits are expressed in OpenQASM 2.0 with *numeric* gate parameters
(the QASM string is valid stand-alone) and expose a ``parameter_sampler``
that draws a new parameter vector from the appropriate distribution.

Adding a new family
-------------------
1.  Write a function ``build_<name>(n_qubits, depth, **kwargs) -> CircuitModel``.
2.  Register it in :data:`CIRCUIT_REGISTRY` at the bottom of this file.
3.  Done — the factory :func:`build_circuit` should pick it up automatically.
"""

from __future__ import annotations
import textwrap
from typing import Any, Callable
import numpy as np
from mssim.circuits.model import CircuitModel


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


def build_kicked_ising(
    n_qubits: int, 
    depth: int, 
    J: float = 1.0, 
    h: float = 0.5, 
    b: float = 0.5,
    observable: list[str] | None = None,
) -> CircuitModel:
    """
    Generates a CircuitModel for the forward-time Kicked Ising Floquet circuit.
    
    The period consists of an even and odd layer of symmetric U blocks.
    Parameters (J, h, b) are fixed.

    Parameters
    ----------
    n_qubits:
        Total number of qubits in the 1D chain.
    depth:
        Number of Floquet periods (t).
    J, h, b:
        Theoretical Ising coupling, longitudinal field, and transverse kick strengths.
    observable:
        Defaults to all-Z.
    """
    if n_qubits < 2:
        raise ValueError("The Floquet circuit requires at least 2 qubits.")
    
    if observable is None:
        observable = ["Z"] * n_qubits

    lines = [_qasm_header(n_qubits)]
    
    # Map physical parameters to QASM rotation angles
    theta_J = 2.0 * J
    theta_h = 2.0 * h
    theta_b = 2.0 * b
    
    def apply_u_block(q_n: int, q_n1: int):
        """Constructs the symmetric U_{n, n+1} block."""
        # 1. RZ(2h) on bottom wire
        lines.append(f"rz({theta_h:.6f}) q[{q_n}];")
        
        # 2. RZZ(2J) decomposed
        lines.append(f"cx q[{q_n}], q[{q_n1}];")
        lines.append(f"rz({theta_J:.6f}) q[{q_n1}];")
        lines.append(f"cx q[{q_n}], q[{q_n1}];")
        
        # 3. RX(2b) on both wires
        lines.append(f"rx({theta_b:.6f}) q[{q_n}];")
        lines.append(f"rx({theta_b:.6f}) q[{q_n1}];")
        
        # 4. RZZ(2J) decomposed
        lines.append(f"cx q[{q_n}], q[{q_n1}];")
        lines.append(f"rz({theta_J:.6f}) q[{q_n1}];")
        lines.append(f"cx q[{q_n}], q[{q_n1}];")
        
        # 5. RZ(2h) on bottom wire
        lines.append(f"rz({theta_h:.6f}) q[{q_n}];")

    # Construct the Floquet stroboscopic evolution
    for layer in range(depth):
        lines.append(f"// --- Floquet Period {layer + 1} ---")
        
        # Even sub-layer
        for i in range(0, n_qubits - 1, 2):
            apply_u_block(i, i + 1)
            
        # Odd sub-layer
        for i in range(1, n_qubits - 1, 2):
            apply_u_block(i, i + 1)

    return CircuitModel(
        name="kicked_ising",
        n_qubits=n_qubits,
        depth=depth,
        qasm="\n".join(lines),
        observable=observable,
        n_params=0,                         # no random rotations to parametrize to collect statistics
        parameter_sampler=lambda: [],       # we are not sampling, nothing random in this circut
        metadata={"J": J, "h": h, "b": b},
    )

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


CIRCUIT_REGISTRY: dict[str, Any] = {
    "kicked-ising":       build_kicked_ising,
    "ising":              build_ising,
    "hardware_efficient": build_hardware_efficient,
}

def build_circuit(
    name: str,
    n_qubits: int,
    depth: int,
    **kwargs: Any,
) -> CircuitModel:
    """
    Instantiate a :class:`~mssim.circuits.model.CircuitModel` by family name.

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
