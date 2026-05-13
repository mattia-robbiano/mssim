"""
tests/test_basic.py
====================
Lightweight unit tests that do NOT require any quantum simulation backend
(qibo, quimb, qiskit, mpstab) so they can run in CI with minimal deps.

Run with:  pytest tests/
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# CircuitModel & library
# ---------------------------------------------------------------------------


def test_circuit_model_observable_length_check():
    from mssim.circuits.model import CircuitModel

    with pytest.raises(ValueError, match="Observable length"):
        CircuitModel(
            name="test",
            n_qubits=4,
            depth=1,
            qasm="OPENQASM 2.0;",
            observable=["Z", "Z"],   # too short
            n_params=0,
        )


def test_build_circuit_random_rx():
    from mssim.circuits.library import build_circuit

    model = build_circuit("random_rx", n_qubits=4, depth=2)
    assert model.n_qubits == 4
    assert model.depth == 2
    assert len(model.observable) == 4
    assert model.n_params == 4 * 2  # n_qubits × depth

    params = model.sample_parameters()
    assert len(params) == model.n_params
    assert all(0.0 <= p < 2 * np.pi for p in params)


def test_build_circuit_ising():
    from mssim.circuits.library import build_circuit

    model = build_circuit("ising", n_qubits=6, depth=3, J=1.0, h=0.5, dt=0.05)
    assert model.n_params == 0
    assert model.sample_parameters() == []
    assert "OPENQASM" in model.qasm


def test_build_circuit_qaoa():
    from mssim.circuits.library import build_circuit

    model = build_circuit("qaoa", n_qubits=5, depth=2)
    assert model.n_params == 2 * 2  # 2 * depth


def test_build_circuit_hardware_efficient():
    from mssim.circuits.library import build_circuit

    model = build_circuit("hardware_efficient", n_qubits=4, depth=3)
    assert model.n_params == 4 * (3 + 1)


def test_unknown_circuit():
    from mssim.circuits.library import build_circuit

    with pytest.raises(KeyError, match="not_a_circuit"):
        build_circuit("not_a_circuit", n_qubits=4, depth=2)


# ---------------------------------------------------------------------------
# Output: ResultRow serialisation
# ---------------------------------------------------------------------------


def _make_row(**overrides):
    from mssim.output import ResultRow

    defaults = dict(
        run_id=0,
        engine="test_engine",
        circuit="random_rx",
        n_qubits=4,
        depth=2,
        n_params=8,
        parameters=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        observable=["Z", "I", "Z", "I"],
        expectation_value=0.5,
        elapsed_seconds=0.01,
        fidelity=0.99,
    )
    defaults.update(overrides)
    return ResultRow(**defaults)


def test_result_row_to_dict():
    row = _make_row()
    d = row.to_dict()
    assert d["engine"] == "test_engine"
    assert d["expectation_value"] == 0.5
    assert isinstance(d["parameters"], list)


def test_result_row_fidelity_none():
    row = _make_row(fidelity=None)
    d = row.to_dict()
    assert d["fidelity"] is None


def test_save_and_load_jsonl():
    from mssim.output import save_result, load_results_jsonl

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        fname = f.name

    try:
        for i in range(5):
            save_result(fname, _make_row(run_id=i, expectation_value=float(i)), fmt="jsonl")

        rows = load_results_jsonl(fname)
        assert len(rows) == 5
        assert [r["run_id"] for r in rows] == list(range(5))
    finally:
        os.unlink(fname)


def test_batch_result_summary():
    from mssim.output import BatchResult

    rows = [_make_row(run_id=i, expectation_value=float(i) * 0.1) for i in range(10)]
    br = BatchResult(
        engine="test",
        circuit="random_rx",
        n_qubits=4,
        depth=2,
        n_runs=10,
        rows=rows,
    )
    s = br.summary()
    assert s["n_runs"] == 10
    assert "expval_mean" in s
    assert "fidelity_mean" in s


# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------


def test_unknown_engine():
    from mssim.engines.library import build_engines

    with pytest.raises(KeyError, match="not_an_engine"):
        build_engines(["not_an_engine"])


def test_build_engines_empty_list():
    from mssim.engines.library import build_engines

    engines = build_engines([])
    assert engines == []
