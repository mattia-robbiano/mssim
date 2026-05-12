# mssim — Multi System Simulator

A quantum circuit classical simulation package supporting multiple simulation engines.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from mssim import build_circuit, build_engines, Executor

# Create a circuit
model = build_circuit("random_rx", n_qubits=6, depth=4)

# Set up engines
engines = build_engines(["tn", "sv"], max_bond_dimension=32)

# Run simulation
executor = Executor(n_runs=10, output_file="results.jsonl", output_fmt="jsonl")
batch_results = executor.run(model, engines)

# Print results
for br in batch_results:
    print(br.summary())
```

## Package Structure

```
mssim/
├── __init__.py
├── circuits/
│   ├── __init__.py
│   ├── model.py             # CircuitModel class
│   ├── library.py           # Circuit families (Ising, QAOA, Random…)
│   ├── to_qibo.py           # QASM → Qibo
│   ├── to_qiskit.py         # QASM → Qiskit
│   └── to_quimb.py          # Qibo → Quimb
├── engines/
│   ├── __init__.py
│   ├── abstract.py          # BenchmarkEngine base class
│   ├── mpstab_engine.py     # MPStab / HSMPO engine
│   ├── quimb_engine.py      # Quimb MPS engine
│   ├── statevector_engine.py# Qiskit statevector engine
│   └── registry.py          # Engine registry & factory
├── executor.py              # Executor: runs engines, collects statistics
└── output.py                # ResultRow dataclass + atomic HDF5/JSONL writer
```

## Supported Engines

- `"tn"` — Tensor Network (Quimb)
- `"mpstab"` — MPStab
- `"sv"` — Statevector (Qiskit)

├── main.py                      # Entry-point called by run.sh
└── README.md
```

## Quick start

```bash
# Install in editable mode (within your cluster venv)
pip install -e .

# Run locally with the example config
python main.py --settings config/example_settings.json \
               --n_qubits 6 --depth 3 --engine tn

# Submit to SLURM
sbatch scripts/run.sh config/example_settings.json
```

## Settings JSON fields

| Field | Type | Description |
|-------|------|-------------|
| `model.circuit` | str | Circuit family name (e.g. `"random_clifford"`) |
| `model.observable` | list[str] | Pauli string, e.g. `["Z","I","Z"]` |
| `model.kwargs` | dict | Circuit-specific extra parameters |
| `execution.engines` | list[str] | `["tn", "sv", "mpstab"]` or `["all"]` |
| `execution.n_runs` | int | Random-parameter samples per setting |
| `execution.max_bond_dimension` | int\|null | MPS bond dimension cap |
| `sweep.n_qubits` | list[int] | Qubit counts to sweep |
| `sweep.depth` | list[int] | Circuit depth values to sweep |
| `output.filename` | str | Output file path (`.jsonl` or `.h5`) |
| `output.format` | str | `"jsonl"` or `"hdf5"` |
