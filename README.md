# TrOMA

TrOMA is a Python library for optimization of black-box functions with binary inputs

B. Chevalier, S. Yamaguchi, W. Roga, M. Takeoka, A Compressive Sensing Inspired Monte-Carlo Method for Combinatorial Optimization, arXiv:2510.24755 (2025).

The method is based on building a surrogate model from sketches through the process we call MCCO modeling. The MCCO cost function that is built can be converted into an Ising Hamiltonian to optimize the function with a quantum process.

Alternatively, the library can be used to efficiently deal with large size compressive sensing problems and benefit from quantum computers as explained in:

B.Chevalier, W. Roga, M.takeoka, Compressed sensing enhanced by a quantum approximate optimization algorithm, Phys. Rev. A 110, 062410 (2024).

The API is centered around:

- explicit or constraint-based sketch construction,
- `matching_pursuit` for sparse reconstruction,
- classical and quantum optimizers (including QAOA with a Qiskit backend),
- a complete MCCO workflow through `solve_via_mcco` and `embedding_and_solve_via_mcco`.

The main entry point is the `troma` module.

See the doc at https://baptistechev.github.io/TrOMA/

## Installation

```bash
pip install troma
```

Then in Python:

```python
import troma
print(troma.__version__)
```

## Main API

The most useful objects are exposed directly at package level:

```python
import troma
from troma import (
    ConstraintSketch,
    ExplicitSketch,
    get_optimizer,
    bind_optimizer,
    matching_pursuit,
    mcco_modeling,
    solve_via_mcco,
    embedding_and_solve_via_mcco,
    spectrum_embedding,
    spectrum_restriction,
    reverse_spectrum_restriction,
)
from troma import data_structure as ds
```

## Quick example: Compressive Sensing abstract reconstruction

```python
import numpy as np

from troma import ConstraintSketch, matching_pursuit
from troma import data_structure as ds

number_spins = 7
interaction_size = 4

spectrum_bin = [
    [0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 1],
]
spectrum_val = [-0.5, 1.8, -0.3, 1.5]

constraints = ConstraintSketch.build_nearest_neighbors_sketch(
    number_spins,
    interaction_size,
    2,
)
y = ConstraintSketch.compute_marginal((spectrum_bin, spectrum_val), constraints)

solution = matching_pursuit(
    "abstract",
    y,
    constraints,
    number_spins,
    iteration_number=2,
    interaction_size=interaction_size,
)

print(solution)
```

## Choosing an optimizer for matching pursuit

The recommended path is to use the common interface:

```python
from troma import get_optimizer, bind_optimizer

opti = get_optimizer("dual_annealing")
# or, for QAOA with default local simulator:
opti = bind_optimizer("qaoa", number_shots=4096)
```

Available optimizers:

| Name | Type | Notes |
|---|---|---|
| `spin_chain_nn_max` | Classical | Spin-chain nearest-neighbor heuristic |
| `brute_force_max` | Classical | Exhaustive search |
| `dual_annealing` | Classical | SciPy dual annealing |
| `simulated_annealing` | Classical | Neal simulated annealing |
| `digital_annealing` | Quantum-inspired | D-Wave Neal QUBO solver |
| `qaoa` | Quantum | QAOA via Qiskit (see below) |

## Quantum optimizer: QAOA with Qiskit

The `qaoa` optimizer uses [Qiskit](https://qiskit.org/) and supports both local simulation and real IBM quantum hardware.

### Local simulation (default)

By default, QAOA runs on `qiskit_aer.AerSimulator`:

```python
from troma import bind_optimizer, matching_pursuit, ConstraintSketch

opti = bind_optimizer(
    "qaoa",
    number_shots=4096,
    number_layers=4,
    method="COBYLA",
    optimizer_options={"maxiter": 100},
)

solution = matching_pursuit(
    "abstract", y, constraints, number_spins,
    iteration_number=2,
    interaction_size=interaction_size,
    optimizer=opti,
)
```

### Custom Qiskit sampler (simulation or hardware)

Pass any `SamplerV2`-compatible sampler via the `sampler` keyword. This lets you target different Aer backends or real IBM hardware:

```python
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2
from troma import bind_optimizer

backend = AerSimulator()
sampler = SamplerV2(mode=backend)

opti = bind_optimizer("qaoa", sampler=sampler, number_shots=4096, number_layers=4)
```

### Running on IBM quantum hardware

Connect to the IBM Quantum platform via `QiskitRuntimeService` and pass the backend sampler directly:

```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from troma import bind_optimizer, matching_pursuit

service = QiskitRuntimeService()
backend = service.backend("ibm_marrakesh")
sampler = SamplerV2(mode=backend)

opti = bind_optimizer(
    "qaoa",
    sampler=sampler,
    number_shots=4096,
    number_layers=4,
    method="COBYLA",
    optimizer_options={"maxiter": 10, "maxfev": 15},
)

solution = matching_pursuit(
    "abstract", y, constraints, number_spins,
    iteration_number=1,
    interaction_size=interaction_size,
    optimizer=opti,
)
```

## MCCO workflow　-- Black-box optimization

To run the full pipeline in a single function:

```python
from troma import solve_via_mcco, get_optimizer

def objective_function(dit_string):
    return int(sum(dit_string))

result = solve_via_mcco(
    objective_function=objective_function,
    number_samples=500,
    dit_string_length=8,
    interaction_size=4,
    iteration_number=5,
    threshold_parameter="Auto",
    optimizer=get_optimizer("spin_chain_nn_max"),
)

print(result["solution_pos"], result["solution_val"])
```

The output dictionary contains:

- `spectrum_pos` — sampled configuration indices
- `spectrum_val` — corresponding objective values
- `spectrum_dits` — dit string representations
- `constraints` — constraint sketch used
- `y` — computed marginals
- `solution_pos` — reconstructed solution indices
- `solution_val` — reconstructed solution values

## Demo notebooks

- [examples/demo.ipynb](examples/demo.ipynb) — core MCCO workflow
- [examples/demo quantum_hardware.ipynb](examples/demo%20quantum_hardware.ipynb) — QAOA on Aer and IBM quantum hardware
- [examples/demo_embeddings.ipynb](examples/demo_embeddings.ipynb) — embedding and restricted search

## Notes

- The root `troma` API exposes the most common workflows.
- Submodules (`troma.optimization`, `troma.decoding`, `troma.sketchs`) remain available for more advanced use cases.
- Internal helpers should not necessarily be considered a stable public API.


