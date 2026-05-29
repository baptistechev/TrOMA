# TrOMA

TrOMA is a Python library for optimization of black-box functions with binary inputs.

B. Chevalier, S. Yamaguchi, W. Roga, M. Takeoka, A Compressive Sensing Inspired Monte-Carlo Method for Combinatorial Optimization, arXiv:2510.24755 (2025).

The method builds a surrogate model from sketches through a process called MCCO modeling. The resulting cost function can be converted into an Ising Hamiltonian and optimized with a quantum processor.

Alternatively, the library can be used to efficiently deal with large size compressive sensing problems and benefit from quantum computers as explained in:

B. Chevalier, W. Roga, M. Takeoka, Compressed sensing enhanced by a quantum approximate optimization algorithm, Phys. Rev. A 110, 062410 (2024).

See the doc at https://baptistechev.github.io/TrOMA/

## Installation

```bash
pip install troma
```

```python
import troma
print(troma.__version__)
```

## Main API

```python
from troma import (
    CombinatorialProblem,
    ConstraintSketchMap,
    matching_pursuit,
    get_optimizer,
    bind_optimizer,
    DitString,
    Restriction,
    spectrum_embedding,
)
```

## Core workflow

The MCCO pipeline has five steps: define the problem, sample it, build a sketch map, sketch, and run matching pursuit.

```python
from troma import CombinatorialProblem, ConstraintSketchMap, matching_pursuit, get_optimizer

# 1. Define the problem
def objective(dit_string):
    ...  # returns a scalar reward

problem = CombinatorialProblem(objective, problem_size=12, problem_dimension=2)

# 2. Sample the search space
problem.sampling(n_samples=800, seed=3)

# 3. Build a sketch map
sketch_map = ConstraintSketchMap(
    sketch_length=12,
    interaction_size=2,
    constraints="nearest_neighbors",
)

# 4. Sketch the problem
problem_sketch = problem.sketching(sketch_map)

# 5. Run matching pursuit
result = matching_pursuit(
    problem_sketch,
    iteration_number=5,
    optimizer=get_optimizer("spin_chain_nn_max"),
)

print(result.positions)   # integer indices of the best configurations found
print(result.values)      # corresponding objective values
print(result.dit_strings) # corresponding DitString objects
```

The shorthand `"nearest_neighbors"` string can be passed directly to `sketching` instead of constructing a `ConstraintSketchMap` manually:

```python
problem_sketch = problem.sketching("nearest_neighbors", interaction_size=2)
```

## Optimizers

Pass an optimizer via the `optimizer` keyword of `matching_pursuit`.

```python
from troma import get_optimizer, bind_optimizer

opti = get_optimizer("dual_annealing")
```

| Name | Type | Notes |
|---|---|---|
| `spin_chain_nn_max` | Classical | Spin-chain nearest-neighbor heuristic |
| `brute_force_max` | Classical | Exhaustive search |
| `dual_annealing` | Classical | SciPy dual annealing |
| `simulated_annealing` | Classical | Neal simulated annealing |
| `digital_annealing` | Quantum-inspired | D-Wave Neal QUBO solver |
| `qaoa` | Quantum | QAOA via Qiskit |
| `aoa` | Quantum | AOA (Adaptive Optimization Algorithm) via Qiskit — Hamming-weight-preserving mixer |

## Quantum optimizer: QAOA

### Local simulation

```python
from qiskit_aer import AerSimulator
from troma import bind_optimizer, matching_pursuit

backend = AerSimulator()
opti = bind_optimizer("qaoa", backend=backend, number_shots=4096)

result = matching_pursuit(problem_sketch, iteration_number=2, optimizer=opti)
print(result.positions)
```

### Running on IBM quantum hardware

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from troma import bind_optimizer, matching_pursuit

service = QiskitRuntimeService()
backend = service.backend("ibm_marrakesh")

opti = bind_optimizer(
    "qaoa",
    backend=backend,
    number_shots=4096,
    number_layers=4,
    method="COBYLA",
    optimizer_options={"maxiter": 10},
    sampler_options={"max_execution_time": 6},
)

result = matching_pursuit(problem_sketch, iteration_number=1, optimizer=opti)
print(result.positions)
```

### Pre-training before running on hardware

`pretrain=True` runs a grid scan + local refinement on a local `AerSimulator` first, then
transfers the warm-started parameters to the real device, reducing the number of QPU iterations needed.

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from troma import bind_optimizer, matching_pursuit

service = QiskitRuntimeService()
backend = service.backend("ibm_marrakesh")

opti = bind_optimizer(
    "qaoa",
    backend=backend,
    number_layers=4,
    number_shots=1024,
    method="COBYLA",
    optimizer_options={"maxiter": 10},
    pretrain=True,
    pretrain_options={
        "num_grid_points": 20,       # 20² = 400 grid points for p=1 scan
        "number_shots": 1024,
        "sim_method": "statevector",
        "max_sim_iter": 60,
    },
)

result = matching_pursuit(
    problem_sketch,
    iteration_number=1,
    optimizer=opti,
    post_processing="2_bit_swap",
    verbose=True,
)
print(result.positions)
```

## Quantum optimizer: AOA

AOA (Adaptive Optimization Algorithm, [arXiv:2211.13227](https://arxiv.org/abs/2211.13227)) uses a Hamming-weight-preserving mixer, making it well suited for problems where the number of active bits is constrained.

```python
from qiskit_ibm_runtime import QiskitRuntimeService
from troma import bind_optimizer, matching_pursuit

service = QiskitRuntimeService()
backend = service.backend("ibm_marrakesh")

opti = bind_optimizer(
    "aoa",
    backend=backend,
    number_layers=4,
    number_shots=4096,
    method="COBYLA",
    initial_state="dicke",
    hamming_weight=1,
    mixer="ring",
    optimizer_options={"maxiter": 10},
)

result = matching_pursuit(problem_sketch, iteration_number=1, optimizer=opti)
print(result.positions)
```

## Restricted search space

When some coordinates are known or trivially fixed, restrict the problem before sampling. The restriction maps solutions back to the full space automatically.

```python
import numpy as np
from troma import CombinatorialProblem, Restriction, matching_pursuit, get_optimizer

# Define the problem over the full space
problem = CombinatorialProblem(ev_conf, problem_size=13)

# Fix the first bit to 1, optimize over bits 1–12
restriction = Restriction(
    dit_restrictions=np.arange(1, 13),   # indices of free coordinates
    dit_value_restrictions=None,
    additional_dits_val=1,               # value imposed on fixed coordinates
)
restricted_problem = problem.restrict(restriction)

# Sample and sketch in the restricted space
restricted_problem.sampling(n_samples=400)
problem_sketch = restricted_problem.sketching("nearest_neighbors", interaction_size=4)

# Result positions are mapped back to the full space
result = matching_pursuit(problem_sketch, iteration_number=5, optimizer=get_optimizer("spin_chain_nn_max"))
print(result.positions)
```

## Spectrum embedding

`spectrum_embedding` embeds a spectrum into a higher-dimensional space by inserting
additional coordinates with a fixed value.

```python
from troma import DitString, spectrum_embedding

emb_spectrum = spectrum_embedding(
    spectrum_bin,
    additional_dits=[0],        # positions of the new coordinates
    dimension_mapping=None,
    additional_dits_val=1,      # value to assign to the added coordinates
)
emb_pos = [DitString(s).to_integer() for s in emb_spectrum]
```

## DitString

`DitString` is the standard representation for configurations. It bundles the
sequence of dit values with the alphabet size (dimension).

```python
from troma import DitString

s = DitString([0, 1, 1, 0], dimension=2)
i = s.to_integer()                                         # convert to integer index
s2 = DitString.from_integer(i, length=4, dimension=2)     # reconstruct from index
arr = s.tolist()                                           # plain Python list
```

## Demo notebooks

- [examples/demo.ipynb](examples/demo.ipynb) — core MCCO workflow
- [examples/demo quantum_hardware.ipynb](examples/demo%20quantum_hardware.ipynb) — QAOA on Aer and IBM quantum hardware
- [examples/demo_embeddings.ipynb](examples/demo_embeddings.ipynb) — embedding and restricted search

## Notes

- Submodules (`troma.optimization`, `troma.decoding`, `troma.sketchs`) remain available for advanced use.
- Internal helpers should not be considered a stable public API.
