# TrOMA

TrOMA is a small Python library for reconstructing or approximating sparse discrete functions from marginals/sketches, with an API centered around:

- explicit or constraint-based sketch construction,
- `matching_pursuit` for sparse reconstruction,
- several classical and quantum optimizers,
- a complete MCCO workflow through `solve_via_mcco`.

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
)
from troma import data_structure as ds
```

## Quick example: abstract reconstruction

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

## Quick example: explicit sketch

```python
import numpy as np

from troma import ExplicitSketch, matching_pursuit

number_spins = 7
interaction_size = 4

full_spectrum = np.zeros((2**number_spins,))

sketch = ExplicitSketch.build_nearest_neighbors_sketch(number_spins, interaction_size)
y = ExplicitSketch.compute_marginal(full_spectrum, sketch)

solution = matching_pursuit("explicit", y, sketch, 2)
print(solution)
```

## Choosing an optimizer

The recommended path is to use the common interface:

```python
from troma import get_optimizer, bind_optimizer, matching_pursuit

opti = get_optimizer("dual_annealing")
# or
opti = bind_optimizer("qaoa", number_shots=4096)
```

Optimizers used in the demo notebook:

- `spin_chain_nn_max`
- `brute_force_max`
- `dual_annealing`
- `simulated_annealing`
- `digital_annealing`
- `qaoa`

## MCCO workflow

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

print(result["solution"])
```

The output is a dictionary containing in particular:

- `spectrum_pos`
- `spectrum_val`
- `spectrum_bin`
- `constraints`
- `y`
- `solution`

## Demo notebook

For a more complete example, see:

- [examples/demo.ipynb](examples/demo.ipynb)

## Notes

- The root `troma` API exposes the most common workflows.
- Submodules (`troma.optimization`, `troma.decoding`, `troma.sketchs`) remain available for more advanced use cases.
- Internal helpers should not necessarily be considered a stable public API.


