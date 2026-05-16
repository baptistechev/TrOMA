from importlib.metadata import PackageNotFoundError, version

# Import core structure classes
from .core.structure import DitString, CylinderSet, Restriction, Sample

# Import embedding
from .core.embedding import (
	spectrum_embedding,
	spectrum_restriction,
	reverse_spectrum_restriction
)

# Import sketch map (new API - replaces old sketch.py)
from .sketch_map import ConstraintSketchMap, ExplicitSketchMap

# Import problem sketch (no circular deps with sketch_map)
from .problem_sketch import ProblemSketch, CombinatorialProblemSketch, RestrictedProblemSketch

# Import combinatorial problem (depends on sketch_map)
from .combinatorial_problem import CombinatorialProblem

# Import optimization
from .optimization.optimizer import (
	bind_optimizer,
	get_optimizer,
	optimize,
)

# Import matching pursuit (depends on problem_sketch)
from .matching_pursuit import (
	bind_matching_pursuit,
	get_matching_pursuit,
	matching_pursuit,
	MatchingPursuitResults,
)

# Import higher-level APIs (depends on other modules)
# TODO: pipelines.py will be removed - keep commented for now
# from .pipelines import solve_via_mcco, embedding_and_solve_via_mcco

# Import old sketchs module for backward compatibility (functions only, not classes)
from .sketchs import (
	constraint_compute_marginal,
	constraints_for_nearest_neighbors_interactions,
	constraints_for_all_interactions,
	reconstruct_structured_matrix_column,
	explicit_compute_marginal,
	nearest_neighbors_interactions_sketch,
	all_interactions_sketch,
	random_sketch,
)

try:
	__version__ = version("troma")
except PackageNotFoundError:
	__version__ = "0+unknown"

__all__ = [
	"__version__",
	"DitString",
	"CylinderSet",
	"spectrum_embedding",
	"spectrum_restriction",
	"reverse_spectrum_restriction",
	"CombinatorialProblem",
	"Restriction",
	"Sample",
	"ProblemSketch",
	"CombinatorialProblemSketch",
	"RestrictedProblemSketch",
	"ConstraintSketchMap",
	"ExplicitSketchMap",
	"bind_optimizer",
	"get_optimizer",
	"optimize",
	"bind_matching_pursuit",
	"get_matching_pursuit",
	"matching_pursuit",
	"MatchingPursuitResults",
	"constraint_compute_marginal",
	"constraints_for_nearest_neighbors_interactions",
	"constraints_for_all_interactions",
	"reconstruct_structured_matrix_column",
	"explicit_compute_marginal",
	"nearest_neighbors_interactions_sketch",
	"all_interactions_sketch",
	"random_sketch",
]
