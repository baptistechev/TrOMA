from importlib.metadata import PackageNotFoundError, version

from .modeling import mcco_modeling
from .mcco_workflow import solve_via_mcco
from .data_structure import (
	integer_to_dit_string,
	dit_string_to_integer,
	dit_string_to_computational_basis,
    create_cylinder_set_indicator,
    kronecker_develop,
    belongs_to_cylinder_set
)
from .sketchs import ConstraintSketch, ExplicitSketch
from .optimization.optimizer import (
	bind_optimizer,
	get_optimizer,
	optimize,
)
from .decoding.matching_pursuit import (
	bind_matching_pursuit,
	get_matching_pursuit,
	matching_pursuit,
)

try:
	__version__ = version("troma")
except PackageNotFoundError:
	__version__ = "0+unknown"

__all__ = [
	"__version__",
	"mcco_modeling",
	"solve_via_mcco",
	"integer_to_dit_string",
	"dit_string_to_integer",
	"dit_string_to_computational_basis",
	"create_cylinder_set_indicator",
    "kronecker_develop",
	"belongs_to_cylinder_set",
	"ConstraintSketch",
	"ExplicitSketch",
	"bind_optimizer",
	"get_optimizer",
	"optimize",
	"bind_matching_pursuit",
	"get_matching_pursuit",
	"matching_pursuit",
]
