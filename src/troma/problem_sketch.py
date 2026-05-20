from __future__ import annotations

from abc import ABC, abstractmethod

from .sketch_map import SketchMap
from .combinatorial_problem import CombinatorialProblem, RestrictedProblem
from .core.structure import Sample, Restriction
from ._validation import _Validator
from .core.structure import Hamiltonian


class ProblemSketch(ABC):
    """Abstract base class for problem sketches."""

    @abstractmethod
    def update_sketch(self, r: list[float]) -> "ProblemSketch":
        """Return a new sketch instance with updated sketch values."""
    
    def to_hamiltonian(self) -> Hamiltonian:
        """Convert the problem sketch to a Hamiltonian representation."""
        return Hamiltonian.from_problem_sketch(self)
    
    def to_hubo(self) -> Hamiltonian:
        """Convert the problem sketch to a HUBO representation."""
        return self.to_hamiltonian().to_hubo()

class CombinatorialProblemSketch(ProblemSketch):
    def __init__(
        self,
        problem: CombinatorialProblem,
        sketch_map: SketchMap,
        sketch_values: list[float]
    ) -> None:
        _Validator.ensure_instance("problem", problem, CombinatorialProblem)
        _Validator.ensure_instance("sketch_map", sketch_map, SketchMap)

        self.objective_function = problem.objective_function
        self.problem_size: int = problem.problem_size
        self.problem_dimension: int = problem.problem_dimension
        self.sample: Sample = problem.sample
        self.sketch_map: SketchMap = sketch_map
        self.sketch_values: list[float] = sketch_values

    @classmethod
    def from_copy(
        cls,
        other: "CombinatorialProblemSketch",
    ) -> "CombinatorialProblemSketch":
        """Build a new instance by copying an existing one."""
        _Validator.ensure_instance("other", other, CombinatorialProblemSketch)
        new_instance = cls.__new__(cls)
        new_instance.objective_function = other.objective_function
        new_instance.problem_size = other.problem_size
        new_instance.problem_dimension = other.problem_dimension
        new_instance.sample = other.sample
        new_instance.sketch_map = other.sketch_map
        new_instance.sketch_values = other.sketch_values
        return new_instance

    def update_sketch(self, r: list[float]) -> "CombinatorialProblemSketch":
        """Return a new instance with updated sketch values."""
        new_instance = CombinatorialProblemSketch.from_copy(self)
        new_instance.sketch_values = r
        return new_instance

class RestrictedProblemSketch(ProblemSketch):
    def __init__(
        self,
        problem: RestrictedProblem,
        sketch_map: SketchMap,
        sketch_values: list[float]
    ) -> None:
        _Validator.ensure_instance("problem", problem, RestrictedProblem)
        _Validator.ensure_instance("sketch_map", sketch_map, SketchMap)

        self.objective_function = problem.objective_function
        self.problem_size: int = problem.problem_size
        self.problem_dimension: int = problem.problem_dimension
        self.restriction: Restriction = problem.restriction
        self.sample: Sample = problem.sample
        self.restricted_problem_size: int = problem.restricted_problem_size
        self.restricted_problem_dimension: int = problem.restricted_problem_dimension
        self.sketch_map: SketchMap = sketch_map
        self.sketch_values: list[float] = sketch_values

    @classmethod
    def from_copy(
        cls,
        other: "RestrictedProblemSketch",
    ) -> "RestrictedProblemSketch":
        """Build a new instance by copying an existing one."""
        _Validator.ensure_instance("other", other, RestrictedProblemSketch)
        new_instance = cls.__new__(cls)
        new_instance.objective_function = other.objective_function
        new_instance.problem_size = other.problem_size
        new_instance.problem_dimension = other.problem_dimension
        new_instance.restriction = other.restriction
        new_instance.sample = other.sample
        new_instance.restricted_problem_size = other.restricted_problem_size
        new_instance.restricted_problem_dimension = other.restricted_problem_dimension
        new_instance.sketch_map = other.sketch_map
        new_instance.sketch_values = other.sketch_values
        return new_instance

    def update_sketch(self, r: list[float]) -> "RestrictedProblemSketch":
        """Return a new instance with updated sketch values."""
        new_instance = RestrictedProblemSketch.from_copy(self)
        new_instance.sketch_values = r
        return new_instance
