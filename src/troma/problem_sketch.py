from __future__ import annotations

from abc import ABC

from .sketch_map import SketchMap
from .combinatorial_problem import CombinatorialProblem, RestrictedProblem
from .core.structure import Sample, Restriction
from ._validation import ensure_instance


class ProblemSketch(ABC):
    """Abstract base class for problem sketches."""
    pass


class CombinatorialProblemSketch(ProblemSketch):
    def __init__(
        self,
        problem: CombinatorialProblem,
        sketch_map: SketchMap,
        sketch_values: list[float] | None = None,
    ) -> None:
        ensure_instance("problem", problem, CombinatorialProblem)
        ensure_instance("sketch_map", sketch_map, SketchMap)

        self.objective_function = problem.objective_function
        self.problem_size: int = problem.problem_size
        self.problem_dimension: int = problem.problem_dimension
        self.sample: Sample = problem.sample
        self.sketch_map: SketchMap = sketch_map
        self.sketch_values: list[float] | None = sketch_values


class RestrictedProblemSketch(ProblemSketch):
    def __init__(
        self,
        problem: RestrictedProblem,
        sketch_map: SketchMap,
        sketch_values: list[float] | None = None,
    ) -> None:
        ensure_instance("problem", problem, RestrictedProblem)
        ensure_instance("sketch_map", sketch_map, SketchMap)

        self.objective_function = problem.objective_function
        self.problem_size: int = problem.problem_size
        self.problem_dimension: int = problem.problem_dimension
        self.restriction: Restriction = problem.restriction
        self.sample: Sample = problem.sample
        self.restricted_problem_size: int = problem.restricted_problem_size
        self.restricted_problem_dimension: int = problem.restricted_problem_dimension
        self.sketch_map: SketchMap = sketch_map
        self.sketch_values: list[float] | None = sketch_values
