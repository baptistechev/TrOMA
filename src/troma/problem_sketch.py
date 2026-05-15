
from __future__ import annotations

from .sketch_map import SketchMap
from .combinatorial_problem import CombinatorialProblem, RestrictedProblem

from abc import ABC

class ProblemSketch(ABC):
    """Abstract base class for problem sketches."""
    pass
    

class CombinatorialProblemSketch(ProblemSketch):
    def __init__(
        self, 
        problem, 
        sketch_map: SketchMap, 
        sketch_values = None
    ):
        
        if not isinstance(problem, CombinatorialProblem):
            raise TypeError(
                "problem must be an instance of CombinatorialProblem"
            )
        if not isinstance(sketch_map, SketchMap):
            raise TypeError("sketch_map must be an instance of SketchMap")

        self.objective_function = problem.objective_function
        self.problem_size = problem.problem_size
        self.problem_dimension = problem.problem_dimension
        self.sample = problem.sample
        self.sketch_map = sketch_map
        self.sketch_values = sketch_values

class RestrictedProblemSketch(ProblemSketch):
    def __init__(
        self, 
        problem, 
        sketch_map: SketchMap, 
        sketch_values = None
    ):
        
        if not isinstance(problem, RestrictedProblem):
            raise TypeError(
                "problem must be an instance of RestrictedProblem"
            )
        if not isinstance(sketch_map, SketchMap):
            raise TypeError("sketch_map must be an instance of SketchMap")

        self.objective_function = problem.objective_function
        self.problem_size = problem.problem_size
        self.problem_dimension = problem.problem_dimension
        self.restriction = problem.restriction
        self.sample = problem.sample
        self.restricted_problem_size = problem.restricted_problem_size
        self.restricted_problem_dimension = problem.restricted_problem_dimension
        
        self.sketch_map = sketch_map
        self.sketch_values = sketch_values