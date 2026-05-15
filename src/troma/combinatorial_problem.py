from __future__ import annotations

import enum
from functools import singledispatchmethod

from .sketch_map import ConstraintSketchMap, SketchMap
from .core.structure import Sample, Restriction
from .core.sampling import objective_sampling, restricted_objective_sampling

class SketchType(enum.StrEnum):
    NEAREST_NEIGHBORS = "nearest_neighbors"
    ALL_INTERACTIONS = "all_interactions"

class CombinatorialProblem:

    def __init__(
        self,
        objective_function: callable,
        problem_size: int,
        problem_dimension: int = 2,
    ):
        self.objective_function = objective_function
        self.problem_size = problem_size
        self.problem_dimension = problem_dimension
        self.sample = Sample()

    def restrict(self, restriction: Restriction):
        return RestrictedProblem(self, restriction=restriction)

    def sampling(self, n_samples, sampling_function=None, sampling_args=None, threshold_parameter = None):
        """Sample the combinatorial problem by evaluating the objective function on a random subset of the search space, and then applying a threshold to the sampled values. The sampling is done according to a custom sampling function.
        """
        self.sample = objective_sampling(
            self.objective_function,
            n_samples,
            self.problem_size,
            dit_dimension=self.problem_dimension,
            sampling_function=sampling_function,
            sampling_args=sampling_args,
            threshold_parameter=threshold_parameter,
        )
        return self.sample
    
    @singledispatchmethod
    def sketching(self, constraints, interaction_size: int | None = None):
        raise TypeError(
            "constraints must be a SketchMap instance or a SketchType string."
        )

    @sketching.register
    def _(self, constraints: SketchMap, interaction_size: int | None = None):
        """Sketch from a given SketchMap
        
        Parameters
        ----------
        constraints : SketchMap
            The SketchMap containing the constraints for the combinatorial problem.
        interaction_size : int | None, optional
            Unused when constraints is a SketchMap. Provided for API consistency.
        
        Returns
        -------
        ProblemSketch
            A sketch of the combinatorial problem.
        """

        #check the sketch size align the problem size
        if constraints.sketch_length != self.problem_size:
            raise ValueError(
                f"Sketch length {constraints.sketch_length} does not match problem size {self.problem_size}."
            )
        
        from .problem_sketch import CombinatorialProblemSketch

        sketch_values = constraints.compute_marginal(self.sample.dit_strings, self.sample.values)
        return CombinatorialProblemSketch(problem=self, sketch_map=constraints, sketch_values=sketch_values)

    @sketching.register(str)
    def _(self, constraints: SketchType, interaction_size: int | None = None):
        """Sketch from a SketchType string, building the appropriate ConstraintSketchMap.
        
        Parameters
        ----------
        constraints : SketchType or str
            The sketch type to build: "nearest_neighbors" or "all_interactions".
        interaction_size : int, required
            Size of interactions to consider. Must be specified when using SketchType.
        
        Returns
        -------
        ProblemSketch
            A sketch of the combinatorial problem.
        
        Raises
        ------
        TypeError
            If interaction_size is not provided.
        """
        if interaction_size is None:
            raise TypeError(
                "interaction_size is required when constraints is a SketchType. "
                "Please specify the interaction size explicitly."
            )
        
        constraints = SketchType(constraints)
        sketch_map = ConstraintSketchMap(
            sketch_length=self.problem_size,
            interaction_size=interaction_size,
            sketch_dimension=self.problem_dimension,
            constraints=constraints
        )

        return self.sketching(sketch_map)

    def mcco_sketching(self, n_samples, constraints: SketchMap, sampling_function=None, sampling_args=None, threshold_parameter = None):
        """Sample the objective function and compute the MCCO sketch from a given SketchMap
        """

        self.sampling(n_samples, sampling_function=sampling_function, sampling_args=sampling_args, threshold_parameter=threshold_parameter)
        return self.sketching(constraints)

class RestrictedProblem(CombinatorialProblem):
    """Represents a restricted combinatorial problem."""
    
    def __init__(
        self,
        problem,
        restriction: Restriction | None = None,
        dit_restrictions: list[int] | None = None,
        dit_value_restrictions: list[int] | None = None,
        additional_dits_val: int = 0,
    ):
        if not isinstance(problem, (CombinatorialProblem, RestrictedProblem)):
            raise TypeError("problem must be an instance of CombinatorialProblem or RestrictedProblem")

        super().__init__(
            objective_function=problem.objective_function,
            problem_size=problem.problem_size,
            problem_dimension=problem.problem_dimension,
        )

        # Backward compatibility for legacy positional calls:
        # RestrictedProblem(problem, dit_restrictions, dit_value_restrictions, additional_dits_val)
        if restriction is not None and not isinstance(restriction, Restriction):
            dit_restrictions = restriction
            restriction = None

        if restriction is None:
            restriction = Restriction(
                dit_restrictions=dit_restrictions,
                dit_value_restrictions=dit_value_restrictions,
                additional_dits_val=additional_dits_val,
            )
        self.restriction = restriction
        self.restricted_problem_size = (
            len(self.restriction.dit_restrictions)
            if self.restriction.dit_restrictions is not None
            else self.problem_size
        )
        self.restricted_problem_dimension = (
            len(self.restriction.dit_value_restrictions)
            if self.restriction.dit_value_restrictions is not None
            else self.problem_dimension
        )

    def sampling(self, n_samples, sampling_function=None, sampling_args=None, threshold_parameter = None):
        """
        """

        self.sample = restricted_objective_sampling(
            self.objective_function,
            n_samples,
            dit_string_length=self.problem_size,
            dit_dimension=self.problem_dimension,
            sampling_function=sampling_function,
            sampling_args=sampling_args,
            threshold_parameter=threshold_parameter,
            restriction=self.restriction,
        )
        return self.sample
    
    @singledispatchmethod
    def sketching(self, constraints, interaction_size: int | None = None):
        raise TypeError(
            "constraints must be a SketchMap instance or a SketchType string."
        )
    
    @sketching.register
    def _(self, constraints: SketchMap, interaction_size: int | None = None):
        """
        """

        #check the sketch size align the restricted problem size
        if constraints.sketch_length != self.restricted_problem_size:
            raise ValueError(
                f"Sketch length {constraints.sketch_length} does not match restricted problem size {self.restricted_problem_size}."
            )
        
        from .problem_sketch import RestrictedProblemSketch

        sketch = constraints.compute_marginal(self.sample.dit_strings, self.sample.values)
        return RestrictedProblemSketch(problem=self, sketch_map=constraints, sketch=sketch)

    @sketching.register(str)
    def _(self, constraints: SketchType, interaction_size: int | None = None):
        """
        """

        if interaction_size is None:
            raise TypeError(
                "interaction_size is required when constraints is a SketchType. "
                "Please specify the interaction size explicitly."
            )

        constraints = SketchType(constraints)
        sketch_map = ConstraintSketchMap(
            sketch_length=self.restricted_problem_size,
            interaction_size=interaction_size,
            sketch_dimension=self.restricted_problem_dimension,
            constraints=constraints
        )

        from .problem_sketch import RestrictedProblemSketch

        sketch_values = sketch_map.compute_marginal(self.sample.dit_strings, self.sample.values)
        return RestrictedProblemSketch(problem=self, sketch_map=sketch_map, sketch_values=sketch_values)