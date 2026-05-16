from __future__ import annotations

import enum
from collections.abc import Callable
from functools import singledispatchmethod
from typing import Any

from .sketch_map import ConstraintSketchMap, SketchMap
from .core.structure import Sample, Restriction
from .core.sampling import objective_sampling, restricted_objective_sampling
from ._validation import ensure_instance


class SketchType(enum.StrEnum):
    NEAREST_NEIGHBORS = "nearest_neighbors"
    ALL_INTERACTIONS = "all_interactions"


class CombinatorialProblem:

    def __init__(
        self,
        objective_function: Callable,
        problem_size: int,
        problem_dimension: int = 2,
    ) -> None:
        self.objective_function: Callable = objective_function
        self.problem_size: int = problem_size
        self.problem_dimension: int = problem_dimension
        self.sample: Sample = Sample()

    def restrict(self, restriction: Restriction) -> RestrictedProblem:
        return RestrictedProblem(self, restriction=restriction)

    def sampling(
        self,
        n_samples: int,
        sampling_function: Callable | None = None,
        sampling_args: dict | None = None,
        threshold_parameter: float | str | None = None,
    ) -> Sample:
        """Sample the combinatorial problem by evaluating the objective function on a random subset of the search space."""
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
    def sketching(self, constraints: Any, interaction_size: int | None = None) -> Any:
        raise TypeError(
            "constraints must be a SketchMap instance or a SketchType string."
        )

    @sketching.register
    def _(self, constraints: SketchMap, interaction_size: int | None = None) -> Any:
        """Sketch from a given SketchMap.

        Parameters
        ----------
        constraints : SketchMap
            The SketchMap containing the constraints for the combinatorial problem.
        interaction_size : int | None, optional
            Unused when constraints is a SketchMap. Provided for API consistency.

        Returns
        -------
        CombinatorialProblemSketch
            A sketch of the combinatorial problem.
        """
        if constraints.sketch_length != self.problem_size:
            raise ValueError(
                f"Sketch length {constraints.sketch_length} does not match problem size {self.problem_size}."
            )

        from .problem_sketch import CombinatorialProblemSketch

        sketch_values = constraints.compute_marginal(self.sample.dit_strings, self.sample.values)
        return CombinatorialProblemSketch(problem=self, sketch_map=constraints, sketch_values=sketch_values)

    @sketching.register(str)
    def _(self, constraints: SketchType, interaction_size: int | None = None) -> Any:
        """Sketch from a SketchType string shorthand.

        Parameters
        ----------
        constraints : SketchType or str
            The sketch type: "nearest_neighbors" or "all_interactions".
        interaction_size : int, required
            Size of interactions. Must be provided when using a string shorthand.

        Returns
        -------
        CombinatorialProblemSketch
            A sketch of the combinatorial problem.
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
            constraints=constraints,
        )
        return self.sketching(sketch_map)

    def mcco_sketching(
        self,
        n_samples: int,
        constraints: SketchMap,
        sampling_function: Callable | None = None,
        sampling_args: dict | None = None,
        threshold_parameter: float | str | None = None,
    ) -> Any:
        """Sample and then sketch in one call."""
        self.sampling(
            n_samples,
            sampling_function=sampling_function,
            sampling_args=sampling_args,
            threshold_parameter=threshold_parameter,
        )
        return self.sketching(constraints)


class RestrictedProblem(CombinatorialProblem):
    """Represents a restricted combinatorial problem."""

    def __init__(
        self,
        problem: CombinatorialProblem | RestrictedProblem,
        restriction: Restriction | None = None,
        dit_restrictions: list[int] | None = None,
        dit_value_restrictions: list[int] | None = None,
        additional_dits_val: int = 0,
    ) -> None:
        ensure_instance("problem", problem, (CombinatorialProblem, RestrictedProblem))

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
        self.restriction: Restriction = restriction
        self.restricted_problem_size: int = (
            len(self.restriction.dit_restrictions)
            if self.restriction.dit_restrictions is not None
            else self.problem_size
        )
        self.restricted_problem_dimension: int = (
            len(self.restriction.dit_value_restrictions)
            if self.restriction.dit_value_restrictions is not None
            else self.problem_dimension
        )

    def sampling(
        self,
        n_samples: int,
        sampling_function: Callable | None = None,
        sampling_args: dict | None = None,
        threshold_parameter: float | str | None = None,
    ) -> Sample:
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
    def sketching(self, constraints: Any, interaction_size: int | None = None) -> Any:
        raise TypeError(
            "constraints must be a SketchMap instance or a SketchType string."
        )

    @sketching.register
    def _(self, constraints: SketchMap, interaction_size: int | None = None) -> Any:
        if constraints.sketch_length != self.restricted_problem_size:
            raise ValueError(
                f"Sketch length {constraints.sketch_length} does not match restricted problem size {self.restricted_problem_size}."
            )

        from .problem_sketch import RestrictedProblemSketch

        sketch = constraints.compute_marginal(self.sample.dit_strings, self.sample.values)
        return RestrictedProblemSketch(problem=self, sketch_map=constraints, sketch_values=sketch)

    @sketching.register(str)
    def _(self, constraints: SketchType, interaction_size: int | None = None) -> Any:
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
            constraints=constraints,
        )

        from .problem_sketch import RestrictedProblemSketch

        sketch_values = sketch_map.compute_marginal(self.sample.dit_strings, self.sample.values)
        return RestrictedProblemSketch(problem=self, sketch_map=sketch_map, sketch_values=sketch_values)
