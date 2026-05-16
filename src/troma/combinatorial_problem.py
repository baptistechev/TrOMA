from __future__ import annotations

import enum
from collections.abc import Callable
from functools import singledispatchmethod
from typing import Any

import numpy as np

from .sketch_map import ConstraintSketchMap, SketchMap
from .core.structure import DitString, Sample, Restriction
from .core.embedding import reverse_spectrum_restriction
from ._validation import ensure_instance, ensure_callable, ensure_int, ensure_optional_dict


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

    @staticmethod
    def _uniform_sampling(
        n_samples: int,
        length: int,
        dimension: int,
    ) -> tuple[np.ndarray, list[DitString]]:
        """Uniform random sampler over the dit-string space."""
        total_states = dimension ** length
        rng = np.random.default_rng()
        indexes = rng.integers(0, total_states, size=n_samples, dtype=np.int64)
        dit_strings = [DitString.from_integer(int(i), length, dimension) for i in indexes]
        return indexes, dit_strings

    @staticmethod
    def _evaluate_and_filter(
        indexes: np.ndarray,
        dit_strings: list[DitString],
        objective_function: Callable,
        threshold_parameter: float | str | None,
        full_dit_strings: list[DitString] | None = None,
    ) -> Sample:
        """Evaluate the objective, apply threshold, return sorted non-zero Sample.

        Parameters
        ----------
        full_dit_strings : list[DitString] or None, optional
            When provided, the objective is evaluated on these (full-space) dit strings
            while ``dit_strings`` (restricted-space) are stored in the returned Sample.
            Used by RestrictedProblem so the objective always receives full-space inputs.
        """
        eval_strings = full_dit_strings if full_dit_strings is not None else dit_strings
        values = np.array([objective_function(np.asarray(s)) for s in eval_strings])

        if threshold_parameter == "Auto":
            non_zero = values[values != 0]
            threshold_parameter = np.percentile(non_zero, 90) if non_zero.size > 0 else 0
        if threshold_parameter is not None:
            values[values < threshold_parameter] = 0

        triples = [
            (int(i), s, int(v))
            for i, s, v in zip(indexes, dit_strings, values) if v != 0
        ]
        triples.sort(key=lambda t: t[0])
        if triples:
            out_indexes, out_strings, out_values = zip(*triples)
        else:
            out_indexes, out_strings, out_values = [], [], []
        return Sample(
            indexes=list(out_indexes),
            values=list(out_values),
            dit_strings=list(out_strings),
        )

    def sampling(
        self,
        n_samples: int,
        sampling_function: Callable | None = None,
        sampling_args: dict | None = None,
        threshold_parameter: float | str | None = None,
    ) -> Sample:
        """Sample the problem by evaluating the objective on a random subset of the search space.

        Parameters
        ----------
        n_samples : int
            Number of configurations to sample.
        sampling_function : Callable or None, optional
            Custom sampler. Must accept ``(n_samples, length, dimension)`` and return
            ``(indexes, dit_strings)``. Defaults to uniform random sampling.
        sampling_args : dict or None, optional
            Extra keyword arguments forwarded to ``sampling_function``.
        threshold_parameter : float, "Auto", or None, optional
            Samples whose objective value is below the threshold are discarded.
            ``"Auto"`` sets the threshold to the 90th percentile of non-zero values.
        """
        n_samples = ensure_int("n_samples", n_samples, min_value=1)
        if sampling_function is None:
            sampling_function = self._uniform_sampling
        else:
            ensure_callable("sampling_function", sampling_function)
        ensure_optional_dict("sampling_args", sampling_args)
        if threshold_parameter is not None and threshold_parameter != "Auto":
            if not isinstance(threshold_parameter, (int, float, np.integer, np.floating)):
                raise TypeError("threshold_parameter must be a real number, 'Auto', or None.")

        indexes, dit_strings = sampling_function(
            n_samples, self.problem_size, self.problem_dimension,
            **(sampling_args or {}),
        )
        self.sample = self._evaluate_and_filter(
            indexes, dit_strings, self.objective_function, threshold_parameter,
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
            ``"nearest_neighbors"`` or ``"all_interactions"``.
        interaction_size : int, required
            Size of interactions.

        Returns
        -------
        CombinatorialProblemSketch
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

        # Backward compatibility for legacy positional calls
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
        """Sample the restricted problem.

        Samples from the restricted search space, maps configurations back to
        the full space for objective evaluation, and stores the restricted-space
        dit strings in the sample.
        """
        # No active restriction — delegate to the unrestricted sampler.
        if (
            self.restriction.dit_restrictions is None
            and self.restriction.dit_value_restrictions is None
        ):
            return super().sampling(n_samples, sampling_function, sampling_args, threshold_parameter)

        n_samples = ensure_int("n_samples", n_samples, min_value=1)
        if sampling_function is None:
            sampling_function = self._uniform_sampling
        else:
            ensure_callable("sampling_function", sampling_function)
        ensure_optional_dict("sampling_args", sampling_args)
        if threshold_parameter is not None and threshold_parameter != "Auto":
            if not isinstance(threshold_parameter, (int, float, np.integer, np.floating)):
                raise TypeError("threshold_parameter must be a real number, 'Auto', or None.")

        # Sample in the restricted space.
        indexes_rest, dit_strings_rest = sampling_function(
            n_samples, self.restricted_problem_size, self.restricted_problem_dimension,
            **(sampling_args or {}),
        )

        # Map back to full space to evaluate the objective.
        dit_strings_full = reverse_spectrum_restriction(
            dit_strings_rest,
            original_size=self.problem_size,
            dit_restrictions=self.restriction.dit_restrictions,
            dit_value_restrictions=self.restriction.dit_value_restrictions,
            additional_dits_val=self.restriction.additional_dits_val,
        )

        # Evaluate, threshold and keep non-zero samples (store restricted dit strings).
        self.sample = self._evaluate_and_filter(
            indexes_rest, dit_strings_rest, self.objective_function, threshold_parameter,
            full_dit_strings=dit_strings_full,
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
