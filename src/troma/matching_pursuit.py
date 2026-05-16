from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from importlib import import_module
from typing import Any, Callable
import numpy as np
from .core.data_structure import integer_to_dit_string
from .core.structure import DitString
from .core.embedding import reverse_spectrum_restriction
from ._validation import ensure_callable, ensure_nonempty_str, ensure_tuple, ensure_optional_dict, ensure_instance

# Import for ProblemSketch support
from .problem_sketch import ProblemSketch, RestrictedProblemSketch
from .sketch_map import ConstraintSketchMap, ExplicitSketchMap


MatchingPursuitFunction = Callable[..., Any]


@dataclass(frozen=True)
class MatchingPursuitResults:
    """Structured output of ``matching_pursuit``.

    Attributes
    ----------
    positions : np.ndarray
        Selected line positions (atom indices).
    values : np.ndarray
        Coefficients associated with ``positions``.
    dit_strings : list[DitString]
        Dit-string encoding of each selected position.
    backend_name : str
        Backend used to run matching pursuit (``"explicit"`` or ``"abstract"``).
    dit_string_length : int
        Dit-string length used for encoding.
    dit_dimension : int
        Dit dimension used for encoding.
    interaction_size : int | None
        Interaction size when available.
    marginals : np.ndarray
        Marginals used to run matching pursuit.
    raw : np.ndarray
        Raw backend output as a 2-column array ``[index, coefficient]``.
    """

    positions: np.ndarray
    values: np.ndarray
    dit_strings: list[DitString]
    backend_name: str
    dit_string_length: int
    dit_dimension: int
    interaction_size: int | None
    marginals: np.ndarray
    raw: np.ndarray

    @property
    def n_lines(self) -> int:
        """Number of selected lines."""
        return int(self.positions.size)

    @property
    def line_positions(self) -> np.ndarray:
        """Alias for selected line positions."""
        return self.positions

    @property
    def line_values(self) -> np.ndarray:
        """Alias for selected line values."""
        return self.values

    def as_array(self) -> np.ndarray:
        """Return the raw 2-column backend output."""
        return self.raw

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary view of this result."""
        return {
            "positions": self.positions,
            "values": self.values,
            "dit_strings": self.dit_strings,
            "backend_name": self.backend_name,
            "dit_string_length": self.dit_string_length,
            "dit_dimension": self.dit_dimension,
            "interaction_size": self.interaction_size,
            "marginals": self.marginals,
            "raw": self.raw,
            "n_lines": self.n_lines,
        }


def _coerce_solution_array(solution: Any) -> np.ndarray:
    """Normalize backend solution to shape ``(n, 2)``."""
    arr = np.asarray(solution)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise ValueError("Matching pursuit backend returned a 1D result with invalid shape.")
        return arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Matching pursuit backend must return an array-like of shape (n, 2).")
    return arr


def _build_matching_pursuit_results(
    solution: Any,
    *,
    backend_name: str,
    marginals: Any,
    dit_string_length: int,
    dit_dimension: int,
    interaction_size: int | None,
) -> MatchingPursuitResults:
    """Build a structured ``MatchingPursuitResults`` object from backend output."""
    raw = _coerce_solution_array(solution)
    positions = raw[:, 0].astype(int) if raw.size > 0 else np.array([], dtype=int)
    values = raw[:, 1].astype(float) if raw.size > 0 else np.array([], dtype=float)
    dit_strings = [
        integer_to_dit_string(
            int(index),
            dit_string_length=dit_string_length,
            dit_dimension=dit_dimension,
            convention="R",
        )
        for index in positions
    ]

    return MatchingPursuitResults(
        positions=positions,
        values=values,
        dit_strings=dit_strings,
        backend_name=backend_name,
        dit_string_length=int(dit_string_length),
        dit_dimension=int(dit_dimension),
        interaction_size=None if interaction_size is None else int(interaction_size),
        marginals=np.asarray(marginals, dtype=float),
        raw=raw,
    )


class MatchingPursuit(ABC):
    """Common interface for all matching-pursuit backends."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run matching pursuit and return the sparse reconstruction."""


class FunctionMatchingPursuit(MatchingPursuit):
    """Adapter exposing a plain matching-pursuit function through a common interface."""

    def __init__(
        self,
        name: str,
        function: MatchingPursuitFunction,
        default_args: tuple[Any, ...] = (),
        default_kwargs: dict[str, Any] | None = None,
    ) -> None:
        ensure_nonempty_str("name", name)
        ensure_callable("function", function)
        ensure_tuple("default_args", default_args)
        ensure_optional_dict("default_kwargs", default_kwargs)
        self.name = name
        self._function = function
        self._default_args = default_args
        self._default_kwargs = dict(default_kwargs or {})
        try:
            self._signature = inspect.signature(function)
        except (TypeError, ValueError):
            self._signature = None

    def _prepare_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self._signature is None:
            return args, kwargs

        parameters = list(self._signature.parameters.values())
        accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters)
        if accepts_var_kwargs:
            return args, kwargs

        eligible_keyword_names = {
            param.name
            for param in parameters
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        consumed_positional_or_keyword = set()
        positional_budget = len(args)
        for param in parameters:
            if positional_budget <= 0:
                break
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    consumed_positional_or_keyword.add(param.name)
                positional_budget -= 1

        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in eligible_keyword_names and key not in consumed_positional_or_keyword
        }
        return args, filtered_kwargs

    def run(self, *args: Any, **kwargs: Any) -> Any:
        final_args = self._default_args + args
        final_kwargs = dict(self._default_kwargs)
        final_kwargs.update(kwargs)
        final_args, final_kwargs = self._prepare_call(final_args, final_kwargs)
        return self._function(*final_args, **final_kwargs)

    def with_defaults(self, *args: Any, **kwargs: Any) -> "FunctionMatchingPursuit":
        """Return a new matching-pursuit adapter with additional default arguments pre-bound."""

        merged_kwargs = dict(self._default_kwargs)
        merged_kwargs.update(kwargs)
        return FunctionMatchingPursuit(
            name=self.name,
            function=self._function,
            default_args=self._default_args + args,
            default_kwargs=merged_kwargs,
        )


_MATCHING_PURSUIT_REGISTRY: dict[str, tuple[str, str]] = {
    "explicit": ("decoding.decoding_proced", "matchingpursuit_explicit"),
    "abstract": ("decoding.decoding_proced", "matchingpursuit_abstract"),
}


def _load_module(module_name: str):
    ensure_nonempty_str("module_name", module_name)
    if __package__:
        try:
            return import_module(f".{module_name}", package=__package__)
        except Exception:
            pass
    return import_module(module_name)


def _resolve_matching_pursuit_function(name: str) -> MatchingPursuitFunction:
    ensure_nonempty_str("name", name)
    key = name.lower()
    if key not in _MATCHING_PURSUIT_REGISTRY:
        raise ValueError(
            f"Unknown matching pursuit '{name}'. Available backends: {', '.join(list_matching_pursuits())}."
        )

    module_name, function_name = _MATCHING_PURSUIT_REGISTRY[key]
    module = _load_module(module_name)
    function = getattr(module, function_name)
    return function


def list_matching_pursuits() -> list[str]:
    """
    Return all registered matching-pursuit backend names.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    list[str]
        A list of all registered matching-pursuit backend names.
    """
    return sorted(_MATCHING_PURSUIT_REGISTRY)


def get_matching_pursuit(name: str) -> MatchingPursuit:
    """
    Instantiate a matching-pursuit adapter from a registered backend name.
    
    Parameters
    ----------
    name : str
        The name of the matching-pursuit backend to retrieve. Available backends can be listed with `list_matching_pursuits()`.
    
    Returns
    -------
    MatchingPursuit
        An instance of the requested matching-pursuit adapter.

        Examples
        --------
        - ``get_matching_pursuit("explicit")`` returns an adapter usable as
            ``mp.run(marginals, sketch=sketch, iteration_number=10, step=0.2)``.
        - ``get_matching_pursuit("abstract")`` returns an adapter usable as
            ``mp.run(marginals, dit_constraints=constraints, dit_string_length=6, iteration_number=10, interaction_size=2)``.
    """
    ensure_nonempty_str("name", name)
    function = _resolve_matching_pursuit_function(name)
    return FunctionMatchingPursuit(name=name.lower(), function=function)


def bind_matching_pursuit(name: str, *args: Any, **kwargs: Any) -> MatchingPursuit:
    """
    Instantiate a matching-pursuit adapter with default arguments pre-bound.
    
    Parameters
    ----------
    name : str
        The name of the matching-pursuit backend to retrieve. Available backends can be listed with `list_matching_pursuits()`.
    *args : Any
        Positional arguments to pre-bind. Typical examples are ``(marginals,)``.
    **kwargs : Any
        Keyword arguments matching the chosen backend. For example:

        - ``bind_matching_pursuit("explicit", marginals, sketch=sketch, iteration_number=10)``
        - ``bind_matching_pursuit("explicit", marginals, sketch=sketch, iteration_number=10, step=0.2, optimizer=get_optimizer("brute_force_max"))``
        - ``bind_matching_pursuit("abstract", marginals, dit_constraints=constraints, dit_string_length=6, iteration_number=10)``
        - ``bind_matching_pursuit("abstract", marginals, dit_constraints=constraints, dit_string_length=6, iteration_number=10, interaction_size=2, dit_dimension=2)``
    
    Returns
    -------
    MatchingPursuit
        An instance of the requested matching-pursuit adapter with the specified default arguments.
    """
    ensure_nonempty_str("name", name)
    matching_pursuit = get_matching_pursuit(name)
    if not isinstance(matching_pursuit, FunctionMatchingPursuit):
        return matching_pursuit
    return matching_pursuit.with_defaults(*args, **kwargs)


def run_matching_pursuit(name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Convenience helper to run matching pursuit in one call.
    
    Parameters
    ----------
    name : str
        The name of the matching-pursuit backend to retrieve. Available backends can be listed with `list_matching_pursuits()`.
    *args : Any
        Positional arguments to pass to the matching-pursuit function.
    **kwargs : Any
        Keyword arguments to pass to the matching-pursuit function.
    
    Returns
    -------
    Any
        The result of the matching-pursuit function.
    """
    ensure_nonempty_str("name", name)
    return get_matching_pursuit(name).run(*args, **kwargs)


def matching_pursuit(problem_sketch: ProblemSketch, **kwargs: Any) -> MatchingPursuitResults:
    """
    Run matching pursuit to find a sparse reconstruction.

    Parameters
    ----------
    problem_sketch : ProblemSketch
        A ProblemSketch instance containing the problem configuration and data.
        The appropriate backend ("explicit" or "abstract") is selected
        automatically from ``problem_sketch.sketch_map``.
    **kwargs : Any
        Additional backend keyword arguments (for example ``iteration_number``,
        ``step`` or ``optimizer``).
    
    Returns
    -------
    MatchingPursuitResults
        Structured matching-pursuit result with positions, values, dit strings,
        and run metadata.
    
    Examples
    --------
    Using the ProblemSketch API:

    >>> from troma import CombinatorialProblem, ProblemSketch
    >>> from troma.sketch_map import ConstraintSketchMap
    >>> problem = CombinatorialProblem(objective_func, problem_size=3, problem_dimension=2)
    >>> problem.sampling(100)
    >>> sketch_map = ConstraintSketchMap(sketch_length=3)
    >>> sketch_map.build_from_nearest_neighbors(interaction_size=2)
    >>> problem_sketch = problem.sketching(sketch_map)
    >>> result = matching_pursuit(problem_sketch, iteration_number=10)
    """
    return _matching_pursuit_from_problem_sketch(problem_sketch, **kwargs)


def _matching_pursuit_from_problem_sketch(problem_sketch: ProblemSketch, **kwargs: Any) -> MatchingPursuitResults:
    """
    Internal helper to run matching pursuit using a ProblemSketch instance.
    
    Automatically selects the appropriate backend based on the sketch_map type
    and extracts all required parameters from the ProblemSketch.
    
    Parameters
    ----------
    problem_sketch : ProblemSketch
        The problem sketch containing all configuration and data.
    **kwargs : Any
        Additional parameters like iteration_number, step, optimizer, etc.
    
    Returns
    -------
    MatchingPursuitResults
        Structured matching-pursuit result.
    """
    ensure_instance("problem_sketch", problem_sketch, ProblemSketch)
    
    # Extract marginals from the problem sketch
    marginals = problem_sketch.sketch_values
    if not marginals:
        raise ValueError("ProblemSketch must have a sketch. Call problem.mcco_sketching() first.")
    
    # Determine backend type and extract parameters based on sketch_map type
    sketch_map = problem_sketch.sketch_map
    run_dit_string_length = problem_sketch.problem_size
    run_dit_dimension = problem_sketch.problem_dimension

    # Restricted sketches are solved in restricted coordinates first.
    if isinstance(problem_sketch, RestrictedProblemSketch):
        run_dit_string_length = problem_sketch.restricted_problem_size
        run_dit_dimension = problem_sketch.restricted_problem_dimension
    
    if isinstance(sketch_map, ConstraintSketchMap):
        # Use abstract backend with constraints
        backend_name = "abstract"
        backend_kwargs = {
            "marginals": marginals,
            "dit_constraints": sketch_map.map,
            "dit_string_length": run_dit_string_length,
            "dit_dimension": run_dit_dimension,
        }
        # Add optional parameters from sketch_map if available
        if sketch_map.interaction_size is not None:
            backend_kwargs["interaction_size"] = sketch_map.interaction_size
        interaction_size = sketch_map.interaction_size
    
    elif isinstance(sketch_map, ExplicitSketchMap):
        # Use explicit backend with sketch matrix
        backend_name = "explicit"
        backend_kwargs = {
            "marginals": marginals,
            "sketch": sketch_map.map,
        }
        interaction_size = sketch_map.interaction_size
    
    else:
        raise TypeError(f"Unsupported sketch_map type: {type(sketch_map).__name__}. "
                       "Must be ConstraintSketchMap or ExplicitSketchMap.")
    
    # Merge user-provided kwargs; explicit user values override inferred defaults.
    backend_kwargs.update(kwargs)
    
    raw_solution = run_matching_pursuit(backend_name, **backend_kwargs)
    result = _build_matching_pursuit_results(
        raw_solution,
        backend_name=backend_name,
        marginals=marginals,
        dit_string_length=run_dit_string_length,
        dit_dimension=run_dit_dimension,
        interaction_size=interaction_size,
    )

    if not isinstance(problem_sketch, RestrictedProblemSketch):
        return result

    restriction = problem_sketch.restriction
    mapped_dit_strings = reverse_spectrum_restriction(
        result.dit_strings,
        original_size=problem_sketch.problem_size,
        dit_restrictions=restriction.dit_restrictions,
        dit_value_restrictions=restriction.dit_value_restrictions,
        additional_dits_val=restriction.additional_dits_val,
    )
    mapped_positions = np.array(
        [s.to_integer() for s in mapped_dit_strings],
        dtype=int,
    )

    return MatchingPursuitResults(
        positions=mapped_positions,
        values=result.values,
        dit_strings=mapped_dit_strings,
        backend_name=result.backend_name,
        dit_string_length=problem_sketch.problem_size,
        dit_dimension=problem_sketch.problem_dimension,
        interaction_size=result.interaction_size,
        marginals=result.marginals,
        raw=result.raw,
    )
