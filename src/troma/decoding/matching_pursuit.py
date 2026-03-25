from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from importlib import import_module
from typing import Any, Callable
from .._validation import ensure_callable as _ensure_callable


MatchingPursuitFunction = Callable[..., Any]


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
        if not isinstance(name, str) or not name:
            raise TypeError("name must be a non-empty string.")
        _ensure_callable("function", function)
        if not isinstance(default_args, tuple):
            raise TypeError("default_args must be a tuple.")
        if default_kwargs is not None and not isinstance(default_kwargs, dict):
            raise TypeError("default_kwargs must be a dict or None.")
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
    "explicit": ("decoding_proced", "matchingpursuit_explicit"),
    "abstract": ("decoding_proced", "matchingpursuit_abstract"),
}


def _load_module(module_name: str):
    if not isinstance(module_name, str) or not module_name:
        raise TypeError("module_name must be a non-empty string.")
    if __package__:
        try:
            return import_module(f".{module_name}", package=__package__)
        except Exception:
            pass
    return import_module(module_name)


def _resolve_matching_pursuit_function(name: str) -> MatchingPursuitFunction:
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string.")
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
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string.")
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
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string.")
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
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string.")
    return get_matching_pursuit(name).run(*args, **kwargs)


def matching_pursuit(name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Alias of :func:`run_matching_pursuit`.
    
    Parameters
    ----------
    name : str
        The name of the matching-pursuit backend to retrieve. Available backends can be listed with `list_matching_pursuits()`.
    *args : Any
        Positional arguments for the selected backend. In practice this is often
        ``(marginals,)``.
    **kwargs : Any
        Keyword arguments for the selected backend. Common usable combinations are:

        - ``matching_pursuit("explicit", marginals, sketch=sketch, iteration_number=10)``
        - ``matching_pursuit("explicit", marginals, sketch=sketch, iteration_number=10, step=0.2)``
        - ``matching_pursuit("abstract", marginals, dit_constraints=constraints, dit_string_length=6, iteration_number=10)``
        - ``matching_pursuit("abstract", marginals, dit_constraints=constraints, dit_string_length=6, iteration_number=10, interaction_size=2, dit_dimension=2)``
    
    Returns
    -------
    Any
        The result of the matching-pursuit function.
    """
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string.")
    return run_matching_pursuit(name, *args, **kwargs)
