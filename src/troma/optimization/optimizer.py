from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from importlib import import_module
from typing import Any, Callable
from .._validation import ensure_callable as _ensure_callable


OptimizerFunction = Callable[..., int]


class Optimizer(ABC):
    """Common interface for all optimization backends."""

    @abstractmethod
    def optimize(self, *args: Any, **kwargs: Any) -> int:
        """Run the optimization and return the best configuration index."""


class FunctionOptimizer(Optimizer):
    """Adapter exposing a plain optimization function through the Optimizer interface."""

    def __init__(
        self,
        name: str,
        function: OptimizerFunction,
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

    def optimize(self, *args: Any, **kwargs: Any) -> int:
        final_args = self._default_args + args
        final_kwargs = dict(self._default_kwargs)
        final_kwargs.update(kwargs)
        final_args, final_kwargs = self._prepare_call(final_args, final_kwargs)
        return int(self._function(*final_args, **final_kwargs))

    def with_defaults(self, *args: Any, **kwargs: Any) -> "FunctionOptimizer":
        """Return a new optimizer with additional default arguments pre-bound."""

        merged_kwargs = dict(self._default_kwargs)
        merged_kwargs.update(kwargs)
        return FunctionOptimizer(
            name=self.name,
            function=self._function,
            default_args=self._default_args + args,
            default_kwargs=merged_kwargs,
        )


_OPTIMIZER_REGISTRY: dict[str, tuple[str, str]] = {
    "brute_force_max": ("classical", "brute_force_max"),
    "spin_chain_nn_max": ("classical", "spin_chain_nn_max"),
    "dual_annealing": ("classical", "dual_annealing"),
    "simulated_annealing": ("classical", "simulated_annealing"),
    "digital_annealing": ("quantum", "digital_annealing"),
    "qaoa": ("quantum", "QAOA"),
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


def _resolve_optimizer_function(name: str) -> OptimizerFunction:
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string.")
    key = name.lower()
    if key not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. Available optimizers: {', '.join(list_optimizers())}."
        )

    module_name, function_name = _OPTIMIZER_REGISTRY[key]
    module = _load_module(module_name)
    function = getattr(module, function_name)
    return function


def list_optimizers() -> list[str]:
    """
    Return all registered optimizer names.
    
    Parameters
    ----------
    None

    Returns
    -------
    list[str]
        A list of all registered optimizer names.
    """

    return sorted(_OPTIMIZER_REGISTRY)


def get_optimizer(name: str) -> Optimizer:
    """
    Instantiate an optimizer adapter from a registered name.
    
    Parameters
    ----------
    name : str
        The name of the optimizer to retrieve. Available optimizers can be listed with `list_optimizers()`.
        optimizers()`.
    
    Returns
    -------
    Optimizer
        An instance of the requested optimizer.
    """
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string.")
    function = _resolve_optimizer_function(name)
    return FunctionOptimizer(name=name.lower(), function=function)


def bind_optimizer(name: str, *args: Any, **kwargs: Any) -> Optimizer:
    """
    Instantiate an optimizer with default arguments pre-bound.
    
    Parameters
    ----------
    name : str
        The name of the optimizer to retrieve. Available optimizers can be listed with `list_optimizers()`.
    *args : Any
        Positional arguments to pre-bind to the optimizer function.
    **kwargs : Any
        Keyword arguments to pre-bind to the optimizer function.
    
    Returns
    -------
    Optimizer
        An instance of the requested optimizer with the specified default arguments.
    """
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string.")
    optimizer = get_optimizer(name)
    if not isinstance(optimizer, FunctionOptimizer):
        return optimizer
    return optimizer.with_defaults(*args, **kwargs)


def optimize(name: str, *args: Any, **kwargs: Any) -> int:
    """
    Convenience helper to optimize in one call.
    
    Parameters    
    ----------
    name : str
        The name of the optimizer to retrieve. Available optimizers can be listed with `list_optimizers()`.
    *args : Any
        Positional arguments to pass to the optimizer function.
    **kwargs : Any
        Keyword arguments to pass to the optimizer function.
    
    Returns
    -------
    int
        The result of the optimization.
    """
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string.")
    return get_optimizer(name).optimize(*args, **kwargs)
