from __future__ import annotations

import numbers
from collections.abc import Iterable
from typing import Any

import numpy as np


class _Validator:
    """ A collection of static methods for validating inputs across the TROMA library. Each method checks a specific type or property and raises informative exceptions when validation fails. This centralizes all input validation logic, making it easier to maintain and ensuring consistent error handling throughout the codebase.
    """

    @staticmethod
    def ensure_int(
        name: str,
        value: Any,
        *,
        min_value: int | None = None,
        max_value: int | None = None,
        allow_bool: bool = False,
    ) -> int:
        if not isinstance(value, numbers.Integral) or (not allow_bool and isinstance(value, bool)):
            raise TypeError(f"{name} must be an integer.")

        int_value = int(value)
        if min_value is not None and int_value < min_value:
            raise ValueError(f"{name} must be >= {min_value}.")
        if max_value is not None and int_value > max_value:
            raise ValueError(f"{name} must be <= {max_value}.")
        return int_value

    @staticmethod
    def ensure_real(name: str, value: Any, *, allow_bool: bool = False) -> float:
        if not isinstance(value, numbers.Real) or (not allow_bool and isinstance(value, bool)):
            raise TypeError(f"{name} must be a real number.")
        return float(value)


    @staticmethod 
    def ensure_callable(name: str, value: Any) -> None:
        if not callable(value):
            raise TypeError(f"{name} must be callable.")

    @staticmethod
    def ensure_str(name: str, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")
        return value

    @staticmethod
    def ensure_nonempty_str(name: str, value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must be a non-empty string.")
        return value

    @staticmethod
    def ensure_dict(name: str, value: Any) -> dict:
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a dict.")
        return value

    @staticmethod
    def ensure_tuple(name: str, value: Any) -> tuple:
        if not isinstance(value, tuple):
            raise TypeError(f"{name} must be a tuple.")
        return value

    @staticmethod
    def ensure_sequence(name: str, value: Any) -> list:
        """Validate that value is a list, tuple, or ndarray and return as list."""
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TypeError(f"{name} must be a sequence (list, tuple, or array).")
        return list(value)


    @staticmethod
    def ensure_instance(name: str, value: Any, expected_type: type | tuple) -> Any:
        """Validate that value is an instance of expected_type and return it."""
        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                type_names = " or ".join(t.__name__ for t in expected_type)
            else:
                type_names = expected_type.__name__
            raise TypeError(f"{name} must be an instance of {type_names}.")
        return value


    @staticmethod
    def ensure_iterable(name: str, value: Any, *, allow_str: bool = False) -> None:
        if isinstance(value, str) and not allow_str:
            raise TypeError(f"{name} must be an iterable (non-string).")
        if not isinstance(value, Iterable):
            raise TypeError(f"{name} must be iterable.")


    @staticmethod
    def ensure_one_of(name: str, value: Any, allowed_values: set[Any] | tuple[Any, ...] | list[Any]) -> None:
        if value not in allowed_values:
            allowed = ", ".join(repr(v) for v in allowed_values)
            raise ValueError(f"{name} must be one of: {allowed}.")


    @staticmethod
    def ensure_optional_int(name: str, value: Any, *, min_value: int | None = None) -> int | None:
        if value is None:
            return None
        return _Validator.ensure_int(name, value, min_value=min_value)


    @staticmethod
    def ensure_optional_dict(name: str, value: Any) -> dict | None:
        if value is None:
            return None
        return _Validator.ensure_dict(name, value)

    @staticmethod
    def ensure_optional_real(name: str, value: Any, *, allow_bool: bool = False) -> float | None:
        if value is None:
            return None
        return _Validator.ensure_real(name, value, allow_bool=allow_bool)


    @staticmethod
    def ensure_int_or_digit(
        name: str,
        value: Any,
        *,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int:
        if isinstance(value, str):
            if not value.isdigit():
                raise ValueError(f"{name} string values must contain only digits.")
            value = int(value)
        return _Validator.ensure_int(name, value, min_value=min_value, max_value=max_value)


    @staticmethod
    def ensure_unique_items(name: str, values: Any) -> None:
        _Validator.ensure_iterable(name, values)
        values = list(values)
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique items.")


    @staticmethod
    def ensure_same_length(name_a: str, values_a: Any, name_b: str, values_b: Any) -> None:
        _Validator.ensure_iterable(name_a, values_a)
        _Validator.ensure_iterable(name_b, values_b)
        if len(values_a) != len(values_b):
            raise ValueError(f"{name_a} and {name_b} must have the same length.")


    @staticmethod
    def ensure_vector_collection(
        name: str,
        vectors: Any,
        *,
        vector_length: int | None = None,
        binary: bool = False,
    ) -> list:
        _Validator.ensure_iterable(name, vectors)
        vectors_list = list(vectors)
        for vector in vectors_list:
            if not hasattr(vector, "__len__"):
                raise TypeError(f"Each element of {name} must be a vector-like object.")
            if vector_length is not None and len(vector) != vector_length:
                raise ValueError(f"Each vector in {name} must have length equal to {vector_length}.")
            if binary and any(value not in (0, 1) for value in vector):
                raise ValueError(f"Each vector in {name} must contain only 0 or 1 values.")
        return vectors_list

    @staticmethod
    def ensure_not_none(name: str, value: Any) -> Any:
        if value is None:
            raise ValueError(f"{name} must not be None.")
        return value