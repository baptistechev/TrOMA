from __future__ import annotations

import numbers
from collections.abc import Iterable
from typing import Any


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


def ensure_real(name: str, value: Any, *, allow_bool: bool = False) -> float:
    if not isinstance(value, numbers.Real) or (not allow_bool and isinstance(value, bool)):
        raise TypeError(f"{name} must be a real number.")
    return float(value)


def ensure_callable(name: str, value: Any) -> None:
    if not callable(value):
        raise TypeError(f"{name} must be callable.")


def ensure_iterable(name: str, value: Any, *, allow_str: bool = False) -> None:
    if isinstance(value, str) and not allow_str:
        raise TypeError(f"{name} must be an iterable (non-string).")
    if not isinstance(value, Iterable):
        raise TypeError(f"{name} must be iterable.")


def ensure_one_of(name: str, value: Any, allowed_values: set[Any] | tuple[Any, ...] | list[Any]) -> None:
    if value not in allowed_values:
        allowed = ", ".join(repr(v) for v in allowed_values)
        raise ValueError(f"{name} must be one of: {allowed}.")


def ensure_optional_int(name: str, value: Any, *, min_value: int | None = None) -> int | None:
    if value is None:
        return None
    return ensure_int(name, value, min_value=min_value)


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
    return ensure_int(name, value, min_value=min_value, max_value=max_value)


def ensure_unique_items(name: str, values: Any) -> None:
    ensure_iterable(name, values)
    values = list(values)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique items.")


def ensure_same_length(name_a: str, values_a: Any, name_b: str, values_b: Any) -> None:
    ensure_iterable(name_a, values_a)
    ensure_iterable(name_b, values_b)
    if len(values_a) != len(values_b):
        raise ValueError(f"{name_a} and {name_b} must have the same length.")


def ensure_vector_collection(
    name: str,
    vectors: Any,
    *,
    vector_length: int | None = None,
    binary: bool = False,
) -> list:
    ensure_iterable(name, vectors)
    vectors_list = list(vectors)
    for vector in vectors_list:
        if not hasattr(vector, "__len__"):
            raise TypeError(f"Each element of {name} must be a vector-like object.")
        if vector_length is not None and len(vector) != vector_length:
            raise ValueError(f"Each vector in {name} must have length equal to {vector_length}.")
        if binary and any(value not in (0, 1) for value in vector):
            raise ValueError(f"Each vector in {name} must contain only 0 or 1 values.")
    return vectors_list
