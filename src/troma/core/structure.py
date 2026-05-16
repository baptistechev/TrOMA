from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class DitString:
    """A validated dit string bundled with its length and dimension metadata.

    Attributes
    ----------
    length : int
        Number of dits.
    dimension : int
        Number of possible values per dit (base of the number system).
    dit_string : list[int]
        The actual dit values, each in ``[0, dimension - 1]``.
    """

    __slots__ = ("length", "dimension", "dit_string")

    def __init__(
        self,
        dit_string: Iterable[int],
        length: int | None = None,
        dimension: int = 2,
    ) -> None:
        from .._validation import ensure_int
        values = [ensure_int(f"dit_string element", v) for v in dit_string]
        self.length: int = len(values) if length is None else ensure_int("length", length, min_value=0)
        self.dimension: int = ensure_int("dimension", dimension, min_value=2)
        if length is not None and self.length != len(values):
            raise ValueError(
                f"length ({length}) must match the number of values provided ({len(values)})."
            )
        for i, value in enumerate(values):
            if value < 0 or value >= self.dimension:
                raise ValueError(
                    f"Each dit value must be in [0, dimension-1]. "
                    f"Got {value} at index {i} with dimension={self.dimension}."
                )
        self.dit_string: list[int] = values

    # ------------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------------

    def to_integer(self, convention: str = "R") -> int:
        """Convert this dit string to its integer index.

        Parameters
        ----------
        convention : str, optional
            ``"R"`` treats the first dit as most significant (default).
            ``"L"`` treats the first dit as least significant.

        Returns
        -------
        int
            The integer representation of this dit string.
        """
        if self.length == 0:
            return 0
        basis = np.power(self.dimension, np.arange(self.length), dtype=float)
        if convention == "R":
            basis = basis[::-1]
        return int(np.dot(np.array(self.dit_string, dtype=int), basis))

    @classmethod
    def from_integer(
        cls,
        integer: int,
        length: int,
        dimension: int = 2,
        convention: str = "R",
    ) -> DitString:
        """Build a DitString from an integer index.

        Parameters
        ----------
        integer : int
            Non-negative integer to encode.
        length : int
            Number of dits in the output string.
        dimension : int, optional
            Number of possible values per dit. Default is 2 (binary).
        convention : str, optional
            ``"R"`` puts the most-significant dit first (default).
            ``"L"`` puts the least-significant dit first.

        Returns
        -------
        DitString
            The encoded dit string.
        """
        from .._validation import ensure_int
        integer = ensure_int("integer", integer, min_value=0)
        length = ensure_int("length", length, min_value=0)
        dimension = ensure_int("dimension", dimension, min_value=2)
        values: list[int] = []
        current = integer
        while current > 0:
            current, rem = divmod(current, dimension)
            values.append(rem)
        values += [0] * (length - len(values))
        if convention == "R":
            values = values[::-1]
        return cls(values, length=length, dimension=dimension)

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[int]:
        return iter(self.dit_string)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return self.dit_string[index]

    # ------------------------------------------------------------------
    # Numpy interoperability
    # ------------------------------------------------------------------

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.dit_string, dtype=dtype)

    def tolist(self) -> list[int]:
        """Return the dit string as a plain Python list."""
        return list(self.dit_string)

    def to_computational_basis(self) -> list[list[int]]:
        """Convert to one-hot computational basis vectors.

        Returns
        -------
        list[list[int]]
            A list of ``dimension``-length one-hot vectors, one per dit.
        """
        result: list[list[int]] = []
        for value in self.dit_string:
            vec = [0] * self.dimension
            vec[value] = 1
            result.append(vec)
        return result

    # ------------------------------------------------------------------
    # Hashing and equality (needed for use as dict keys)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DitString):
            return self.dit_string == other.dit_string and self.dimension == other.dimension
        return NotImplemented

    def __hash__(self) -> int:
        return hash((tuple(self.dit_string), self.dimension))

    def __repr__(self) -> str:
        return f"DitString({self.dit_string}, dimension={self.dimension})"


@dataclass
class Sample:
    indexes: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dit_strings: list[DitString] = field(default_factory=list)


@dataclass
class Restriction:
    dit_restrictions: list[int] | None = None
    dit_value_restrictions: list[int] | None = None
    additional_dits_val: int = 0

    def __init__(
        self,
        dit_restrictions: list[int] | None = None,
        dit_value_restrictions: list[int] | None = None,
        additional_dits_val: int = 0,
    ) -> None:
        self.dit_restrictions = dit_restrictions
        self.dit_value_restrictions = dit_value_restrictions
        self.additional_dits_val = additional_dits_val
