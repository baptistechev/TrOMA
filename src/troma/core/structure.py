from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._validation import ensure_int, ensure_str, ensure_unique_items


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
        values = [ensure_int("dit_string element", v) for v in dit_string]
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
            ``"R"`` — most-significant dit first (default).
            ``"L"`` — least-significant dit first.

        Returns
        -------
        int
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
            Dit base. Default is 2.
        convention : str, optional
            ``"R"`` — most-significant dit first (default).
            ``"L"`` — least-significant dit first.

        Returns
        -------
        DitString
        """
        integer = ensure_int("integer", integer, min_value=0)
        length = ensure_int("length", length, min_value=0)
        dimension = ensure_int("dimension", dimension, min_value=2)
        if integer > 0 and length == 0:
            raise ValueError("length must be positive to encode a non-zero integer.")
        if length > 0 and integer >= dimension ** length:
            raise ValueError(
                "integer cannot be represented with the given dimension and length."
            )
        values: list[int] = []
        current = integer
        while current > 0:
            current, rem = divmod(current, dimension)
            values.append(rem)
        values += [0] * (length - len(values))
        if convention == "R":
            values = values[::-1]
        return cls(values, length=length, dimension=dimension)

    def to_computational_basis(self) -> CylinderSet:
        """Convert to a :class:`CylinderSet` where every position is fixed to its dit value.

        Returns
        -------
        CylinderSet
            A cylinder set with no wildcard positions — each position has a
            one-hot vector pinned to the corresponding dit value.
        """
        vecs: list[list[int]] = []
        for value in self.dit_string:
            vec = [0] * self.dimension
            vec[value] = 1
            vecs.append(vec)
        return CylinderSet(vecs, self.dimension)

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


class CylinderSet:
    """A factorized cylinder-set indicator over a dit string.

    A cylinder set is described by one local binary vector per dit position.
    Each vector has length ``dimension``:

    * ``[1, 0, ..., 0]`` — dit must equal 0
    * ``[0, 1, ..., 0]`` — dit must equal 1
    * ``[1, 1, ..., 1]`` — dit is unconstrained (wildcard)

    Attributes
    ----------
    vectors : list[list[int]]
        The local binary indicator vectors, one per dit position.
    dimension : int
        Dit base — length of each local vector.
    length : int
        Number of dit positions.
    """

    __slots__ = ("vectors", "dimension", "length")

    def __init__(self, vectors: list[list[int]], dimension: int) -> None:
        dimension = ensure_int("dimension", dimension, min_value=2)
        if not isinstance(vectors, (list, tuple)):
            raise TypeError("vectors must be a list of binary indicator vectors.")
        vecs = [list(v) for v in vectors]
        for i, vec in enumerate(vecs):
            if len(vec) != dimension:
                raise ValueError(
                    f"Each vector must have length equal to dimension ({dimension}). "
                    f"Got length {len(vec)} at index {i}."
                )
            if any(v not in (0, 1) for v in vec):
                raise ValueError(
                    f"Each vector must be binary (0 or 1). Got {vec} at index {i}."
                )
        self.vectors: list[list[int]] = vecs
        self.dimension: int = dimension
        self.length: int = len(vecs)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_positions(
        cls,
        fixed_positions: Iterable[int],
        set_size: int,
        dimension: int = 2,
    ) -> list[CylinderSet]:
        """Generate all cylinder sets for a given set of fixed dit positions.

        Each returned :class:`CylinderSet` fixes the specified positions to one
        particular combination of values and leaves all other positions as wildcards.
        The number of returned sets equals ``dimension ** len(fixed_positions)``.

        Parameters
        ----------
        fixed_positions : Iterable[int]
            Indices of the dit positions to constrain.
        set_size : int
            Total number of dit positions in the string.
        dimension : int, optional
            Dit base. Default is 2.

        Returns
        -------
        list[CylinderSet]
            One cylinder set per possible assignment of the fixed positions.

        Example
        -------
        >>> sets = CylinderSet.for_positions([1], set_size=3, dimension=2)
        >>> # Returns two sets: one with pos-1 fixed to 0, one with pos-1 fixed to 1.
        """
        set_size = ensure_int("set_size", set_size, min_value=0)
        dimension = ensure_int("dimension", dimension, min_value=2)
        fixed_pos_list = [ensure_int("fixed position", p, min_value=0) for p in fixed_positions]
        ensure_unique_items("fixed_positions", fixed_pos_list)
        for p in fixed_pos_list:
            if p >= set_size:
                raise ValueError(
                    f"Position {p} is out of range for set_size={set_size}."
                )

        result: list[CylinderSet] = []
        n_fixed = len(fixed_pos_list)
        for config in range(dimension ** n_fixed):
            dits = DitString.from_integer(config, n_fixed, dimension)
            vecs = [[1] * dimension for _ in range(set_size)]
            for pos, dit_val in zip(fixed_pos_list, dits):
                cp_vec = [0] * dimension
                cp_vec[int(dit_val)] = 1
                vecs[pos] = cp_vec
            result.append(cls(vecs, dimension))
        return result

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def kronecker_develop(self, convention: str = "R") -> np.ndarray:
        """Expand this cylinder-set indicator via Kronecker products.

        Parameters
        ----------
        convention : str, optional
            ``"R"`` — outermost product first, i.e. position 0 is most
            significant (default). ``"L"`` — position 0 is least significant.

        Returns
        -------
        np.ndarray
            Full-space indicator vector of length ``dimension ** length``.
        """
        ensure_str("convention", convention)
        if convention not in ("R", "L"):
            raise ValueError("convention must be 'R' or 'L'.")
        vecs = self.vectors[::-1] if convention == "L" else self.vectors
        developed = np.array([1], dtype=int)
        for vec in vecs:
            developed = np.kron(developed, np.array(vec, dtype=int))
        return developed

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.vectors)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> list[int]:
        return self.vectors[index]

    # ------------------------------------------------------------------
    # Equality and representation
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CylinderSet):
            return self.vectors == other.vectors and self.dimension == other.dimension
        return NotImplemented

    def __repr__(self) -> str:
        return f"CylinderSet({self.vectors}, dimension={self.dimension})"


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
