from __future__ import annotations
from typing import Any
import numpy as np

# --- MatchingPursuitResults (moved from matching_pursuit.py) ---
from dataclasses import dataclass

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

from collections.abc import Iterable, Iterator
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._validation import _Validator


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
        values = [_Validator.ensure_int("dit_string element", v) for v in dit_string]
        self.length: int = len(values) if length is None else _Validator.ensure_int("length", length, min_value=0)
        self.dimension: int = _Validator.ensure_int("dimension", dimension, min_value=2)
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
        integer = _Validator.ensure_int("integer", integer, min_value=0)
        length = _Validator.ensure_int("length", length, min_value=0)
        dimension = _Validator.ensure_int("dimension", dimension, min_value=2)
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
        dimension = _Validator.ensure_int("dimension", dimension, min_value=2)
        _Validator.ensure_instance("vectors", vectors, (list, tuple))
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
        set_size = _Validator.ensure_int("set_size", set_size, min_value=0)
        dimension = _Validator.ensure_int("dimension", dimension, min_value=2)
        fixed_pos_list = [_Validator.ensure_int("fixed position", p, min_value=0) for p in fixed_positions]
        _Validator.ensure_unique_items("fixed_positions", fixed_pos_list)
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
        _Validator.ensure_str("convention", convention)
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
class Hamiltonian:
    """Ising-Z Hamiltonian container.

    Attributes
    ----------
    terms : dict[tuple[int, ...], float]
        Mapping from Z-term support to coefficient.
    num_qubits : int
        Number of qubits for this Hamiltonian.
    """

    terms: dict[tuple[int, ...], float] = field(default_factory=dict)
    num_qubits: int = 1

    def __post_init__(self) -> None:
        self.num_qubits = _Validator.ensure_int("num_qubits", self.num_qubits, min_value=1)
        _Validator.ensure_dict("terms", self.terms)

        validated_terms: dict[tuple[int, ...], float] = {}
        for raw_term, raw_coeff in self.terms.items():
            _Validator.ensure_tuple("Hamiltonian term key", raw_term)
            term: tuple[int, ...] = tuple(_Validator.ensure_int("qubit index", q, min_value=0) for q in raw_term)
            for q in term:
                if q >= self.num_qubits:
                    raise ValueError(
                        f"Qubit index {q} is out of range for num_qubits={self.num_qubits}."
                    )
            validated_terms[term] = float(raw_coeff)
        self.terms = validated_terms

    #Need to be removed after changing the optimizations method to take a ProblemSketch
    @classmethod
    def from_constraints(
        cls,
        constraints_sketch: list,
        marginals: list[float] | np.ndarray,
        bit_string_length: int,
    ) -> "Hamiltonian":
        """Build a Hamiltonian from constraints and corresponding marginals."""
        bit_string_length = _Validator.ensure_int("bit_string_length", bit_string_length, min_value=1)
        marginals = list(marginals)
        if len(constraints_sketch) != len(marginals):
            raise ValueError("constraints_sketch and marginals must have the same length.")

        coeffs: dict[tuple[int, ...], float] = defaultdict(float)
        for constraint, yi in zip(constraints_sketch, marginals):
            weight = float(yi[0] if np.ndim(yi) > 0 else yi)
            if np.isclose(weight, 0.0):
                continue

            fixed: list[tuple[int, int]] = []
            if isinstance(constraint, dict):
                for raw_idx, bit in constraint.items():
                    idx = int(raw_idx)
                    if idx < 0 or idx >= bit_string_length:
                        raise ValueError(
                            f"Qubit index {idx} out of range for n_qubits={bit_string_length}."
                        )
                    if bit == 0:
                        fixed.append((idx, +1))
                    elif bit == 1:
                        fixed.append((idx, -1))
                    else:
                        raise ValueError(
                            f"Bit value must be 0 or 1, got {bit} at position {idx}."
                        )
                fixed.sort()
            else:
                if len(constraint) != bit_string_length:
                    raise ValueError(
                        f"Pattern length {len(constraint)} does not match n_qubits={bit_string_length}."
                    )
                for idx, local_state in enumerate(constraint):
                    if local_state == [1, 0]:
                        fixed.append((idx, +1))
                    elif local_state == [0, 1]:
                        fixed.append((idx, -1))
                    elif local_state == [1, 1]:
                        continue
                    else:
                        raise ValueError(f"Invalid local pattern at qubit {idx}: {local_state}")

            k = len(fixed)
            base_coeff = weight / (2 ** k)
            for mask in range(1 << k):
                z_idx: list[int] = []
                sign = 1.0
                for bit_pos, (qubit_idx, local_sign) in enumerate(fixed):
                    if (mask >> bit_pos) & 1:
                        z_idx.append(qubit_idx)
                        sign *= local_sign
                coeffs[tuple(z_idx)] += base_coeff * sign

        return cls(
            terms={term: coef for term, coef in coeffs.items() if not np.isclose(coef, 0.0)},
            num_qubits=bit_string_length,
        )

    @classmethod
    def from_problem_sketch(cls, problem_sketch: Any) -> "Hamiltonian":
        """Build a Hamiltonian from a ProblemSketch instance."""
        from ..problem_sketch import ProblemSketch, RestrictedProblemSketch
        from ..sketch_map import ConstraintSketchMap

        _Validator.ensure_instance("problem_sketch", problem_sketch, ProblemSketch)
        _Validator.ensure_instance("problem_sketch.sketch_map", problem_sketch.sketch_map, ConstraintSketchMap)
        if not problem_sketch.sketch_values:
            raise ValueError("problem_sketch.sketch_values is empty. Build sketch values first.")

        bit_string_length = int(problem_sketch.problem_size)
        if isinstance(problem_sketch, RestrictedProblemSketch):
            bit_string_length = int(problem_sketch.restricted_problem_size)

        return cls.from_constraints(
            constraints_sketch=problem_sketch.sketch_map.map,
            marginals=problem_sketch.sketch_values,
            bit_string_length=bit_string_length,
        )


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
