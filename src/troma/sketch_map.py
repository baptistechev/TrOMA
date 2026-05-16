"""
sketch_map.py
=============
Two concrete sketch representations sharing a common abstract interface:

- :class:`ConstraintSketchMap` — implicit description via dit constraints
- :class:`ExplicitSketchMap`   — explicit matrix stored in memory
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from numbers import Real
import enum
from typing import Any, Union

import numpy as np

from .core.structure import DitString, CylinderSet
from ._validation import (
    ensure_int,
    ensure_str,
    ensure_instance,
    ensure_sequence,
    ensure_dict,
)


class SketchType(enum.StrEnum):
    NEAREST_NEIGHBORS = "nearest_neighbors"
    ALL_INTERACTIONS = "all_interactions"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SketchMap(ABC):
    """Abstract interface for sketch representations."""

    def __init__(
        self,
        sketch_length: int,
        interaction_size: int | None = None,
        sketch_map: Any = None,
        sketch_dimension: int = 2,
        constraints: str | SketchType | None = None,
    ) -> None:
        self.map = sketch_map
        self.sketch_length = sketch_length
        self.interaction_size = interaction_size
        self.sketch_dimension = sketch_dimension
        if constraints is not None:
            constraints = SketchType(constraints)
            if constraints == SketchType.NEAREST_NEIGHBORS:
                self.build_from_nearest_neighbors()
            elif constraints == SketchType.ALL_INTERACTIONS:
                self.build_from_all_interactions()
            else:
                raise ValueError(f"Invalid constraints type: {constraints}")

    @abstractmethod
    def build_from_nearest_neighbors(self, interaction_size: int | None = None) -> None:
        """Build the sketch for nearest-neighbor interactions."""

    @abstractmethod
    def build_from_all_interactions(self, interaction_size: int | None = None) -> None:
        """Build the sketch for all (non-consecutive) interactions."""

    def use_custom_sketch(self, custom_sketch: Any) -> None:
        self.map = custom_sketch

    def set_interaction_size(self, interaction_size: int) -> None:
        self.interaction_size = interaction_size

    @abstractmethod
    def compute_marginal(self, function_data: Any) -> Any:
        """Compute marginals given function data and the sketch structure."""


# ---------------------------------------------------------------------------
# ConstraintSketchMap — implicit / sparse representation
# ---------------------------------------------------------------------------

class ConstraintSketchMap(SketchMap):
    """Sketch backed by dit constraints (implicit representation).

    The sketch is a ``list[dict[int, int]]`` — one sparse constraint
    ``{dit_index: dit_value}`` per row.  The function is provided as a sparse
    ``(input_dits, values)`` pair.
    """

    def __init__(
        self,
        sketch_length: int,
        interaction_size: int,
        sketch_map: Any = None,
        sketch_dimension: int = 2,
        constraints: str | SketchType | None = None,
    ) -> None:
        super().__init__(
            sketch_length=sketch_length,
            interaction_size=interaction_size,
            sketch_map=sketch_map,
            sketch_dimension=sketch_dimension,
            constraints=constraints,
        )

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def build_from_nearest_neighbors(self, interaction_size: int | None = None) -> None:
        """Build nearest-neighbor constraints and store them in ``self.map``."""
        if interaction_size is not None:
            self.interaction_size = interaction_size
        n = ensure_int("sketch_length", self.sketch_length, min_value=1)
        k = ensure_int("interaction_size", self.interaction_size, min_value=1)
        d = ensure_int("sketch_dimension", self.sketch_dimension, min_value=1)
        if k > n:
            raise ValueError("interaction_size must be <= sketch_length.")
        constraints: list[dict[int, int]] = []
        for window in [tuple(range(i, i + k)) for i in range(n - k + 1)]:
            for values in itertools.product(range(d), repeat=k):
                constraints.append(dict(zip(window, values)))
        self.map = constraints

    def build_from_all_interactions(self, interaction_size: int | None = None) -> None:
        """Build all-interaction constraints and store them in ``self.map``."""
        if interaction_size is not None:
            self.interaction_size = interaction_size
        n = ensure_int("sketch_length", self.sketch_length, min_value=1)
        k = ensure_int("interaction_size", self.interaction_size, min_value=1)
        d = ensure_int("sketch_dimension", self.sketch_dimension, min_value=1)
        if k > n:
            raise ValueError("interaction_size must be <= sketch_length.")
        constraints: list[dict[int, int]] = []
        for positions in itertools.combinations(range(n), k):
            for values in itertools.product(range(d), repeat=k):
                constraints.append(dict(zip(positions, values)))
        self.map = constraints

    # ------------------------------------------------------------------
    # Marginal computation
    # ------------------------------------------------------------------

    def compute_marginal(self, *args: Any) -> Union[float, list[float]]:
        """Compute marginals from sparse function data against the stored constraint map.

        Accepts either:
        - ``(function_data,)`` where *function_data* is ``(input_dits, values)``
        - ``(input_dits, values)``
        """
        if self.map is None:
            raise ValueError("Sketch map is not initialized. Build a sketch first.")
        if len(args) == 1:
            input_dits, values = args[0]
        elif len(args) == 2:
            input_dits, values = args
        else:
            raise TypeError(
                "compute_marginal expects either (function_data,) or (input_dits, values)."
            )
        return self._compute_sparse_marginal(input_dits, values, self.map)

    def _compute_sparse_marginal(
        self,
        function_input_dits: list[DitString],
        function_values: list[float] | np.ndarray,
        dit_constraints: Any,
    ) -> float | list[float]:
        """Internal: compute marginals for explicit constraints (not necessarily self.map)."""
        if not isinstance(function_input_dits, (list, tuple)):
            raise TypeError("function_input_dits must be a list of DitString instances.")
        for i, s in enumerate(function_input_dits):
            ensure_instance(f"function_input_dits[{i}]", s, DitString)
        if not isinstance(function_values, (list, tuple, np.ndarray)):
            raise TypeError("function_values must be a sequence of numeric values.")
        if len(function_input_dits) != len(function_values):
            raise ValueError("function_input_dits and function_values must have the same length.")

        if len(function_input_dits) == 0:
            if (
                isinstance(dit_constraints, (list, tuple, np.ndarray))
                and len(dit_constraints) > 0
                and not all(np.isscalar(x) for x in dit_constraints)
            ):
                return [0.0 for _ in dit_constraints]
            return 0.0

        for value in function_values:
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError("Values in function_values must be numeric.")

        dit_string_length = function_input_dits[0].length
        dit_dimension = function_input_dits[0].dimension
        if dit_string_length == 0:
            raise ValueError("Each DitString in function_input_dits must be non-empty.")

        def _single(constraint: Any) -> float:
            if isinstance(constraint, dict):
                self._validate_constraint_dict(constraint, dit_string_length, dit_dimension)
                items = list(constraint.items())
                filtered = [
                    v for s, v in zip(function_input_dits, function_values)
                    if all(int(s[idx]) == int(val) for idx, val in items)
                ]
                return float(np.sum(filtered) if filtered else 0.0)
            self._validate_full_assignment(constraint, dit_string_length, dit_dimension)
            dit_values = list(constraint)
            filtered = [
                v for s, v in zip(function_input_dits, function_values)
                if all(int(s[i]) == int(dv) for i, dv in enumerate(dit_values))
            ]
            return float(np.sum(filtered) if filtered else 0.0)

        if isinstance(dit_constraints, dict):
            return _single(dit_constraints)
        if isinstance(dit_constraints, np.ndarray):
            return _single(dit_constraints.tolist()) if dit_constraints.ndim == 1 \
                else [_single(c) for c in dit_constraints.tolist()]
        if isinstance(dit_constraints, (list, tuple)):
            if not dit_constraints:
                return []
            if all(np.isscalar(x) for x in dit_constraints):
                return _single(dit_constraints)
            return [_single(c) for c in dit_constraints]
        raise TypeError(
            "dit_constraints must be a dict, a full dit assignment, or a list of constraints."
        )

    def _validate_constraint_dict(
        self, constraint: dict, dit_string_length: int, dit_dimension: int
    ) -> None:
        ensure_dict("constraint", constraint)
        for dit_idx, dit_val in constraint.items():
            ensure_int("constraint index", dit_idx)
            if int(dit_idx) < 0 or int(dit_idx) >= dit_string_length:
                raise ValueError("Constraint index out of range for dit string length.")
            ensure_int("constraint value", dit_val)
            if int(dit_val) < 0 or int(dit_val) >= dit_dimension:
                raise ValueError("Constraint values must be in [0, dit_dimension - 1].")

    def _validate_full_assignment(
        self, constraint: Any, dit_string_length: int, dit_dimension: int
    ) -> None:
        if not isinstance(constraint, (DitString, list, tuple, np.ndarray)):
            raise TypeError("A full dit assignment must be a DitString or a sequence of integers.")
        if len(constraint) != dit_string_length:
            raise ValueError("A full dit assignment must have length dit_string_length.")
        for dit_val in constraint:
            ensure_int("full-assignment value", dit_val)
            if int(dit_val) < 0 or int(dit_val) >= dit_dimension:
                raise ValueError("Full-assignment values must be in [0, dit_dimension - 1].")

    # ------------------------------------------------------------------
    # Column reconstruction
    # ------------------------------------------------------------------

    def reconstruct_structured_matrix_column(self, index: int) -> np.ndarray:
        """Reconstruct a column of the sketch matrix for a given integer index."""
        if self.map is None:
            raise ValueError("Sketch map is not initialized. Build a sketch first.")

        index = ensure_int("index", index, min_value=0)
        dit_string_length = ensure_int("sketch_length", self.sketch_length, min_value=1)
        dit_dimension = ensure_int("sketch_dimension", self.sketch_dimension, min_value=1)
        dit_constraints = ensure_sequence("dit_constraints", self.map)
        if index >= dit_dimension ** dit_string_length:
            raise ValueError("index out of range for provided dit_string_length and dit_dimension.")

        dit_str = DitString.from_integer(index, dit_string_length, dit_dimension)
        column: list[int] = []
        for constraint in dit_constraints:
            for pos, val in enumerate(dit_str):
                if pos in constraint and int(val) != constraint[pos]:
                    column.append(0)
                    break
            else:
                column.append(1)
        return np.array(column)


# ---------------------------------------------------------------------------
# ExplicitSketchMap — explicit matrix representation
# ---------------------------------------------------------------------------

class ExplicitSketchMap(SketchMap):
    """Sketch backed by an explicit matrix stored in memory.

    The sketch is an ``np.ndarray`` whose rows are Kronecker-developed
    cylinder-set indicator vectors.  The function must be given as a dense
    array of values over the full dit-string spectrum.
    """

    def build_from_nearest_neighbors(self, interaction_size: int | None = None) -> None:
        """Build nearest-neighbor sketch matrix and store it in ``self.map``."""
        if interaction_size is not None:
            self.interaction_size = interaction_size
        if self.interaction_size is None:
            raise ValueError(
                "interaction_size is required. Set it on the instance before calling this method."
            )
        n = ensure_int("sketch_length", self.sketch_length, min_value=1)
        k = ensure_int("interaction_size", self.interaction_size, min_value=1)
        d = ensure_int("sketch_dimension", self.sketch_dimension, min_value=2)
        if k > n:
            raise ValueError("interaction_size cannot be greater than sketch_length.")
        cylinder_sets: list[CylinderSet] = []
        for window in [tuple(range(i, i + k)) for i in range(n - k + 1)]:
            cylinder_sets += CylinderSet.for_positions(window, n, d)
        self.map = np.array([cs.kronecker_develop() for cs in cylinder_sets])

    def build_from_all_interactions(self, interaction_size: int | None = None) -> None:
        """Build all-interactions sketch matrix and store it in ``self.map``."""
        if interaction_size is not None:
            self.interaction_size = interaction_size
        if self.interaction_size is None:
            raise ValueError(
                "interaction_size is required. Set it on the instance before calling this method."
            )
        n = ensure_int("sketch_length", self.sketch_length, min_value=1)
        k = ensure_int("interaction_size", self.interaction_size, min_value=1)
        d = ensure_int("sketch_dimension", self.sketch_dimension, min_value=2)
        if k > n:
            raise ValueError("interaction_size cannot be greater than sketch_length.")
        cylinder_sets: list[CylinderSet] = []
        for positions in itertools.combinations(range(n), k):
            cylinder_sets += CylinderSet.for_positions(positions, n, d)
        self.map = np.array([cs.kronecker_develop() for cs in cylinder_sets])

    def compute_marginal(
        self,
        function_data: list[float] | np.ndarray,
    ) -> list[float]:
        """Compute marginals via matrix multiplication against the full-spectrum value vector.

        Parameters
        ----------
        function_data : list of float or np.ndarray
            One value per dit string in lexicographic order (full spectrum).
        """
        if self.map is None:
            raise ValueError("Sketch map is not initialized. Build a sketch first.")
        if not isinstance(function_data, (list, tuple, np.ndarray)):
            raise TypeError("function_data must be a sequence of numeric values.")
        if not hasattr(self.map, "__matmul__"):
            raise TypeError("sketch map must support matrix multiplication (@).")
        values = np.asarray(function_data)
        if hasattr(self.map, "shape") and len(self.map.shape) >= 2:
            if self.map.shape[1] != values.shape[0]:
                raise ValueError(
                    "function_data length must match sketch map column count."
                )
        return np.asarray(self.map @ values).flatten().tolist()

    def random_sketch(
        self,
        size: int,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate and store a random Gaussian sketch matrix (i.i.d. N(0, 1/size) entries).

        Parameters
        ----------
        size : int
            Number of measurement rows.
        random_state : int, np.random.Generator, or None, optional
            Seed or Generator for reproducibility.
        """
        n = ensure_int("sketch_length", self.sketch_length, min_value=1)
        m = ensure_int("size", size, min_value=1)
        d = ensure_int("sketch_dimension", self.sketch_dimension, min_value=2)
        cols = d ** n
        rng = random_state if isinstance(random_state, np.random.Generator) \
            else np.random.default_rng(random_state)
        self.map = rng.standard_normal((m, cols)) / np.sqrt(m)
        return self.map
