"""
sketch_base.py
==============
Common interface grouping the two sketch representations:

- :class:`ConstraintSketch`  — matrix described implicitly via dit constraints
  (memory-efficient, no need to materialise the full matrix).
- :class:`ExplicitSketch`    — matrix stored explicitly in memory
  (fast repeated computation once the matrix has been built).

Both classes expose the same method names so they can be used
interchangeably:

.. code-block:: python

    from sketch_base import ConstraintSketch, ExplicitSketch

    # Build the sketch once (stored in .map)
    c = ConstraintSketch(sketch_length=n, interaction_size=k, sketch_dimension=d)
    c.build_from_nearest_neighbors()
    # Or
    e = ExplicitSketch(sketch_length=n, interaction_size=k, sketch_dimension=d)
    e.build_from_nearest_neighbors()

    # Then compute marginals with the matching instance
    result = c.compute_marginal((input_dits, values))
    result = e.compute_marginal(full_spectrum_values)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import enum
from typing import Any, Union

import numpy as np

from .sketchs.abstract import (
    compute_marginal as _constraint_compute_marginal,
    constraints_for_nearest_neighbors_interactions,
    constraints_for_all_interactions,
    reconstruct_structured_matrix_column as _reconstruct_structured_matrix_column,
)
from .sketchs.explicit import (
    compute_marginal as _explicit_compute_marginal,
    nearest_neighbors_interactions_sketch,
    all_interactions_sketch,
    random_sketch as _random_sketch,
)

class SketchType(enum.StrEnum):
    NEAREST_NEIGHBORS = "nearest_neighbors"
    ALL_INTERACTIONS = "all_interactions"

# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SketchMap(ABC):
    """Abstract interface for sketcher.

    A *sketch* is a compact representation of the structured matrix used to
    compute marginals of a function defined on the full dit-string spectrum.
    Concrete subclasses differ in how that matrix is stored / described:

    * :class:`ConstraintSketchMap`  — implicit description via dit constraints.
    * :class:`ExplicitSketchMap`    — explicit ``np.ndarray`` stored in memory.

    The sketch is stored in the ``map`` instance attribute.
    """

    # ------------------------------------------------------------------
    # Sketch construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        sketch_length: int,
        interaction_size: int | None = None,
        sketch_map=None,
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
    def build_from_nearest_neighbors(
        self,
        interaction_size: int | None = None,
    ) -> None:
        """Build the sketch structure for nearest-neighbor interactions.

        Parameters
        ----------
        interaction_size : int | None, optional
            The size of interactions to consider. If not provided, uses
            ``self.interaction_size``.
        Notes
        -----
        Uses ``self.sketch_length``, ``self.interaction_size`` and
        ``self.sketch_dimension``.

        Updates
        -------
        ``self.map`` with the constructed sketch object whose type depends
        on the concrete implementation (:class:`list` of :class:`dict` for
        :class:`ConstraintSketch`, :class:`np.ndarray` for
        :class:`ExplicitSketch`).
        """

    @abstractmethod
    def build_from_all_interactions(
        self,
        interaction_size: int | None = None,
    ) -> None:
        """Build the sketch structure for *all* (non-consecutive) interactions.

        Parameters
        ----------
        interaction_size : int | None, optional
            The size of interactions to consider. If not provided, uses
            ``self.interaction_size``.

        Notes
        -----
        Uses ``self.sketch_length``, ``self.interaction_size`` and
        ``self.sketch_dimension``.

        Updates
        -------
        ``self.map`` with the constructed sketch object (see
        :meth:`build_from_nearest_neighbors`).
        """
    
    def use_custom_sketch(self, custom_sketch: Any) -> None:
        self.map = custom_sketch

    def set_interaction_size(self, interaction_size: int) -> None:
        self.interaction_size = interaction_size

    # ------------------------------------------------------------------
    # Marginal computation
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_marginal(self, function_data: Any) -> Any:
        """Compute marginals given function data and a sketch structure.

        Parameters
        ----------
        function_data:
            * For :class:`ConstraintSketch`: a ``(function_input_dits, function_values)``
              tuple where *function_input_dits* is a list of dit strings and
              *function_values* is the corresponding list of values.
            * For :class:`ExplicitSketch`: a flat list/array of floats holding
              the function value for every dit string in lexicographic order
              (full spectrum).
        The sketch object is read from ``self.map`` after one of the build
        methods has been called.

        Returns
        -------
        ``float``, ``list[float]``, or ``np.ndarray`` depending on the
        concrete implementation and the number of constraints / rows.
        """


# ---------------------------------------------------------------------------
# Concrete implementation: constraint-based (abstract representation)
# ---------------------------------------------------------------------------

class ConstraintSketchMap(SketchMap):
    """Sketch backed by dit constraints (implicit / abstract representation).

    The sketch is a :class:`list` of :class:`dict` ``{dit_index: dit_value}``.
    The function is provided as a sparse ``(input_dits, values)`` pair —
    no need to enumerate every possible dit string.

    See :mod:`abstract` for the underlying implementation.
    """

    def __init__(
        self,
        sketch_length: int,
        interaction_size: int,
        sketch_map=None,
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

    def build_from_nearest_neighbors(
        self,
        interaction_size: int | None = None,
    ) -> None:
        """Build nearest-neighbor constraints and store them in ``self.map``."""
        if interaction_size is not None:
            self.interaction_size = interaction_size
        self.map = constraints_for_nearest_neighbors_interactions(
            self.sketch_length,
            self.interaction_size,
            self.sketch_dimension,
        )

    def build_from_all_interactions(
        self,
        interaction_size: int | None = None,
    ) -> None:
        """Build all-interaction constraints and store them in ``self.map``."""
        if interaction_size is not None:
            self.interaction_size = interaction_size
        self.map = constraints_for_all_interactions(
            self.sketch_length,
            self.interaction_size,
            self.sketch_dimension,
        )

    def compute_marginal(self, *args) -> Union[float, list[float]]:
        """Compute marginals from sparse function data and constraint sketch.

        Accepts either:
        - ``(function_data,)`` where ``function_data`` is
          ``(function_input_dits, function_values)``.
        - ``(function_input_dits, function_values)``.
        """
        if self.map is None:
            raise ValueError("Sketch map is not initialized. Build a sketch first.")

        if len(args) == 1:
            function_data = args[0]
            function_input_dits, function_values = function_data
            return _constraint_compute_marginal(function_input_dits, function_values, self.map)
        if len(args) == 2:
            function_input_dits, function_values = args
            return _constraint_compute_marginal(function_input_dits, function_values, self.map)
        raise TypeError(
            "compute_marginal expects either (function_data,) or "
            "(function_input_dits, function_values)."
        )

    def reconstruct_structured_matrix_column(
        self,
        index: int,
    ) -> np.ndarray:
        return _reconstruct_structured_matrix_column(
            index,
            self.map,
            self.sketch_length,
            self.sketch_dimension,
        )

# ---------------------------------------------------------------------------
# Concrete implementation: explicit matrix
# ---------------------------------------------------------------------------

class ExplicitSketchMap(SketchMap):
    """Sketch backed by an explicit matrix stored in memory.

    The sketch is a :class:`np.ndarray` whose rows correspond to indicator
    vectors for each cylinder set.  The function must be given as a dense
    array of values over the *full* dit-string spectrum.

    See :mod:`explicit` for the underlying implementation.
    """

    def build_from_nearest_neighbors(
        self,
        interaction_size: int | None = None,
    ) -> None:
        """Build nearest-neighbor sketch matrix and store it in ``self.map``."""
        if interaction_size is not None:
            self.interaction_size = interaction_size
        if self.interaction_size is None:
            raise ValueError(
                "interaction_size is required to build the sketch. "
                "Set it on the instance before calling this method."
            )
        self.map = nearest_neighbors_interactions_sketch(
            self.sketch_length,
            self.interaction_size,
            self.sketch_dimension,
        )

    def build_from_all_interactions(
        self,
        interaction_size: int | None = None,
    ) -> None:
        """Build all-interactions sketch matrix and store it in ``self.map``."""
        if interaction_size is not None:
            self.interaction_size = interaction_size
        if self.interaction_size is None:
            raise ValueError(
                "interaction_size is required to build the sketch. "
                "Set it on the instance before calling this method."
            )
        self.map = all_interactions_sketch(
            self.sketch_length,
            self.interaction_size,
            self.sketch_dimension,
        )

    def compute_marginal(
        self,
        function_data: Union[list, np.ndarray],
    ) -> np.ndarray:
        """Compute marginals from a full-spectrum array and an explicit sketch.

        Parameters
        ----------
        function_data:
            1-D array/list of floats, one value per dit string in lexicographic
            order (full spectrum, length =
            ``self.sketch_dimension ** self.sketch_length``).
        The sketch matrix is read from ``self.map``.
        """
        if self.map is None:
            raise ValueError("Sketch map is not initialized. Build a sketch first.")
        return _explicit_compute_marginal(function_data, self.map)

    def random_sketch(
        self,
        size: int,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        self.map = _random_sketch(
            self.sketch_length,
            size,
            self.sketch_dimension,
            random_state=random_state,
        )
