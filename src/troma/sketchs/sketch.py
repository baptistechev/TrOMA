"""
sketch_base.py
==============
Common interface grouping the two sketch representations:

- :class:`ConstraintSketch`  — matrix described implicitly via dit constraints
  (memory-efficient, no need to materialise the full matrix).
- :class:`ExplicitSketch`    — matrix stored explicitly in memory
  (fast repeated computation once the matrix has been built).

Both classes expose the same three static methods so they can be used
interchangeably:

.. code-block:: python

    from sketch_base import ConstraintSketch, ExplicitSketch

    # Build the sketch once
    sketch = ConstraintSketch.build_nearest_neighbors_sketch(n, k)
    # Or
    sketch = ExplicitSketch.build_nearest_neighbors_sketch(n, k)

    # Then compute marginals with the matching class
    result = ConstraintSketch.compute_marginal((input_dits, values), sketch)
    result = ExplicitSketch.compute_marginal(full_spectrum_values, sketch)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from .abstract import (
    compute_marginal as _constraint_compute_marginal,
    constraints_for_nearest_neighbors_interactions,
    constraints_for_all_interactions,
    reconstruct_structured_matrix_column as _reconstruct_structured_matrix_column,
)
from .explicit import (
    compute_marginal as _explicit_compute_marginal,
    nearest_neighbors_interactions_sketch,
    all_interactions_sketch,
    random_sketch as _random_sketch,
)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Sketch(ABC):
    """Abstract interface for sketch-based marginal computation.

    A *sketch* is a compact representation of the structured matrix used to
    compute marginals of a function defined on the full dit-string spectrum.
    Concrete subclasses differ in how that matrix is stored / described:

    * :class:`ConstraintSketch`  — implicit description via dit constraints.
    * :class:`ExplicitSketch`    — explicit ``np.matrix`` stored in memory.

    All three methods are *static* so that the class itself acts as a
    strategy/namespace: no instance state is needed.
    """

    # ------------------------------------------------------------------
    # Sketch construction
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def build_nearest_neighbors_sketch(
        dit_string_length: int,
        interaction_size: int,
        dit_dimension: int = 2,
    ):
        """Build the sketch structure for nearest-neighbor interactions.

        Parameters
        ----------
        dit_string_length:
            Number of dits in each string.
        interaction_size:
            Number of consecutive dits per interaction window.
        dit_dimension:
            Number of possible values per dit (default: 2 for bits).

        Returns
        -------
        Sketch object whose type depends on the concrete implementation
        (:class:`list` of :class:`dict` for :class:`ConstraintSketch`,
        :class:`np.matrix` for :class:`ExplicitSketch`).
        """

    @staticmethod
    @abstractmethod
    def build_all_interactions_sketch(
        dit_string_length: int,
        interaction_size: int,
        dit_dimension: int = 2,
    ):
        """Build the sketch structure for *all* (non-consecutive) interactions.

        Parameters
        ----------
        dit_string_length:
            Number of dits in each string.
        interaction_size:
            Number of dits per interaction (all combinations considered).
        dit_dimension:
            Number of possible values per dit (default: 2 for bits).

        Returns
        -------
        Sketch object (see :meth:`build_nearest_neighbors_sketch`).
        """

    # ------------------------------------------------------------------
    # Marginal computation
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def compute_marginal(function_data, sketch):
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
        sketch:
            The sketch object returned by one of the ``build_*`` methods.

        Returns
        -------
        ``float``, ``list[float]``, or ``np.ndarray`` depending on the
        concrete implementation and the number of constraints / rows.
        """


# ---------------------------------------------------------------------------
# Concrete implementation: constraint-based (abstract representation)
# ---------------------------------------------------------------------------

class ConstraintSketch(Sketch):
    """Sketch backed by dit constraints (implicit / abstract representation).

    The sketch is a :class:`list` of :class:`dict` ``{dit_index: dit_value}``.
    The function is provided as a sparse ``(input_dits, values)`` pair —
    no need to enumerate every possible dit string.

    See :mod:`abstract` for the underlying implementation.
    """

    @staticmethod
    def build_nearest_neighbors_sketch(
        dit_string_length: int,
        interaction_size: int,
        dit_dimension: int = 2,
    ) -> list[dict]:
        """Return a list of nearest-neighbor constraint dicts."""
        return constraints_for_nearest_neighbors_interactions(
            dit_string_length, interaction_size, dit_dimension
        )

    @staticmethod
    def build_all_interactions_sketch(
        dit_string_length: int,
        interaction_size: int,
        dit_dimension: int = 2,
    ) -> list[dict]:
        """Return a list of all-interaction constraint dicts."""
        return constraints_for_all_interactions(
            dit_string_length, interaction_size, dit_dimension
        )

    @staticmethod
    def compute_marginal(
        function_data: tuple,
        sketch: list[dict],
    ) -> Union[float, list[float]]:
        """Compute marginals from sparse function data and constraint sketch.

        Parameters
        ----------
        function_data:
            ``(function_input_dits, function_values)`` — sparse representation
            of the function.
        sketch:
            List of constraint dicts as returned by :meth:`build_nearest_neighbors_sketch`
            or :meth:`build_all_interactions_sketch`.
        """
        function_input_dits, function_values = function_data
        return _constraint_compute_marginal(function_input_dits, function_values, sketch)

    @staticmethod
    def compute_marginals(*args) -> Union[float, list[float]]:
        if len(args) == 2:
            function_data, sketch = args
            return ConstraintSketch.compute_marginal(function_data, sketch)
        if len(args) == 3:
            function_input_dits, function_values, sketch = args
            return _constraint_compute_marginal(function_input_dits, function_values, sketch)
        raise TypeError(
            "compute_marginals expects either (function_data, sketch) or "
            "(function_input_dits, function_values, sketch)."
        )

    @staticmethod
    def reconstruct_structured_matrix_column(
        index: int,
        dit_constraints: list[dict],
        dit_string_length: int,
        dit_dimension: int = 2,
    ) -> np.ndarray:
        return _reconstruct_structured_matrix_column(
            index,
            dit_constraints,
            dit_string_length,
            dit_dimension,
        )


# ---------------------------------------------------------------------------
# Concrete implementation: explicit matrix
# ---------------------------------------------------------------------------

class ExplicitSketch(Sketch):
    """Sketch backed by an explicit matrix stored in memory.

    The sketch is a :class:`np.matrix` whose rows correspond to indicator
    vectors for each cylinder set.  The function must be given as a dense
    array of values over the *full* dit-string spectrum.

    See :mod:`explicit` for the underlying implementation.
    """

    @staticmethod
    def build_nearest_neighbors_sketch(
        dit_string_length: int,
        interaction_size: int,
        dit_dimension: int = 2,
    ) -> np.matrix:
        """Return the nearest-neighbor sketch matrix."""
        return nearest_neighbors_interactions_sketch(
            dit_string_length, interaction_size, dit_dimension
        )

    @staticmethod
    def build_all_interactions_sketch(
        dit_string_length: int,
        interaction_size: int,
        dit_dimension: int = 2,
    ) -> np.matrix:
        """Return the all-interactions sketch matrix."""
        return all_interactions_sketch(
            dit_string_length, interaction_size, dit_dimension
        )

    @staticmethod
    def compute_marginal(
        function_data: Union[list, np.ndarray],
        sketch: np.matrix,
    ) -> np.ndarray:
        """Compute marginals from a full-spectrum array and an explicit sketch.

        Parameters
        ----------
        function_data:
            1-D array/list of floats, one value per dit string in lexicographic
            order (full spectrum, length = ``dit_dimension ** dit_string_length``).
        sketch:
            Explicit sketch matrix as returned by :meth:`build_nearest_neighbors_sketch`
            or :meth:`build_all_interactions_sketch`.
        """
        return _explicit_compute_marginal(function_data, sketch)

    @staticmethod
    def compute_marginals(
        function_data: Union[list, np.ndarray],
        sketch: np.matrix,
    ) -> np.ndarray:
        return ExplicitSketch.compute_marginal(function_data, sketch)

    @staticmethod
    def random_sketch(
        dit_string_length: int,
        m: int,
        dit_dimension: int = 2,
        random_state=None,
    ) -> np.matrix:
        return _random_sketch(
            dit_string_length,
            m,
            dit_dimension=dit_dimension,
            random_state=random_state,
        )
