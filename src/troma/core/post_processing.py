from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .structure import DitString
from .._validation import _Validator

if TYPE_CHECKING:
    from ..combinatorial_problem import CombinatorialProblem
    from ..problem_sketch import ProblemSketch


def greedy_2_bit_swap(
    candidate: DitString,
    problem: "CombinatorialProblem | ProblemSketch",
) -> DitString:
    """Greedy best-improving 2-bit swap local search for binary dit strings.

    Starting from *candidate*, each pass scans all pairs ``(i, j)`` where
    ``x[i] = 1`` and ``x[j] = 0``, computes::

        Δ = f(x with i←0, j←1) − f(x)

    and applies the swap with the largest positive Δ.  Iteration stops when
    no swap yields Δ > 0.

    Parameters
    ----------
    candidate : DitString
        Starting binary configuration (``dimension`` must equal 2).
    problem : CombinatorialProblem or ProblemSketch
        Object exposing an ``objective_function`` attribute — a callable that
        maps a 1-D integer ``np.ndarray`` of dit values to a scalar.  Both
        :class:`~troma.CombinatorialProblem` and all
        :class:`~troma.ProblemSketch` subclasses satisfy this interface.

    Returns
    -------
    DitString
        Locally optimal binary configuration under the 2-bit-swap
        neighbourhood.

    Raises
    ------
    ValueError
        If *candidate* has ``dimension != 2``.
    AttributeError
        If *problem* does not expose an ``objective_function`` attribute.
    """
    _Validator.ensure_instance("candidate", candidate, DitString)
    if not hasattr(problem, "objective_function"):
        raise AttributeError(
            "problem must expose an 'objective_function' attribute "
            "(CombinatorialProblem or ProblemSketch)."
        )
    _Validator.ensure_callable("problem.objective_function", problem.objective_function)
    if candidate.dimension != 2:
        raise ValueError(
            f"greedy_2_bit_swap requires a binary DitString (dimension=2), "
            f"got dimension={candidate.dimension}."
        )

    obj = problem.objective_function
    x: list[int] = candidate.tolist()
    n = len(x)
    fx = float(obj(np.array(x, dtype=int)))

    improved = True
    while improved:
        improved = False

        ones  = [i for i in range(n) if x[i] == 1]
        zeros = [j for j in range(n) if x[j] == 0]

        best_delta = 0.0
        best_i: int | None = None
        best_j: int | None = None

        for i in ones:
            for j in zeros:
                x[i], x[j] = 0, 1
                f_new = float(obj(np.array(x, dtype=int)))
                delta = f_new - fx
                x[i], x[j] = 1, 0  # revert

                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_i is not None:
            x[best_i], x[best_j] = 0, 1  # type: ignore[index]
            fx += best_delta
            improved = True

    return DitString(x, dimension=2)
