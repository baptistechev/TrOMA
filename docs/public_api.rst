Public API
==========

This page documents the supported public interface exposed by ``troma``.

Combinatorial Problem
---------------------

.. currentmodule:: troma

.. autoclass:: CombinatorialProblem
   :members:
   :show-inheritance:

.. autoclass:: RestrictedProblem
   :members:
   :show-inheritance:


Problem Sketches
----------------

.. currentmodule:: troma

.. autoclass:: ProblemSketch
   :members:
   :show-inheritance:

.. autoclass:: CombinatorialProblemSketch
   :members:
   :show-inheritance:

.. autoclass:: RestrictedProblemSketch
   :members:
   :show-inheritance:


Sketch Maps
-----------

.. currentmodule:: troma

.. autoclass:: ConstraintSketchMap
   :members:
   :show-inheritance:

.. autoclass:: ExplicitSketchMap
   :members:
   :show-inheritance:


Matching Pursuit
----------------

.. currentmodule:: troma

.. autofunction:: matching_pursuit
.. autofunction:: get_matching_pursuit
.. autofunction:: bind_matching_pursuit

.. autoclass:: MatchingPursuitResults
   :members:
   :show-inheritance:


Optimization
------------

.. currentmodule:: troma

.. autofunction:: get_optimizer
.. autofunction:: bind_optimizer
.. autofunction:: optimize


Data Structures
---------------

.. currentmodule:: troma

.. autoclass:: DitString
   :members:
   :show-inheritance:

.. autoclass:: CylinderSet
   :members:
   :show-inheritance:

.. autoclass:: Sample
   :members:
   :show-inheritance:

.. autoclass:: Restriction
   :members:
   :show-inheritance:

.. autoclass:: Hamiltonian
   :members:
   :show-inheritance:


Embedding
---------

.. currentmodule:: troma

.. autofunction:: spectrum_embedding
.. autofunction:: spectrum_restriction
.. autofunction:: reverse_spectrum_restriction
