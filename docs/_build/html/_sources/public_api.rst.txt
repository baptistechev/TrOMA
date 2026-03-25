Public API
==========

This page documents the supported public interface exposed by ``troma``.

Modeling
--------

.. currentmodule:: troma

.. autofunction:: mcco_modeling
.. autofunction:: solve_via_mcco


Data utilities
--------------

.. currentmodule:: troma

.. autofunction:: integer_to_dit_string
.. autofunction:: dit_string_to_integer
.. autofunction:: dit_string_to_computational_basis
.. autofunction:: create_cylinder_set_indicator
.. autofunction:: kronecker_develop
.. autofunction:: belongs_to_cylinder_set


Sketches
--------

.. currentmodule:: troma

.. autoclass:: ConstraintSketch
   :members:
   :show-inheritance:

.. autoclass:: ExplicitSketch
   :members:
   :show-inheritance:


Optimization
------------

.. currentmodule:: troma

.. autofunction:: bind_optimizer
.. autofunction:: get_optimizer
.. autofunction:: optimize


Decoding
--------

.. currentmodule:: troma

.. autofunction:: bind_matching_pursuit
.. autofunction:: get_matching_pursuit
.. autofunction:: matching_pursuit