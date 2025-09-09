API Reference
=============

This section provides detailed API documentation for all modules in ``hypothesis_lightcurves``.

Models
------

.. automodule:: hypothesis_lightcurves.models
   :members:
   :undoc-members:
   :show-inheritance:

Lightcurve Class
~~~~~~~~~~~~~~~~

.. autoclass:: hypothesis_lightcurves.models.Lightcurve
   :members:
   :special-members: __init__, __post_init__
   :show-inheritance:

   .. autoattribute:: n_points
   .. autoattribute:: duration
   .. autoattribute:: mean_flux
   .. autoattribute:: std_flux

Generators
----------

.. automodule:: hypothesis_lightcurves.generators
   :members:
   :undoc-members:
   :show-inheritance:

Random Lightcurves
~~~~~~~~~~~~~~~~~~

.. autofunction:: hypothesis_lightcurves.generators.lightcurves

Periodic Lightcurves
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hypothesis_lightcurves.generators.periodic_lightcurves

Transient Lightcurves
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hypothesis_lightcurves.generators.transient_lightcurves

Utilities
---------

.. automodule:: hypothesis_lightcurves.utils
   :members:
   :undoc-members:
   :show-inheritance:

Resampling
~~~~~~~~~~

.. autofunction:: hypothesis_lightcurves.utils.resample_lightcurve

Gap Addition
~~~~~~~~~~~~

.. autofunction:: hypothesis_lightcurves.utils.add_gaps

Periodogram Calculation
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: hypothesis_lightcurves.utils.calculate_periodogram

Binning
~~~~~~~

.. autofunction:: hypothesis_lightcurves.utils.bin_lightcurve
