"""hypothesis_lightcurves - A lightcurve generation testing tool using python-hypothesis."""

__version__ = "0.1.0"

from hypothesis_lightcurves.generators import *
from hypothesis_lightcurves.models import *

# Import visualization module only if matplotlib is available
try:
    from hypothesis_lightcurves.visualization import *
except ImportError:
    pass  # Visualization module is optional

__all__ = ["__version__"]
