"""hypothesis_lightcurves - A lightcurve generation testing tool using python-hypothesis."""

__version__ = "0.1.0"

from hypothesis_lightcurves.generators import (
    baseline_lightcurves,
    modified_lightcurves,
    lightcurves,
    periodic_lightcurves,
    transient_lightcurves,
)
from hypothesis_lightcurves.models import Lightcurve
from hypothesis_lightcurves.modifiers import (
    add_periodic_signal,
    add_transient_event,
    add_noise,
    add_outliers,
    add_gaps,
    add_trend,
    add_flares,
)

__all__ = [
    "__version__",
    # Models
    "Lightcurve",
    # Generators
    "baseline_lightcurves",
    "modified_lightcurves",
    "lightcurves",
    "periodic_lightcurves",
    "transient_lightcurves",
    # Modifiers
    "add_periodic_signal",
    "add_transient_event",
    "add_noise",
    "add_outliers",
    "add_gaps",
    "add_trend",
    "add_flares",
]