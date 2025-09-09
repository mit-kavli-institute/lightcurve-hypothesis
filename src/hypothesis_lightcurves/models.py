"""Data models for representing lightcurves.

This module provides the core data structures for representing astronomical
lightcurves, which are time series of brightness measurements from celestial
objects.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class Lightcurve:
    """A basic lightcurve data model for astronomical time series.

    A lightcurve represents the variation of an astronomical object's
    brightness (flux) over time. This class provides a standard interface
    for working with such time series data, including optional measurement
    uncertainties and metadata.

    Parameters
    ----------
    time : numpy.ndarray
        Array of observation times, typically in days or seconds.
        Must be the same length as flux.
    flux : numpy.ndarray
        Array of flux (brightness) measurements corresponding to each time point.
        Units depend on the photometric system used.
    flux_err : numpy.ndarray, optional
        Array of flux measurement uncertainties (1-sigma errors).
        Must be the same length as flux if provided.
    metadata : dict, optional
        Dictionary containing additional information about the lightcurve,
        such as object name, filter, observatory, etc.
    modifications : list of str, optional
        List tracking any transformations applied to the lightcurve,
        useful for data provenance.

    Attributes
    ----------
    n_points : int
        Number of data points in the lightcurve.
    duration : float
        Total time span of observations.
    mean_flux : float
        Mean flux value across all observations.
    std_flux : float
        Standard deviation of flux values.

    Raises
    ------
    ValueError
        If time and flux arrays have different lengths.
        If flux_err is provided and has different length than flux.

    Examples
    --------
    >>> import numpy as np
    >>> from hypothesis_lightcurves.models import Lightcurve
    >>>
    >>> # Create a simple lightcurve
    >>> time = np.linspace(0, 10, 100)
    >>> flux = 100 + 5 * np.sin(2 * np.pi * time / 2.5)
    >>> lc = Lightcurve(time=time, flux=flux)
    >>>
    >>> # Access properties
    >>> print(f"Number of points: {lc.n_points}")
    >>> print(f"Duration: {lc.duration:.2f} days")
    >>> print(f"Mean flux: {lc.mean_flux:.2f}")

    Notes
    -----
    The lightcurve data is not copied on initialization, so modifications
    to the input arrays will affect the lightcurve object. Use the `copy()`
    method to create an independent copy when needed.
    """

    time: npt.NDArray[np.float64]
    flux: npt.NDArray[np.float64]
    flux_err: npt.NDArray[np.float64] | None = None
    metadata: dict | None = None
    modifications: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the lightcurve data after initialization.

        Raises
        ------
        ValueError
            If arrays have incompatible lengths.
        """
        if len(self.time) != len(self.flux):
            raise ValueError("Time and flux arrays must have the same length")

        if self.flux_err is not None and len(self.flux_err) != len(self.flux):
            raise ValueError("Flux error array must have the same length as flux array")

    @property
    def n_points(self) -> int:
        """Get the number of data points in the lightcurve.

        Returns
        -------
        int
            Number of observations.
        """
        return len(self.time)

    @property
    def duration(self) -> float:
        """Calculate the duration of the lightcurve observation.

        Returns
        -------
        float
            Time span from first to last observation.
        """
        return float(self.time.max() - self.time.min())

    @property
    def mean_flux(self) -> float:
        """Calculate the mean flux of the lightcurve.

        Returns
        -------
        float
            Average flux value across all observations.
        """
        return float(np.mean(self.flux))

    @property
    def std_flux(self) -> float:
        """Calculate the standard deviation of the flux.

        Returns
        -------
        float
            Standard deviation of flux values.
        """
        return float(np.std(self.flux))

    def copy(self) -> "Lightcurve":
        """Create a deep copy of the lightcurve.

        Returns
        -------
        Lightcurve
            A new Lightcurve object with copied data arrays.

        Examples
        --------
        >>> lc_copy = lc.copy()
        >>> lc_copy.flux[0] = 0  # Doesn't affect original
        """
        return Lightcurve(
            time=self.time.copy(),
            flux=self.flux.copy(),
            flux_err=self.flux_err.copy() if self.flux_err is not None else None,
            metadata=self.metadata.copy() if self.metadata else None,
            modifications=self.modifications.copy(),
        )

    def normalize(self) -> "Lightcurve":
        """Return a normalized copy of the lightcurve.

        Normalization transforms the flux to have zero mean and unit variance.
        This is useful for comparing lightcurves with different baseline flux
        levels or for certain analysis techniques.

        Returns
        -------
        Lightcurve
            A new Lightcurve object with normalized flux values.
            If flux_err is present, it is also scaled by the same factor.

        Notes
        -----
        If the flux has zero standard deviation (constant lightcurve),
        only the mean is subtracted, and errors remain unchanged.

        Examples
        --------
        >>> normalized_lc = lc.normalize()
        >>> print(f"Mean: {normalized_lc.mean_flux:.10f}")  # Should be ~0
        >>> print(f"Std: {normalized_lc.std_flux:.10f}")    # Should be ~1
        """
        # Handle edge case where std is zero (constant flux)
        if np.isclose(self.std_flux, 0):
            # Return copy with flux shifted to zero mean
            normalized_flux = self.flux - self.mean_flux
            normalized_flux_err = (
                np.full_like(self.flux_err, np.nan) if self.flux_err is not None else None
            )
        else:
            normalized_flux = (self.flux - self.mean_flux) / self.std_flux
            normalized_flux_err = None
            if self.flux_err is not None:
                normalized_flux_err = self.flux_err / self.std_flux

        new_modifications = self.modifications.copy()
        new_modifications.append("normalized")

        return Lightcurve(
            time=self.time.copy(),
            flux=normalized_flux,
            flux_err=normalized_flux_err,
            metadata=self.metadata.copy() if self.metadata else None,
            modifications=new_modifications,
        )
