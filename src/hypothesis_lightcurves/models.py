"""Data models for representing lightcurves."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class Lightcurve:
    """A basic lightcurve data model.

    Represents astronomical time series data with flux measurements over time,
    along with optional error bars and metadata.

    Parameters
    ----------
    time : numpy.ndarray[float64]
        Array of observation times, typically in days or seconds.
    flux : numpy.ndarray[float64]
        Array of flux measurements corresponding to each time point.
    flux_err : numpy.ndarray[float64] or None, optional
        Array of flux measurement uncertainties. Must have same length as flux
        if provided. Default is None.
    metadata : dict or None, optional
        Dictionary containing additional metadata about the lightcurve
        (e.g., source ID, filter, observing conditions). Default is None.
    modifications : list[str], optional
        List tracking applied modifications to the lightcurve. Automatically
        updated by modifier functions. Default is empty list.

    Attributes
    ----------
    n_points : int
        Number of data points in the lightcurve.
    duration : float
        Time span of the observations (max_time - min_time).
    mean_flux : float
        Mean value of the flux measurements.
    std_flux : float
        Standard deviation of the flux measurements.

    Raises
    ------
    ValueError
        If time and flux arrays have different lengths, or if flux_err
        is provided but has different length than flux.

    Examples
    --------
    >>> import numpy as np
    >>> from hypothesis_lightcurves.models import Lightcurve
    >>>
    >>> # Create a simple lightcurve
    >>> time = np.linspace(0, 10, 100)
    >>> flux = 100 + 5 * np.sin(2 * np.pi * time)
    >>> lc = Lightcurve(time=time, flux=flux)
    >>>
    >>> # Create with errors and metadata
    >>> flux_err = np.ones_like(flux) * 0.5
    >>> metadata = {'source_id': 'TIC12345', 'filter': 'TESS'}
    >>> lc = Lightcurve(time=time, flux=flux, flux_err=flux_err, metadata=metadata)
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
            If array lengths are inconsistent.
        """
        if len(self.time) != len(self.flux):
            raise ValueError("Time and flux arrays must have the same length")

        if self.flux_err is not None and len(self.flux_err) != len(self.flux):
            raise ValueError("Flux error array must have the same length as flux array")

    @property
    def n_points(self) -> int:
        """Number of data points in the lightcurve.

        Returns
        -------
        int
            The number of time/flux measurements.
        """
        return len(self.time)

    @property
    def duration(self) -> float:
        """Duration of the lightcurve observation.

        Returns
        -------
        float
            Time span from first to last observation.
        """
        return float(self.time.max() - self.time.min())

    @property
    def mean_flux(self) -> float:
        """Mean flux of the lightcurve.

        Returns
        -------
        float
            Average flux value across all measurements.
        """
        return float(np.mean(self.flux))

    @property
    def std_flux(self) -> float:
        """Standard deviation of the flux.

        Returns
        -------
        float
            Standard deviation of flux measurements.
        """
        return float(np.std(self.flux))

    def copy(self) -> "Lightcurve":
        """Create a deep copy of the lightcurve.

        Returns
        -------
        Lightcurve
            A new Lightcurve instance with copied data.

        Notes
        -----
        All arrays and mutable objects are deep copied to ensure
        the new instance is independent of the original.
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

        Normalizes flux to have mean=0 and std=1. Flux errors are
        scaled by the same factor if present.

        Returns
        -------
        Lightcurve
            New Lightcurve with normalized flux values.

        Notes
        -----
        - If the flux is constant (std=0), only the mean is subtracted.
        - Flux errors are divided by the same standard deviation used
          for flux normalization.
        - Adds 'normalized' to the modifications list.

        Examples
        --------
        >>> lc = Lightcurve(time=np.array([1, 2, 3]),
        ...                 flux=np.array([10, 20, 30]))
        >>> normalized = lc.normalize()
        >>> np.isclose(normalized.mean_flux, 0, atol=1e-10)
        True
        >>> np.isclose(normalized.std_flux, 1, atol=1e-10)
        True
        """
        # Handle edge case where std is zero (constant flux)
        if self.std_flux == 0 or np.isclose(self.std_flux, 0):
            # Return copy with flux shifted to zero mean
            normalized_flux = self.flux - self.mean_flux
            normalized_flux_err = self.flux_err.copy() if self.flux_err is not None else None
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
