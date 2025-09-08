"""Data models for representing lightcurves."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass
class Lightcurve:
    """A basic lightcurve data model.
    
    Attributes:
        time: Array of observation times
        flux: Array of flux measurements
        flux_err: Optional array of flux uncertainties
        metadata: Optional dictionary for additional metadata
    """
    
    time: npt.NDArray[np.float64]
    flux: npt.NDArray[np.float64]
    flux_err: Optional[npt.NDArray[np.float64]] = None
    metadata: Optional[dict] = None
    
    def __post_init__(self) -> None:
        """Validate the lightcurve data."""
        if len(self.time) != len(self.flux):
            raise ValueError("Time and flux arrays must have the same length")
        
        if self.flux_err is not None and len(self.flux_err) != len(self.flux):
            raise ValueError("Flux error array must have the same length as flux array")
    
    @property
    def n_points(self) -> int:
        """Number of data points in the lightcurve."""
        return len(self.time)
    
    @property
    def duration(self) -> float:
        """Duration of the lightcurve observation."""
        return float(self.time.max() - self.time.min())
    
    @property
    def mean_flux(self) -> float:
        """Mean flux of the lightcurve."""
        return float(np.mean(self.flux))
    
    @property
    def std_flux(self) -> float:
        """Standard deviation of the flux."""
        return float(np.std(self.flux))
    
    def normalize(self) -> "Lightcurve":
        """Return a normalized copy of the lightcurve (mean=0, std=1)."""
        normalized_flux = (self.flux - self.mean_flux) / self.std_flux
        
        normalized_flux_err = None
        if self.flux_err is not None:
            normalized_flux_err = self.flux_err / self.std_flux
        
        return Lightcurve(
            time=self.time.copy(),
            flux=normalized_flux,
            flux_err=normalized_flux_err,
            metadata=self.metadata.copy() if self.metadata else None,
        )