"""Tests for lightcurve generators using property-based testing."""

import numpy as np
from numpy import testing as npt
import pytest
from hypothesis import given, settings, note

from hypothesis_lightcurves.generators import (
    lightcurves,
    periodic_lightcurves,
    transient_lightcurves,
)
from hypothesis_lightcurves.models import Lightcurve


class TestBasicLightcurveGenerator:
    """Test the basic lightcurve generator."""
    
    @given(lc=lightcurves())
    def test_generates_valid_lightcurve(self, lc: Lightcurve) -> None:
        """Test that generated lightcurves are valid."""
        assert isinstance(lc, Lightcurve)
        assert len(lc.time) == len(lc.flux)
        assert lc.n_points >= 10  # Default minimum
        assert lc.n_points <= 1000  # Default maximum
    
    @given(lc=lightcurves(min_points=50, max_points=100))
    def test_respects_point_constraints(self, lc: Lightcurve) -> None:
        """Test that point count constraints are respected."""
        assert 50 <= lc.n_points <= 100
    
    @given(lc=lightcurves(with_errors=True))
    def test_generates_errors_when_requested(self, lc: Lightcurve) -> None:
        """Test that flux errors are generated when requested."""
        assert lc.flux_err is not None
        assert len(lc.flux_err) == len(lc.flux)
        assert np.all(lc.flux_err >= 0)  # Errors should be non-negative
    
    @given(lc=lightcurves())
    def test_time_is_sorted(self, lc: Lightcurve) -> None:
        """Test that time arrays are always sorted."""
        assert np.all(np.diff(lc.time) >= 0)
    
    @given(lc=lightcurves(min_flux=10, max_flux=100))
    def test_flux_bounds(self, lc: Lightcurve) -> None:
        """Test that flux values respect specified bounds."""
        assert np.all(lc.flux >= 10)
        assert np.all(lc.flux <= 100)


class TestPeriodicLightcurveGenerator:
    """Test the periodic lightcurve generator."""
    
    @given(lc=periodic_lightcurves())
    def test_generates_valid_periodic_lightcurve(self, lc: Lightcurve) -> None:
        """Test that periodic lightcurves are valid."""
        assert isinstance(lc, Lightcurve)
        assert lc.metadata is not None
        assert "periodic_period" in lc.metadata
        assert "periodic_amplitude" in lc.metadata
        assert "periodic_phase" in lc.metadata
    
    @given(lc=periodic_lightcurves(min_period=1.0, max_period=2.0))
    def test_period_constraints(self, lc: Lightcurve) -> None:
        """Test that period constraints are respected."""
        assert lc.metadata is not None
        assert 1.0 <= lc.metadata["periodic_period"] <= 2.0
    
    @given(lc=periodic_lightcurves(with_noise=False))
    def test_no_noise_option(self, lc: Lightcurve) -> None:
        """Test that noise-free option works."""
        assert lc.flux_err is None
    
    @given(lc=periodic_lightcurves())
    def test_time_spacing_is_uniform(self, lc: Lightcurve) -> None:
        """Test that time points are uniformly spaced."""
        time_diffs = np.diff(lc.time)
        assert np.allclose(time_diffs, time_diffs[0], rtol=1e-10)


class TestTransientLightcurveGenerator:
    """Test the transient lightcurve generator."""
    
    @given(lc=transient_lightcurves())
    def test_generates_valid_transient_lightcurve(self, lc: Lightcurve) -> None:
        """Test that transient lightcurves are valid."""
        assert isinstance(lc, Lightcurve)
        assert lc.metadata is not None
        assert "burst_peak_time" in lc.metadata
        assert "burst_rise_time" in lc.metadata
        assert "burst_decay_time" in lc.metadata
        assert "burst_peak_flux" in lc.metadata
    
    @given(lc=transient_lightcurves(min_peak_time=20, max_peak_time=30))
    def test_peak_time_constraints(self, lc: Lightcurve) -> None:
        """Test that peak time constraints are respected."""
        assert lc.metadata is not None
        assert 20 <= lc.metadata["burst_peak_time"] <= 30
    
    @given(lc=transient_lightcurves())
    def test_has_rise_and_decay(self, lc: Lightcurve) -> None:
        """Test that transient has both rise and decay phases."""
        assert lc.metadata is not None
        peak_time = lc.metadata["burst_peak_time"]
        
        # Check we have points around peak (might all be after if peak is early)
        # Transients can have peak_time at or before the start of observations
        if peak_time > lc.time.min():
            assert np.any(lc.time < peak_time) or np.isclose(lc.time.min(), peak_time)
        assert np.any(lc.time >= peak_time)
        
        # Check flux generally increases before peak and decreases after
        before_peak = lc.flux[lc.time < peak_time]
        after_peak = lc.flux[lc.time > peak_time]
        
        if len(before_peak) > 1:
            # Allow for noise but general trend should be increasing
            assert np.mean(before_peak[-len(before_peak)//2:]) > np.mean(before_peak[:len(before_peak)//2])
        
        if len(after_peak) > 1:
            # Allow for noise but general trend should be decreasing
            assert np.mean(after_peak[:len(after_peak)//2]) > np.mean(after_peak[-len(after_peak)//2:])


class TestLightcurveProperties:
    """Test properties that should hold for all lightcurves."""
    
    @given(lc=lightcurves())
    def test_normalization_properties(self, lc: Lightcurve) -> None:
        """Test that normalization produces expected properties."""
        normalized = lc.normalize()
        
        note(lc.std_flux)
        note(normalized.std_flux)
        npt.assert_almost_equal(normalized.mean_flux, 0.0)
        
        assert np.isclose(normalized.std_flux, 1.0) or normalized.std_flux < 1
        assert np.array_equal(normalized.time, lc.time)
    
    @given(lc=lightcurves(with_errors=True))
    def test_normalization_scales_errors(self, lc: Lightcurve) -> None:
        """Test that normalization properly scales errors."""
        normalized = lc.normalize()
        
        assert normalized.flux_err is not None
        assert lc.flux_err is not None
       
        # Errors should be scaled by the same factor as flux
        if not (lc.std_flux == 0 or np.isclose(lc.std_flux, 0)):
            scale_factor = lc.std_flux
            note(f"{lc.std_flux=}")
            note(f"{scale_factor=}")
            expected_errors = lc.flux_err / scale_factor
            npt.assert_allclose(normalized.flux_err, expected_errors)
        else:
            # If std is 0 (or very close to 0), errors should be unchanged
            npt.assert_allclose(normalized.flux_err, lc.flux_err)
    
    @given(lc=lightcurves())
    def test_duration_calculation(self, lc: Lightcurve) -> None:
        """Test that duration is correctly calculated."""
        expected_duration = lc.time[-1] - lc.time[0]
        npt.assert_allclose(lc.duration, expected_duration)
    
    @given(lc=lightcurves())
    def test_statistics_are_finite(self, lc: Lightcurve) -> None:
        """Test that statistical properties are finite."""
        assert np.isfinite(lc.mean_flux)
        assert np.isfinite(lc.std_flux)
        assert np.isfinite(lc.duration)
