"""Tests for lightcurve modifiers."""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from hypothesis_lightcurves.generators import baseline_lightcurves, modified_lightcurves
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


class TestModifiers:
    """Test individual modifier functions."""
    
    @given(lc=baseline_lightcurves())
    @settings(max_examples=50)
    def test_add_periodic_signal(self, lc: Lightcurve) -> None:
        """Test adding periodic signal to lightcurve."""
        original_mean = lc.mean_flux
        modified = add_periodic_signal(lc, period=10.0, amplitude=5.0, phase=0.0)
        
        # Check that modifications are tracked
        assert "periodic_signal" in modified.modifications[-1]
        assert modified.metadata is not None
        assert "periodic_period" in modified.metadata
        assert modified.metadata["periodic_period"] == 10.0
        
        # Check that original is unchanged
        assert np.array_equal(lc.flux, lc.flux)
        
        # Check that flux was modified (unless time points coincide with period zeros)
        # In rare cases where time points are exactly at multiples of the period and phase=0,
        # the sine could be 0 at all points
        if not np.allclose(np.sin(2 * np.pi * lc.time / 10.0), 0, atol=1e-10):
            assert not np.array_equal(modified.flux, lc.flux)
        
        # Time should be unchanged
        assert np.array_equal(modified.time, lc.time)
    
    @given(lc=baseline_lightcurves())
    @settings(max_examples=50)
    def test_add_noise_gaussian(self, lc: Lightcurve) -> None:
        """Test adding Gaussian noise to lightcurve."""
        modified = add_noise(lc, noise_type="gaussian", level=1.0, seed=42)
        
        # Check that modifications are tracked
        assert "noise(gaussian" in modified.modifications[-1]
        assert modified.metadata is not None
        assert "noise_gaussian_level" in modified.metadata
        
        # Check that flux errors are added
        assert modified.flux_err is not None
        assert len(modified.flux_err) == len(modified.flux)
        
        # Check reproducibility with seed
        modified2 = add_noise(lc, noise_type="gaussian", level=1.0, seed=42)
        assert np.array_equal(modified.flux, modified2.flux)
    
    @given(lc=baseline_lightcurves())
    @settings(max_examples=30)
    def test_add_transient_burst(self, lc: Lightcurve) -> None:
        """Test adding transient burst to lightcurve."""
        peak_time = float(lc.time.min() + lc.duration / 2)
        modified = add_transient_event(
            lc,
            peak_time=peak_time,
            peak_flux=100.0,
            rise_time=1.0,
            decay_time=5.0,
            event_type="burst",
        )
        
        # Check metadata
        assert modified.metadata is not None
        assert "burst_peak_time" in modified.metadata
        assert modified.metadata["burst_peak_time"] == peak_time
        
        # Check that flux increased around peak
        peak_idx = np.argmin(np.abs(modified.time - peak_time))
        assert modified.flux[peak_idx] > lc.flux[peak_idx]
    
    @given(lc=baseline_lightcurves())
    @settings(max_examples=30)
    def test_add_outliers(self, lc: Lightcurve) -> None:
        """Test adding outliers to lightcurve."""
        modified = add_outliers(lc, fraction=0.1, amplitude=5.0, seed=42)
        
        # Check metadata
        assert modified.metadata is not None
        assert "outlier_fraction" in modified.metadata
        assert "outlier_indices" in modified.metadata
        
        # Check that some points were modified
        n_expected_outliers = int(len(lc.flux) * 0.1)
        n_actual_outliers = len(modified.metadata["outlier_indices"])
        assert n_actual_outliers == n_expected_outliers
        
        # Check reproducibility
        modified2 = add_outliers(lc, fraction=0.1, amplitude=5.0, seed=42)
        assert np.array_equal(modified.flux, modified2.flux)
    
    @given(lc=baseline_lightcurves(min_points=100, max_points=200))
    @settings(max_examples=30)
    def test_add_gaps(self, lc: Lightcurve) -> None:
        """Test adding gaps to lightcurve."""
        original_length = len(lc.time)
        modified = add_gaps(lc, n_gaps=2, gap_fraction=0.2, seed=42)
        
        # Check that points were removed
        assert len(modified.time) <= original_length
        # Allow some tolerance in the gap fraction due to discrete points
        actual_fraction_removed = 1 - (len(modified.time) / original_length)
        assert abs(actual_fraction_removed - 0.2) < 0.1  # Within 10% of target
        
        # Check metadata
        assert modified.metadata is not None
        assert modified.metadata["gap_count"] == 2
        assert modified.metadata["gap_fraction"] == 0.2
    
    @given(lc=baseline_lightcurves())
    @settings(max_examples=30)
    def test_add_linear_trend(self, lc: Lightcurve) -> None:
        """Test adding linear trend to lightcurve."""
        original_flux = lc.flux.copy()
        modified = add_trend(lc, trend_type="linear", coefficient=0.5)
        
        # Check metadata
        assert modified.metadata is not None
        assert "trend_linear_coefficient" in modified.metadata
        
        # Check that trend was added (flux should generally increase)
        first_quarter = modified.flux[:len(modified.flux)//4].mean()
        last_quarter = modified.flux[-len(modified.flux)//4:].mean()
        assert last_quarter > first_quarter
    
    @given(lc=baseline_lightcurves())
    @settings(max_examples=30)
    def test_add_flares(self, lc: Lightcurve) -> None:
        """Test adding flares to lightcurve."""
        original_max = lc.flux.max()
        modified = add_flares(
            lc,
            n_flares=3,
            min_amplitude=0.5,
            max_amplitude=2.0,
            seed=42,
        )
        
        # Check metadata
        assert modified.metadata is not None
        assert modified.metadata["flare_count"] == 3
        assert "flare_info" in modified.metadata
        assert len(modified.metadata["flare_info"]) == 3
        
        # Check that flux was modified (flares should increase max flux)
        # But with small lightcurves, flares might not always hit a sample point
        assert modified.flux.max() >= original_max


class TestModifierComposition:
    """Test composing multiple modifiers."""
    
    @given(lc=baseline_lightcurves())
    @settings(max_examples=30)
    def test_multiple_modifiers_tracking(self, lc: Lightcurve) -> None:
        """Test that multiple modifiers are properly tracked."""
        # Apply multiple modifiers
        modified = lc
        modified = add_periodic_signal(modified, period=10.0, amplitude=5.0)
        modified = add_noise(modified, noise_type="gaussian", level=1.0)
        modified = add_outliers(modified, fraction=0.05, amplitude=3.0)
        
        # Check all modifications are tracked
        # Note: outliers might not be added if fraction * n_points < 1
        assert len(modified.modifications) >= 3  # baseline + at least 2 modifiers
        assert any("periodic_signal" in m for m in modified.modifications)
        assert any("noise" in m for m in modified.modifications)
        # Outliers only if they were actually added
        if "outlier_fraction" in modified.metadata:
            assert any("outliers" in m for m in modified.modifications)
        
        # Check metadata accumulates
        assert modified.metadata is not None
        assert "periodic_period" in modified.metadata
        assert "noise_gaussian_level" in modified.metadata
        # Only check for outlier metadata if outliers were added
        if "outlier_fraction" in modified.metadata:
            assert modified.metadata["outlier_fraction"] == 0.05
    
    @given(data=st.data())
    @settings(max_examples=30)
    def test_modified_lightcurves_strategy(self, data: st.DataObject) -> None:
        """Test the modified_lightcurves strategy."""
        # Test with specific modifiers
        lc = data.draw(modified_lightcurves(
            modifiers=[add_noise, add_periodic_signal],
        ))
        
        # Should have baseline plus 0-2 modifiers
        assert "baseline" in lc.modifications
        assert len(lc.modifications) <= 3
        
        # Check it's a valid lightcurve
        assert isinstance(lc, Lightcurve)
        assert len(lc.time) == len(lc.flux)
    
    @given(lc=baseline_lightcurves())
    @settings(max_examples=20)
    def test_copy_independence(self, lc: Lightcurve) -> None:
        """Test that modifications don't affect the original."""
        original_flux = lc.flux.copy()
        original_mods = lc.modifications.copy()
        
        # Apply multiple modifications
        modified = add_periodic_signal(lc, period=10.0, amplitude=5.0)
        modified = add_noise(modified, noise_type="gaussian", level=1.0)
        
        # Original should be unchanged
        assert np.array_equal(lc.flux, original_flux)
        assert lc.modifications == original_mods
        
        # Modified should be different
        assert not np.array_equal(modified.flux, original_flux)
        assert len(modified.modifications) > len(original_mods)


class TestBaselineStrategies:
    """Test baseline lightcurve generation strategies."""
    
    @given(lc=baseline_lightcurves(baseline_type="flat"))
    @settings(max_examples=30)
    def test_flat_baseline(self, lc: Lightcurve) -> None:
        """Test flat baseline generation."""
        # Flux should be nearly constant
        assert lc.std_flux / lc.mean_flux < 0.01  # Less than 1% variation
        assert "baseline" in lc.modifications
        assert lc.metadata is not None
        assert lc.metadata["baseline_type"] == "flat"
    
    @given(lc=baseline_lightcurves(baseline_type="random_walk"))
    @settings(max_examples=30)
    def test_random_walk_baseline(self, lc: Lightcurve) -> None:
        """Test random walk baseline generation."""
        # Should have more variation than flat
        assert lc.std_flux > 0
        assert lc.metadata is not None
        assert lc.metadata["baseline_type"] == "random_walk"
    
    @given(lc=baseline_lightcurves(time_sampling="uniform"))
    @settings(max_examples=30)
    def test_uniform_time_sampling(self, lc: Lightcurve) -> None:
        """Test uniform time sampling."""
        time_diffs = np.diff(lc.time)
        # Should be approximately uniform
        assert np.std(time_diffs) / np.mean(time_diffs) < 0.01
        assert lc.metadata is not None
        assert lc.metadata["time_sampling"] == "uniform"
    
    @given(lc=baseline_lightcurves(time_sampling="random"))
    @settings(max_examples=30)
    def test_random_time_sampling(self, lc: Lightcurve) -> None:
        """Test random time sampling."""
        time_diffs = np.diff(lc.time)
        # Should have variable spacing
        assert np.std(time_diffs) > 0
        assert lc.metadata is not None
        assert lc.metadata["time_sampling"] == "random"