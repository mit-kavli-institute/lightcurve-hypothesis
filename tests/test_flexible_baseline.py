"""Tests for the flexible baseline_lightcurves generator."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis_lightcurves.generators import baseline_lightcurves
from hypothesis_lightcurves.models import Lightcurve


class TestFlexibleBaseline:
    """Test the flexible baseline_lightcurves with strategies as parameters."""

    def test_static_parameters(self) -> None:
        """Test that static parameters work as before."""

        # Draw a single lightcurve with static parameters
        @given(
            lc=baseline_lightcurves(
                n_points=100, baseline_type="flat", baseline_flux=50.0, time_sampling="uniform"
            )
        )
        @settings(max_examples=1)
        def check_static(lc: Lightcurve) -> None:
            assert lc.n_points == 100
            assert lc.metadata["baseline_type"] == "flat"
            assert lc.metadata["baseline_flux"] == 50.0
            assert lc.metadata["time_sampling"] == "uniform"

        check_static()

    def test_strategy_parameters(self) -> None:
        """Test that strategy parameters work correctly."""

        @given(
            lc=baseline_lightcurves(
                n_points=st.integers(50, 60),
                baseline_type=st.sampled_from(["flat", "smooth"]),
                baseline_flux=st.floats(80, 120),
                time_sampling=st.sampled_from(["uniform", "random"]),
            )
        )
        @settings(max_examples=10)
        def check_strategies(lc: Lightcurve) -> None:
            assert 50 <= lc.n_points <= 60
            assert lc.metadata["baseline_type"] in ["flat", "smooth"]
            assert 80 <= lc.metadata["baseline_flux"] <= 120
            assert lc.metadata["time_sampling"] in ["uniform", "random"]

        check_strategies()

    def test_mixed_parameters(self) -> None:
        """Test mixing static and strategy parameters."""

        @given(
            lc=baseline_lightcurves(
                n_points=100,  # Static
                baseline_type="flat",  # Static
                baseline_flux=st.floats(90, 110),  # Strategy
                time_sampling=st.sampled_from(["uniform", "random"]),  # Strategy
            )
        )
        @settings(max_examples=10)
        def check_mixed(lc: Lightcurve) -> None:
            assert lc.n_points == 100
            assert lc.metadata["baseline_type"] == "flat"
            assert 90 <= lc.metadata["baseline_flux"] <= 110
            assert lc.metadata["time_sampling"] in ["uniform", "random"]

        check_mixed()

    def test_time_range_strategies(self) -> None:
        """Test that time range parameters accept strategies."""

        @given(
            lc=baseline_lightcurves(
                n_points=50,
                min_time=st.floats(0, 10),
                max_time=st.floats(90, 100),
                baseline_type="flat",
            )
        )
        @settings(max_examples=10)
        def check_time_range(lc: Lightcurve) -> None:
            assert lc.n_points == 50
            assert lc.time.min() >= 0
            assert lc.time.max() <= 100

        check_time_range()

    def test_duration_and_start_time(self) -> None:
        """Test duration and start_time parameters for uniform sampling."""

        @given(
            lc=baseline_lightcurves(
                n_points=50,
                duration=st.floats(10, 20),
                start_time=st.floats(0, 5),
                time_sampling="uniform",
                baseline_type="flat",
            )
        )
        @settings(max_examples=10)
        def check_duration(lc: Lightcurve) -> None:
            assert lc.n_points == 50
            assert lc.metadata["time_sampling"] == "uniform"
            # Duration should be approximately what we specified
            actual_duration = lc.time.max() - lc.time.min()
            assert 10 <= actual_duration <= 20
            # Start time should be approximately what we specified
            assert 0 <= lc.time.min() <= 5

        check_duration()

    def test_baseline_types(self) -> None:
        """Test different baseline types."""

        @given(baseline_type=st.sampled_from(["flat", "random_walk", "smooth"]), data=st.data())
        @settings(max_examples=15)
        def check_baseline_types(baseline_type: str, data: st.DataObject) -> None:
            lc = data.draw(
                baseline_lightcurves(n_points=100, baseline_type=baseline_type, baseline_flux=100.0)
            )

            assert lc.metadata["baseline_type"] == baseline_type

            if baseline_type == "flat":
                # Flat should have very low standard deviation
                assert lc.std_flux / lc.mean_flux < 0.25  # Allow for variation factor
            elif baseline_type == "random_walk":
                # Random walk should have some variation
                assert lc.std_flux > 0
            elif baseline_type == "smooth":
                # Smooth should be continuous (no huge jumps)
                flux_diffs = np.diff(lc.flux)
                assert np.max(np.abs(flux_diffs)) < lc.mean_flux * 0.5

        check_baseline_types()

    def test_time_sampling_patterns(self) -> None:
        """Test different time sampling patterns."""

        @given(time_sampling=st.sampled_from(["uniform", "random", "irregular"]), data=st.data())
        @settings(max_examples=15)
        def check_time_sampling(time_sampling: str, data: st.DataObject) -> None:
            lc = data.draw(
                baseline_lightcurves(
                    n_points=st.integers(50, 100), time_sampling=time_sampling, baseline_type="flat"
                )
            )

            assert lc.metadata["time_sampling"] == time_sampling

            if time_sampling == "uniform":
                # Check that time points are uniformly spaced
                time_diffs = np.diff(lc.time)
                assert np.allclose(time_diffs, time_diffs[0], rtol=1e-10)
            elif time_sampling == "random":
                # Random should have varying intervals
                time_diffs = np.diff(lc.time)
                assert np.std(time_diffs) > 0
            elif time_sampling == "irregular":
                # Irregular should have clusters (high variation in spacing)
                time_diffs = np.diff(lc.time)
                assert np.std(time_diffs) > np.mean(time_diffs) * 0.1

        check_time_sampling()

    def test_chained_strategies(self) -> None:
        """Test using the flexible baseline in a chain."""

        @given(data=st.data())
        @settings(max_examples=5)
        def check_chained(data: st.DataObject) -> None:
            # Create a custom baseline strategy
            custom_baseline = baseline_lightcurves(
                n_points=st.integers(100, 200),
                baseline_type=st.sampled_from(["flat", "smooth"]),
                baseline_flux=st.floats(50, 150),
            )

            # Use it to generate a lightcurve
            lc = data.draw(custom_baseline)

            assert 100 <= lc.n_points <= 200
            assert lc.metadata["baseline_type"] in ["flat", "smooth"]
            assert 50 <= lc.metadata["baseline_flux"] <= 150

        check_chained()
