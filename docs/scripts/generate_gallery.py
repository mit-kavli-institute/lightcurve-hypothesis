#!/usr/bin/env python
"""Generate gallery of lightcurve examples for documentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from hypothesis_lightcurves.generators import (
    baseline_lightcurves,
    lightcurves,
    periodic_lightcurves,
    transient_lightcurves,
)
from hypothesis_lightcurves.modifiers import add_gaps, add_noise, add_outliers, add_trend
from hypothesis_lightcurves.visualization import (
    create_gallery_figure,
    plot_lightcurve,
    plot_lightcurve_comparison,
)

# Use non-interactive backend for headless generation
matplotlib.use("Agg")

# Output directories
GALLERY_DIR = Path(__file__).parent.parent / "source" / "_static" / "gallery"
GALLERY_DATA_DIR = GALLERY_DIR / "data"
GALLERY_IMAGES_DIR = GALLERY_DIR / "images"
GALLERY_CODE_DIR = GALLERY_DIR / "code"


def ensure_directories() -> None:
    """Create gallery directories if they don't exist."""
    for directory in [GALLERY_DIR, GALLERY_DATA_DIR, GALLERY_IMAGES_DIR, GALLERY_CODE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def save_example(
    name: str,
    lc: Any,
    code: str,
    description: str,
    category: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a lightcurve example with plot, code, and metadata.

    Parameters
    ----------
    name : str
        Name of the example (used for filenames).
    lc : Lightcurve
        The lightcurve object to save.
    code : str
        Python code to generate the example.
    description : str
        Description of the example.
    category : str
        Category of the example (e.g., "basic", "periodic").
    parameters : dict, optional
        Parameters used to generate the example.

    Returns
    -------
    dict
        Metadata about the saved example.
    """
    # Create and save plot
    fig, ax = plot_lightcurve(lc, title=description)

    # Add parameter text if provided
    if parameters:
        param_text = "\n".join([f"{k}: {v}" for k, v in parameters.items()])
        ax.text(
            0.02,
            0.98,
            param_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    fig.tight_layout()
    image_path = GALLERY_IMAGES_DIR / f"{name}.png"
    fig.savefig(image_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Save code snippet
    code_path = GALLERY_CODE_DIR / f"{name}.py"
    code_path.write_text(code)

    # Save data as numpy arrays
    data_path = GALLERY_DATA_DIR / f"{name}.npz"
    np.savez_compressed(
        data_path,
        time=lc.time,
        flux=lc.flux,
        flux_err=lc.flux_err if lc.flux_err is not None else np.array([]),
    )

    # Create metadata
    metadata = {
        "name": name,
        "description": description,
        "category": category,
        "parameters": parameters or {},
        "n_points": lc.n_points,
        "duration": lc.duration,
        "mean_flux": lc.mean_flux,
        "std_flux": lc.std_flux,
        "has_errors": lc.flux_err is not None,
        "modifications": lc.modifications,
        "image_path": str(image_path.relative_to(GALLERY_DIR.parent.parent)),
        "code_path": str(code_path.relative_to(GALLERY_DIR.parent.parent)),
        "data_path": str(data_path.relative_to(GALLERY_DIR.parent.parent)),
    }

    return metadata


def generate_basic_examples() -> list[dict[str, Any]]:
    """Generate basic lightcurve examples."""
    examples = []

    # Simple random lightcurve
    lc = lightcurves().example()
    code = """from hypothesis_lightcurves.generators import lightcurves

# Generate a basic random lightcurve
lc = lightcurves().example()
fig, ax = lc.plot(title="Basic Random Lightcurve")"""
    examples.append(
        save_example(
            "basic_random",
            lc,
            code,
            "Basic Random Lightcurve",
            "basic",
            {"type": "random", "n_points": lc.n_points},
        )
    )

    # Lightcurve with errors
    lc = lightcurves(with_errors=True).example()
    code = """from hypothesis_lightcurves.generators import lightcurves

# Generate a lightcurve with error bars
lc = lightcurves(with_errors=True).example()
fig, ax = lc.plot(title="Lightcurve with Errors")"""
    examples.append(
        save_example(
            "basic_with_errors",
            lc,
            code,
            "Lightcurve with Error Bars",
            "basic",
            {"type": "with_errors", "n_points": lc.n_points},
        )
    )

    return examples


def generate_baseline_examples() -> list[dict[str, Any]]:
    """Generate baseline lightcurve examples."""
    examples = []

    # Flat baseline
    lc = baseline_lightcurves(baseline_type="flat", n_points=200).example()
    code = """from hypothesis_lightcurves.generators import baseline_lightcurves

# Generate a flat baseline lightcurve
lc = baseline_lightcurves(baseline_type="flat", n_points=200).example()
fig, ax = lc.plot(title="Flat Baseline")"""
    examples.append(
        save_example(
            "baseline_flat",
            lc,
            code,
            "Flat Baseline",
            "baseline",
            {"type": "flat", "n_points": 200},
        )
    )

    # Random walk baseline
    lc = baseline_lightcurves(baseline_type="random_walk", n_points=300).example()
    code = """from hypothesis_lightcurves.generators import baseline_lightcurves

# Generate a random walk baseline
lc = baseline_lightcurves(baseline_type="random_walk", n_points=300).example()
fig, ax = lc.plot(title="Random Walk Baseline")"""
    examples.append(
        save_example(
            "baseline_random_walk",
            lc,
            code,
            "Random Walk Baseline",
            "baseline",
            {"type": "random_walk", "n_points": 300},
        )
    )

    # Smooth baseline
    lc = baseline_lightcurves(baseline_type="smooth", n_points=250).example()
    code = """from hypothesis_lightcurves.generators import baseline_lightcurves

# Generate a smooth baseline
lc = baseline_lightcurves(baseline_type="smooth", n_points=250).example()
fig, ax = lc.plot(title="Smooth Baseline")"""
    examples.append(
        save_example(
            "baseline_smooth",
            lc,
            code,
            "Smooth Baseline",
            "baseline",
            {"type": "smooth", "n_points": 250},
        )
    )

    return examples


def generate_periodic_examples() -> list[dict[str, Any]]:
    """Generate periodic lightcurve examples."""
    examples = []

    # Simple periodic (without noise)
    lc = periodic_lightcurves(
        min_points=200,
        max_points=200,
        min_period=2.4,
        max_period=2.6,
        min_amplitude=0.09,
        max_amplitude=0.11,
        with_noise=False,
    ).example()
    code = """from hypothesis_lightcurves.generators import periodic_lightcurves

# Generate a simple periodic lightcurve
lc = periodic_lightcurves(
    min_points=200, max_points=200,
    min_period=2.4, max_period=2.6,
    min_amplitude=0.09, max_amplitude=0.11,
    with_noise=False
).example()
fig, ax = lc.plot(title="Simple Periodic Signal")"""

    # Extract actual period from metadata if available
    period_val = lc.metadata.get("period", 2.5) if lc.metadata else 2.5
    amplitude_val = lc.metadata.get("amplitude", 0.1) if lc.metadata else 0.1

    examples.append(
        save_example(
            "periodic_simple",
            lc,
            code,
            "Simple Periodic Signal",
            "periodic",
            {"period": period_val, "amplitude": amplitude_val, "n_points": 200},
        )
    )

    # Periodic with noise (default)
    lc = periodic_lightcurves(
        min_points=300,
        max_points=300,
        min_period=0.9,
        max_period=1.1,
        min_amplitude=0.15,
        max_amplitude=0.25,
        with_noise=True,
    ).example()
    code = """from hypothesis_lightcurves.generators import periodic_lightcurves

# Generate a periodic lightcurve with noise
lc = periodic_lightcurves(
    min_points=300, max_points=300,
    min_period=0.9, max_period=1.1,
    min_amplitude=0.15, max_amplitude=0.25,
    with_noise=True
).example()
fig, ax = lc.plot(title="Periodic with Noise")"""

    period_val = lc.metadata.get("period", 1.0) if lc.metadata else 1.0
    amplitude_val = lc.metadata.get("amplitude", 0.2) if lc.metadata else 0.2

    examples.append(
        save_example(
            "periodic_with_noise",
            lc,
            code,
            "Periodic Signal with Noise",
            "periodic",
            {"period": period_val, "amplitude": amplitude_val, "with_noise": True, "n_points": 300},
        )
    )

    # Multi-periodic (combine two signals)
    lc1 = periodic_lightcurves(
        min_points=500,
        max_points=500,
        min_period=0.45,
        max_period=0.55,
        min_amplitude=0.08,
        max_amplitude=0.12,
        with_noise=False,
    ).example()
    lc2 = periodic_lightcurves(
        min_points=500,
        max_points=500,
        min_period=1.2,
        max_period=1.4,
        min_amplitude=0.04,
        max_amplitude=0.06,
        with_noise=False,
    ).example()

    # Ensure same time array
    lc2.time = lc1.time.copy()
    lc1.flux = lc1.flux + lc2.flux - lc2.mean_flux

    code = """from hypothesis_lightcurves.generators import periodic_lightcurves

# Generate a multi-periodic signal by combining two periodic signals
lc1 = periodic_lightcurves(
    min_points=500, max_points=500,
    min_period=0.45, max_period=0.55,
    min_amplitude=0.08, max_amplitude=0.12,
    with_noise=False
).example()
lc2 = periodic_lightcurves(
    min_points=500, max_points=500,
    min_period=1.2, max_period=1.4,
    min_amplitude=0.04, max_amplitude=0.06,
    with_noise=False
).example()

# Combine the signals
lc2.time = lc1.time.copy()
lc1.flux = lc1.flux + lc2.flux - lc2.mean_flux
fig, ax = lc1.plot(title="Multi-periodic Signal")"""

    examples.append(
        save_example(
            "periodic_multi",
            lc1,
            code,
            "Multi-periodic Signal",
            "periodic",
            {"combined": True, "n_points": 500},
        )
    )

    return examples


def generate_transient_examples() -> list[dict[str, Any]]:
    """Generate transient lightcurve examples."""
    examples = []

    # Supernova-like transient
    lc = transient_lightcurves(
        min_points=200,
        max_points=200,
        min_peak_time=45.0,
        max_peak_time=55.0,
        min_rise_time=8.0,
        max_rise_time=12.0,
        min_decay_time=25.0,
        max_decay_time=35.0,
    ).example()
    code = """from hypothesis_lightcurves.generators import transient_lightcurves

# Generate a supernova-like transient
lc = transient_lightcurves(
    min_points=200, max_points=200,
    min_peak_time=45.0, max_peak_time=55.0,
    min_rise_time=8.0, max_rise_time=12.0,
    min_decay_time=25.0, max_decay_time=35.0
).example()
fig, ax = lc.plot(title="Supernova-like Transient")"""
    examples.append(
        save_example(
            "transient_supernova",
            lc,
            code,
            "Supernova-like Transient",
            "transient",
            {
                "type": "supernova-like",
                "peak_time_range": "45-55",
                "rise_time_range": "8-12",
                "decay_time_range": "25-35",
            },
        )
    )

    # Flare transient (short timescales)
    lc = transient_lightcurves(
        min_points=150,
        max_points=150,
        min_peak_time=20.0,
        max_peak_time=30.0,
        min_rise_time=0.3,
        max_rise_time=0.7,
        min_decay_time=1.5,
        max_decay_time=2.5,
    ).example()
    code = """from hypothesis_lightcurves.generators import transient_lightcurves

# Generate a stellar flare (short timescales)
lc = transient_lightcurves(
    min_points=150, max_points=150,
    min_peak_time=20.0, max_peak_time=30.0,
    min_rise_time=0.3, max_rise_time=0.7,
    min_decay_time=1.5, max_decay_time=2.5
).example()
fig, ax = lc.plot(title="Stellar Flare")"""
    examples.append(
        save_example(
            "transient_flare",
            lc,
            code,
            "Stellar Flare",
            "transient",
            {
                "type": "flare",
                "peak_time_range": "20-30",
                "rise_time_range": "0.3-0.7",
                "decay_time_range": "1.5-2.5",
            },
        )
    )

    # Eclipse-like dip (symmetric rise/decay)
    lc = transient_lightcurves(
        min_points=200,
        max_points=200,
        min_peak_time=25.0,
        max_peak_time=35.0,
        min_rise_time=1.5,
        max_rise_time=2.5,
        min_decay_time=1.5,
        max_decay_time=2.5,
    ).example()
    # Invert to create a dip
    lc.flux = 2 * lc.mean_flux - lc.flux
    code = """from hypothesis_lightcurves.generators import transient_lightcurves

# Generate an eclipse-like dip
lc = transient_lightcurves(
    min_points=200, max_points=200,
    min_peak_time=25.0, max_peak_time=35.0,
    min_rise_time=1.5, max_rise_time=2.5,
    min_decay_time=1.5, max_decay_time=2.5
).example()
# Invert to create a dip
lc.flux = 2 * lc.mean_flux - lc.flux
fig, ax = lc.plot(title="Eclipse-like Dip")"""
    examples.append(
        save_example(
            "transient_eclipse",
            lc,
            code,
            "Eclipse-like Dip",
            "transient",
            {"type": "eclipse-like", "peak_time_range": "25-35", "symmetric": True},
        )
    )

    return examples


def generate_modified_examples() -> list[dict[str, Any]]:
    """Generate modified lightcurve examples."""
    examples = []

    # Lightcurve with gaps
    base_lc = periodic_lightcurves(
        min_points=300,
        max_points=300,
        min_period=1.4,
        max_period=1.6,
        min_amplitude=0.14,
        max_amplitude=0.16,
        with_noise=False,
    ).example()
    lc_with_gaps = add_gaps(base_lc, n_gaps=3, gap_fraction=0.1)

    # Save comparison plot
    fig, axes = plot_lightcurve_comparison(
        base_lc, lc_with_gaps, label1="Original", label2="With Gaps", title="Adding Data Gaps"
    )
    image_path = GALLERY_IMAGES_DIR / "modified_gaps.png"
    fig.savefig(image_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    code = """from hypothesis_lightcurves.generators import periodic_lightcurves
from hypothesis_lightcurves.modifiers import add_gaps

# Start with a periodic signal
base_lc = periodic_lightcurves(
    min_points=300, max_points=300,
    min_period=1.4, max_period=1.6,
    min_amplitude=0.14, max_amplitude=0.16,
    with_noise=False
).example()

# Add data gaps
lc_with_gaps = add_gaps(base_lc, n_gaps=3, gap_fraction=0.1)
fig, ax = lc_with_gaps.plot(title="Lightcurve with Gaps")"""
    examples.append(
        {
            "name": "modified_gaps",
            "description": "Lightcurve with Data Gaps",
            "category": "modified",
            "parameters": {"n_gaps": 3, "gap_fraction": 0.1},
            "n_points": lc_with_gaps.n_points,
            "duration": lc_with_gaps.duration,
            "mean_flux": lc_with_gaps.mean_flux,
            "std_flux": lc_with_gaps.std_flux,
            "has_errors": lc_with_gaps.flux_err is not None,
            "modifications": lc_with_gaps.modifications,
            "image_path": str(image_path.relative_to(GALLERY_DIR.parent.parent)),
            "code_path": str(
                (GALLERY_CODE_DIR / "modified_gaps.py").relative_to(GALLERY_DIR.parent.parent)
            ),
        }
    )
    (GALLERY_CODE_DIR / "modified_gaps.py").write_text(code)

    # Lightcurve with outliers
    base_lc = baseline_lightcurves(baseline_type="smooth", n_points=200).example()
    lc_with_outliers = add_outliers(base_lc, fraction=0.05, amplitude=5.0)

    fig, axes = plot_lightcurve_comparison(
        base_lc,
        lc_with_outliers,
        label1="Original",
        label2="With Outliers",
        title="Adding Outliers",
    )
    image_path = GALLERY_IMAGES_DIR / "modified_outliers.png"
    fig.savefig(image_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    code = """from hypothesis_lightcurves.generators import baseline_lightcurves
from hypothesis_lightcurves.modifiers import add_outliers

# Start with a smooth baseline
base_lc = baseline_lightcurves(baseline_type="smooth", n_points=200).example()

# Add outliers
lc_with_outliers = add_outliers(base_lc, fraction=0.05, amplitude=5.0)
fig, ax = lc_with_outliers.plot(title="Lightcurve with Outliers")"""
    examples.append(
        {
            "name": "modified_outliers",
            "description": "Lightcurve with Outliers",
            "category": "modified",
            "parameters": {"outlier_fraction": 0.05, "outlier_amplitude": 5.0},
            "n_points": lc_with_outliers.n_points,
            "duration": lc_with_outliers.duration,
            "mean_flux": lc_with_outliers.mean_flux,
            "std_flux": lc_with_outliers.std_flux,
            "has_errors": lc_with_outliers.flux_err is not None,
            "modifications": lc_with_outliers.modifications,
            "image_path": str(image_path.relative_to(GALLERY_DIR.parent.parent)),
            "code_path": str(
                (GALLERY_CODE_DIR / "modified_outliers.py").relative_to(GALLERY_DIR.parent.parent)
            ),
        }
    )
    (GALLERY_CODE_DIR / "modified_outliers.py").write_text(code)

    # Lightcurve with trend
    base_lc = periodic_lightcurves(
        min_points=250,
        max_points=250,
        min_period=0.7,
        max_period=0.9,
        min_amplitude=0.09,
        max_amplitude=0.11,
        with_noise=False,
    ).example()
    lc_with_trend = add_trend(base_lc, trend_type="quadratic", coefficient=0.2)

    fig, axes = plot_lightcurve_comparison(
        base_lc,
        lc_with_trend,
        label1="Original",
        label2="With Trend",
        title="Adding Quadratic Trend",
    )
    image_path = GALLERY_IMAGES_DIR / "modified_trend.png"
    fig.savefig(image_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    code = """from hypothesis_lightcurves.generators import periodic_lightcurves
from hypothesis_lightcurves.modifiers import add_trend

# Start with a periodic signal
base_lc = periodic_lightcurves(
    min_points=250, max_points=250,
    min_period=0.7, max_period=0.9,
    min_amplitude=0.09, max_amplitude=0.11,
    with_noise=False
).example()

# Add quadratic trend
lc_with_trend = add_trend(base_lc, trend_type="quadratic", coefficient=0.2)
fig, ax = lc_with_trend.plot(title="Lightcurve with Trend")"""
    examples.append(
        {
            "name": "modified_trend",
            "description": "Lightcurve with Quadratic Trend",
            "category": "modified",
            "parameters": {"trend_type": "quadratic", "coefficient": 0.2},
            "n_points": lc_with_trend.n_points,
            "duration": lc_with_trend.duration,
            "mean_flux": lc_with_trend.mean_flux,
            "std_flux": lc_with_trend.std_flux,
            "has_errors": lc_with_trend.flux_err is not None,
            "modifications": lc_with_trend.modifications,
            "image_path": str(image_path.relative_to(GALLERY_DIR.parent.parent)),
            "code_path": str(
                (GALLERY_CODE_DIR / "modified_trend.py").relative_to(GALLERY_DIR.parent.parent)
            ),
        }
    )
    (GALLERY_CODE_DIR / "modified_trend.py").write_text(code)

    return examples


def generate_composite_examples() -> list[dict[str, Any]]:
    """Generate composite/complex lightcurve examples."""
    examples = []

    # Realistic observational data - create manually with sequential modifications
    base_lc = periodic_lightcurves(
        min_points=400,
        max_points=400,
        min_period=1.1,
        max_period=1.3,
        min_amplitude=0.07,
        max_amplitude=0.09,
        with_noise=False,
    ).example()

    # Apply modifications sequentially
    lc = add_noise(base_lc, level=0.01)
    lc = add_gaps(lc, n_gaps=2, gap_fraction=0.08)
    lc = add_outliers(lc, fraction=0.02, amplitude=4)

    code = """from hypothesis_lightcurves.generators import periodic_lightcurves
from hypothesis_lightcurves.modifiers import add_noise, add_gaps, add_outliers

# Generate realistic observational data with multiple effects
base_lc = periodic_lightcurves(
    min_points=400, max_points=400,
    min_period=1.1, max_period=1.3,
    min_amplitude=0.07, max_amplitude=0.09,
    with_noise=False
).example()

# Apply modifications sequentially
lc = add_noise(base_lc, level=0.01)
lc = add_gaps(lc, n_gaps=2, gap_fraction=0.08)
lc = add_outliers(lc, fraction=0.02, amplitude=4)

fig, ax = lc.plot(title="Realistic Observational Data")"""
    examples.append(
        save_example(
            "composite_realistic",
            lc,
            code,
            "Realistic Observational Data",
            "composite",
            {
                "base": "periodic",
                "modifications": ["noise", "gaps", "outliers"],
                "n_points": lc.n_points,
            },
        )
    )

    return examples


def generate_overview_figure(all_examples: list[dict[str, Any]]) -> None:
    """Generate an overview gallery figure."""
    # Select representative examples
    selected = {
        "Basic": lightcurves().example(),
        "Flat Baseline": baseline_lightcurves(baseline_type="flat").example(),
        "Random Walk": baseline_lightcurves(baseline_type="random_walk").example(),
        "Periodic": periodic_lightcurves(
            min_period=0.9, max_period=1.1, min_amplitude=0.14, max_amplitude=0.16
        ).example(),
        "Supernova": transient_lightcurves().example(),
        "With Gaps": add_gaps(periodic_lightcurves().example(), n_gaps=3),
    }

    fig = create_gallery_figure(selected, title="Lightcurve Generator Gallery Overview")
    fig.savefig(GALLERY_IMAGES_DIR / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all gallery examples."""
    print("Generating lightcurve gallery...")

    # Ensure directories exist
    ensure_directories()

    # Generate all examples
    all_examples = []

    print("  Generating basic examples...")
    all_examples.extend(generate_basic_examples())

    print("  Generating baseline examples...")
    all_examples.extend(generate_baseline_examples())

    print("  Generating periodic examples...")
    all_examples.extend(generate_periodic_examples())

    print("  Generating transient examples...")
    all_examples.extend(generate_transient_examples())

    print("  Generating modified examples...")
    all_examples.extend(generate_modified_examples())

    print("  Generating composite examples...")
    all_examples.extend(generate_composite_examples())

    # Generate overview figure
    print("  Generating overview figure...")
    generate_overview_figure(all_examples)

    # Save metadata index
    metadata_path = GALLERY_DIR / "index.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "examples": all_examples,
                "categories": {
                    "basic": "Basic Lightcurves",
                    "baseline": "Baseline Patterns",
                    "periodic": "Periodic Signals",
                    "transient": "Transient Events",
                    "modified": "Modified Lightcurves",
                    "composite": "Composite Examples",
                },
                "total_examples": len(all_examples),
            },
            f,
            indent=2,
        )

    print(f"✅ Generated {len(all_examples)} examples in {GALLERY_DIR}")
    print(f"   - Images: {GALLERY_IMAGES_DIR}")
    print(f"   - Code: {GALLERY_CODE_DIR}")
    print(f"   - Data: {GALLERY_DATA_DIR}")
    print(f"   - Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
