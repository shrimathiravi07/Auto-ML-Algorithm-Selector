"""Visualization helpers for the SmartML Model Selector project.

This module is responsible for turning model names and accuracy scores into
professional bar charts using Matplotlib. The charts are saved into the
project's static image folder so Flask can serve them to the browser.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib

# Use a non-interactive backend so the chart can be generated safely on the
# server without requiring a GUI display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _validate_inputs(algorithms: Sequence[str], accuracies: Sequence[float]) -> tuple[list[str], list[float]]:
    """Validate chart input and return normalized values.

    Args:
        algorithms: Names of the machine learning algorithms.
        accuracies: Accuracy values for each algorithm.

    Returns:
        A tuple containing a list of cleaned algorithm names and a list of
        numeric accuracy scores.

    Raises:
        ValueError: If the inputs are empty, mismatched in length, or invalid.
    """
    if not algorithms or not accuracies:
        raise ValueError("Algorithm names and accuracy scores must not be empty.")

    if len(algorithms) != len(accuracies):
        raise ValueError("Algorithm names and accuracy scores must have the same length.")

    cleaned_algorithms: list[str] = []
    cleaned_scores: list[float] = []

    for name, score in zip(algorithms, accuracies):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each algorithm name must be a non-empty string.")

        try:
            score_value = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError("All accuracy scores must be numeric values.") from exc

        if not 0 <= score_value <= 100:
            raise ValueError("Accuracy scores must be between 0 and 100.")

        cleaned_algorithms.append(name.strip())
        cleaned_scores.append(score_value)

    return cleaned_algorithms, cleaned_scores


def generate_accuracy_chart(
    algorithms: Sequence[str],
    accuracies: Sequence[float],
    output_filename: str = "model_accuracy.png",
    output_dir: str = "static/images",
    title: str = "Model Accuracy Comparison",
) -> str:
    """Create and save a bar chart for model accuracy scores.

    This function is designed to be reusable for Flask views, dashboards, or
    future reporting features. It validates the inputs, creates a clean chart,
    saves it to disk, and returns a Flask-compatible URL path.

    Args:
        algorithms: Names of the machine learning algorithms.
        accuracies: Accuracy scores for matching algorithms.
        output_filename: Name of the image file to save.
        output_dir: Directory where the chart should be stored.
        title: Title displayed on the chart.

    Returns:
        A URL path such as /static/images/model_accuracy.png.

    Raises:
        ValueError: If the inputs are invalid or the output filename is not a PNG.
    """
    if not output_filename.lower().endswith(".png"):
        raise ValueError("output_filename must end with .png")

    cleaned_algorithms, cleaned_scores = _validate_inputs(algorithms, accuracies)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / output_filename

    plt.figure(figsize=(10, 6), dpi=150)
    bars = plt.bar(cleaned_algorithms, cleaned_scores, color="#4C78A8", edgecolor="#2F4F4F")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Algorithm", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, cleaned_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    try:
        relative_path = file_path.relative_to(Path.cwd())
    except ValueError:
        relative_path = file_path

    return "/" + relative_path.as_posix().replace("\\", "/")


__all__ = ["generate_accuracy_chart"]
