"""Residual plots and simple-vs-multiple regression comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import OUTPUT_DIR, save_dataframe, save_text


def _r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared from predictions."""
    sc_total = float(np.sum(np.square(y - np.mean(y))))
    sc_res = float(np.sum(np.square(y - y_pred)))
    return 1.0 - sc_res / sc_total


def _fit_simple_age_baseline(
    age: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the assignment-2 simple regression baseline on age only."""
    x_simple = np.column_stack([np.ones(len(age), dtype=float), age])
    beta = np.linalg.solve(x_simple.T @ x_simple, x_simple.T @ y)
    predictions = x_simple @ beta
    return beta, predictions


def _plot_residuals(
    age: np.ndarray, residuals: np.ndarray, filename: str, title: str
) -> None:
    """Generate the standard residual diagnostics figure."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_title("Histogramme des résidus")
    axes[0].set_xlabel("Résidus")
    axes[0].set_ylabel("Fréquence")

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-Plot des résidus")

    axes[2].scatter(age, residuals, alpha=0.45)
    axes[2].axhline(0, color="red", linestyle="dashed", linewidth=1)
    axes[2].set_title("Résidus en fonction de l'âge")
    axes[2].set_xlabel("Âge")
    axes[2].set_ylabel("Résidus")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(figure)


def analyze_residuals(
    age: np.ndarray, y: np.ndarray, multiple_predictions: np.ndarray
) -> None:
    """Save residual diagnostics and compare with the simple baseline."""
    residuals = y - multiple_predictions
    _plot_residuals(
        age,
        residuals,
        "residuals_plot.png",
        "Analyse des résidus : régression linéaire multiple",
    )

    residual_summary = pd.DataFrame(
        {
            "Statistique": [
                "Moyenne",
                "Écart-type",
                "Minimum",
                "Q1",
                "Médiane",
                "Q3",
                "Maximum",
            ],
            "Valeur": [
                float(np.mean(residuals)),
                float(np.std(residuals, ddof=1)),
                float(np.min(residuals)),
                float(np.quantile(residuals, 0.25)),
                float(np.median(residuals)),
                float(np.quantile(residuals, 0.75)),
                float(np.max(residuals)),
            ],
        }
    )
    save_dataframe("residual_summary.txt", residual_summary, float_digits=6)

    beta_simple, simple_predictions = _fit_simple_age_baseline(age, y)
    r2_simple = _r_squared(y, simple_predictions)
    r2_multiple = _r_squared(y, multiple_predictions)

    rmse_simple = float(np.sqrt(np.mean(np.square(y - simple_predictions))))
    rmse_multiple = float(np.sqrt(np.mean(np.square(y - multiple_predictions))))

    save_text(
        "simple_vs_multiple.txt",
        "\n".join(
            [
                "Régression simple sur l'âge seulement :",
                f"beta_0 = {beta_simple[0]:.6f}",
                f"beta_1 = {beta_simple[1]:.6f}",
                f"R^2 = {r2_simple:.6f}",
                f"RMSE = {rmse_simple:.6f}",
                "",
                "Régression multiple :",
                f"R^2 = {r2_multiple:.6f}",
                f"RMSE = {rmse_multiple:.6f}",
                "",
                f"Gain de R^2 : {r2_multiple - r2_simple:.6f}",
                f"Réduction du RMSE : {rmse_simple - rmse_multiple:.6f}",
            ]
        ),
    )
