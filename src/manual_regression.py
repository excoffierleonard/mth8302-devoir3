"""Manual multiple regression with the normal equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import save_dataframe, save_text
from data_preparation import build_design_matrix, coefficient_names


@dataclass(frozen=True)
class ManualRegressionResult:
    """Minimal output of the manual regression fit."""

    coefficients: pd.Series
    predictions: np.ndarray


def fit_manual_regression(x: pd.DataFrame, y: np.ndarray) -> ManualRegressionResult:
    """Fit the multiple-regression model manually and save coefficients."""
    design_matrix = build_design_matrix(x)
    xtx = design_matrix.T @ design_matrix
    xty = design_matrix.T @ y
    beta = np.linalg.solve(xtx, xty)

    names = coefficient_names(x)
    coefficients = pd.Series(beta, index=names, name="Coefficient")

    save_dataframe(
        "manual_coefficients.txt",
        pd.DataFrame({"Variable": names, "Coefficient": beta}),
        float_digits=6,
    )
    save_text(
        "manual_matrix_formula.txt",
        "\n".join(
            [
                "Estimateur MCO utilisé : beta_hat = (X^T X)^(-1) X^T Y",
                "Implémentation numérique : résolution de (X^T X) beta = X^T Y avec numpy.linalg.solve.",
            ]
        ),
    )

    predictions = design_matrix @ beta
    return ManualRegressionResult(coefficients=coefficients, predictions=predictions)
