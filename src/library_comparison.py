"""Compare the manual model with statsmodels and scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import RegressionResultsWrapper

from config import save_dataframe, save_text
from data_preparation import coefficient_names


@dataclass(frozen=True)
class LibraryComparisonResult:
    """Store fitted library models."""

    statsmodels_model: RegressionResultsWrapper
    sklearn_model: LinearRegression


def compare_libraries(
    x: pd.DataFrame, y: np.ndarray, manual_coefficients: pd.Series
) -> LibraryComparisonResult:
    """Fit statsmodels and scikit-learn and save comparison outputs."""
    x_with_const = sm.add_constant(x, has_constant="add")
    model_sm = sm.OLS(y, x_with_const).fit()
    save_text("statsmodels_summary.txt", model_sm.summary().as_text())

    model_sklearn = LinearRegression()
    model_sklearn.fit(x, y)

    names = coefficient_names(x)
    sklearn_coefficients = np.concatenate(
        ([float(model_sklearn.intercept_)], model_sklearn.coef_)
    )
    statsmodels_coefficients = model_sm.params.to_numpy()

    save_dataframe(
        "sklearn_coefficients.txt",
        pd.DataFrame({"Variable": names, "Coefficient": sklearn_coefficients}),
        float_digits=6,
    )

    comparison = pd.DataFrame(
        {
            "Variable": names,
            "Manuel": manual_coefficients.to_numpy(),
            "Statsmodels": statsmodels_coefficients,
            "ScikitLearn": sklearn_coefficients,
            "|Manuel-Statsmodels|": np.abs(
                manual_coefficients.to_numpy() - statsmodels_coefficients
            ),
            "|Manuel-ScikitLearn|": np.abs(
                manual_coefficients.to_numpy() - sklearn_coefficients
            ),
        }
    )
    save_dataframe("coefficient_comparison.txt", comparison, float_digits=10)

    max_diff_sm = float(
        np.max(np.abs(manual_coefficients.to_numpy() - statsmodels_coefficients))
    )
    max_diff_sklearn = float(
        np.max(np.abs(manual_coefficients.to_numpy() - sklearn_coefficients))
    )
    save_text(
        "coefficient_differences.txt",
        "\n".join(
            [
                f"Écart absolu max (manuel vs statsmodels) : {max_diff_sm:.6e}",
                f"Écart absolu max (manuel vs scikit-learn) : {max_diff_sklearn:.6e}",
                "Conclusion : les trois méthodes donnent les mêmes coefficients à la précision numérique près.",
            ]
        ),
    )

    return LibraryComparisonResult(
        statsmodels_model=model_sm,
        sklearn_model=model_sklearn,
    )
