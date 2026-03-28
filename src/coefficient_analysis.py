"""Coefficient significance analysis with statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

from config import save_dataframe, save_text
from data_preparation import coefficient_names


def _build_formula(columns: list[str]) -> str:
    """Build a patsy formula while preserving original column names."""
    quoted_terms = " + ".join(f'Q("{column}")' for column in columns)
    return f"wage ~ {quoted_terms}"


def _r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared without external helpers."""
    sc_total = float(np.sum(np.square(y - np.mean(y))))
    sc_res = float(np.sum(np.square(y - y_pred)))
    return 1.0 - sc_res / sc_total


def analyze_coefficients(
    x: pd.DataFrame,
    y: np.ndarray,
    manual_predictions: np.ndarray,
    statsmodels_model: RegressionResultsWrapper,
    sklearn_predictions: np.ndarray,
) -> None:
    """Save coefficient-level significance analyses."""
    formula_data = x.copy()
    formula_data["wage"] = y

    formula = _build_formula(x.columns.tolist())
    formula_model = smf.ols(formula, data=formula_data).fit()
    anova_type1 = sm.stats.anova_lm(formula_model, typ=1).reset_index()
    anova_type1 = anova_type1.rename(columns={"index": "Variable"})
    anova_type1["Variable"] = (
        anova_type1["Variable"]
        .astype(str)
        .str.removeprefix('Q("')
        .str.removesuffix('")')
    )
    save_dataframe("statsmodels_anova_type1.txt", anova_type1, float_digits=6)

    names = coefficient_names(x)
    coefficient_table = pd.DataFrame(
        {
            "Variable": names,
            "Coefficient": statsmodels_model.params.to_numpy(),
            "t_stat": statsmodels_model.tvalues.to_numpy(),
            "p_value": statsmodels_model.pvalues.to_numpy(),
        }
    )
    save_dataframe(
        "statsmodels_coefficients_pvalues.txt",
        coefficient_table,
        float_digits=6,
    )

    significant = coefficient_table[
        (coefficient_table["Variable"] != "Intercept")
        & (coefficient_table["p_value"] < 0.05)
    ].sort_values("p_value")
    non_significant = coefficient_table[
        (coefficient_table["Variable"] != "Intercept")
        & (coefficient_table["p_value"] >= 0.05)
    ].sort_values("p_value")

    if significant.empty:
        save_text(
            "significant_variables.txt", "Aucune variable significative au seuil de 5%."
        )
    else:
        save_dataframe("significant_variables.txt", significant, float_digits=6)

    if non_significant.empty:
        save_text(
            "non_significant_variables.txt",
            "Toutes les variables sont significatives au seuil de 5%.",
        )
    else:
        save_dataframe(
            "non_significant_variables.txt",
            non_significant,
            float_digits=6,
        )

    r2_manual = _r_squared(y, manual_predictions)
    r2_sklearn = _r_squared(y, sklearn_predictions)
    r2_statsmodels = float(statsmodels_model.rsquared)

    save_text(
        "r_squared_comparison.txt",
        "\n".join(
            [
                f"R^2 manuel : {r2_manual:.6f}",
                f"R^2 statsmodels : {r2_statsmodels:.6f}",
                f"R^2 scikit-learn : {r2_sklearn:.6f}",
            ]
        ),
    )
    save_text(
        "model_overall_stats.txt",
        "\n".join(
            [
                f"R^2 : {statsmodels_model.rsquared:.6f}",
                f"R^2 ajusté : {statsmodels_model.rsquared_adj:.6f}",
                f"Statistique F globale : {float(statsmodels_model.fvalue):.6f}",
                f"p-valeur du test global : {float(statsmodels_model.f_pvalue):.6e}",
                f"AIC : {statsmodels_model.aic:.6f}",
                f"BIC : {statsmodels_model.bic:.6f}",
            ]
        ),
    )
