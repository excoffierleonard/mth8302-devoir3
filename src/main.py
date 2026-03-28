"""Main orchestrator for MTH8302 Devoir 3."""

from __future__ import annotations

from anova import compute_anova, save_anova_outputs
from coefficient_analysis import analyze_coefficients
from data_preparation import load_and_prepare_data
from library_comparison import compare_libraries
from manual_regression import fit_manual_regression
from residual_analysis import analyze_residuals


def main() -> None:
    """Run every analysis required for the assignment."""
    data, x, y = load_and_prepare_data()

    manual_result = fit_manual_regression(x, y)
    library_result = compare_libraries(x, y, manual_result.coefficients)

    sklearn_predictions = library_result.sklearn_model.predict(x)
    anova_manual = compute_anova(y, manual_result.predictions, p=x.shape[1])
    anova_sklearn = compute_anova(y, sklearn_predictions, p=x.shape[1])
    save_anova_outputs(anova_manual, anova_sklearn)

    analyze_coefficients(
        x=x,
        y=y,
        manual_predictions=manual_result.predictions,
        statsmodels_model=library_result.statsmodels_model,
        sklearn_predictions=sklearn_predictions,
    )
    analyze_residuals(
        age=data["age"].to_numpy(dtype=float),
        y=y,
        multiple_predictions=manual_result.predictions,
    )


if __name__ == "__main__":
    main()
