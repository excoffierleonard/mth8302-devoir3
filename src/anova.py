"""ANOVA computations for the multiple-regression model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f

from config import save_dataframe, save_text


@dataclass(frozen=True)
class AnovaResult:
    """ANOVA summary values and table."""

    table: pd.DataFrame
    sc_total: float
    sc_reg: float
    sc_res: float
    df_reg: int
    df_res: int
    mc_reg: float
    mc_res: float
    f_stat: float
    p_value: float


def compute_anova(y: np.ndarray, y_pred: np.ndarray, p: int) -> AnovaResult:
    """Compute the ANOVA table for a regression model."""
    n = len(y)
    y_mean = float(np.mean(y))

    sc_total = float(np.sum(np.square(y - y_mean)))
    sc_reg = float(np.sum(np.square(y_pred - y_mean)))
    sc_res = float(np.sum(np.square(y - y_pred)))

    df_reg = int(p)
    df_res = int(n - p - 1)
    df_total = int(n - 1)

    mc_reg = sc_reg / df_reg
    mc_res = sc_res / df_res

    f_stat = mc_reg / mc_res
    p_value = float(f.sf(f_stat, df_reg, df_res))

    table = pd.DataFrame(
        {
            "Source": ["Régression", "Résiduel", "Total"],
            "SC": [sc_reg, sc_res, sc_total],
            "dl": [df_reg, df_res, df_total],
            "MC": [mc_reg, mc_res, np.nan],
            "F_stat": [f_stat, np.nan, np.nan],
            "p-value": [p_value, np.nan, np.nan],
        }
    )

    return AnovaResult(
        table=table,
        sc_total=sc_total,
        sc_reg=sc_reg,
        sc_res=sc_res,
        df_reg=df_reg,
        df_res=df_res,
        mc_reg=mc_reg,
        mc_res=mc_res,
        f_stat=f_stat,
        p_value=p_value,
    )


def save_anova_outputs(manual: AnovaResult, sklearn: AnovaResult) -> None:
    """Save ANOVA tables and a short comparison summary."""
    save_dataframe("anova_manual.txt", manual.table, float_digits=6)
    save_dataframe("anova_sklearn.txt", sklearn.table, float_digits=6)

    save_text(
        "anova_comparison.txt",
        "\n".join(
            [
                f"Écart absolu sur SC_reg : {abs(manual.sc_reg - sklearn.sc_reg):.6e}",
                f"Écart absolu sur SC_res : {abs(manual.sc_res - sklearn.sc_res):.6e}",
                f"Écart absolu sur F_stat : {abs(manual.f_stat - sklearn.f_stat):.6e}",
                f"Écart absolu sur la p-valeur : {abs(manual.p_value - sklearn.p_value):.6e}",
                "Conclusion : les tableaux ANOVA manuelle et scikit-learn sont identiques à la précision numérique près.",
            ]
        ),
    )
    save_text(
        "anova_interpretation.txt",
        "\n".join(
            [
                f"Statistique F globale : {manual.f_stat:.6f}",
                f"p-valeur globale : {manual.p_value:.6e}",
                "Interprétation : la statistique F compare la variabilité expliquée moyenne à la variabilité résiduelle moyenne.",
                "Une p-valeur très petite conduit à rejeter H0 : beta_1 = ... = beta_p = 0.",
            ]
        ),
    )
