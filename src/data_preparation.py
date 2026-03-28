"""Load Wage data and create the multiple-regression design matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_PATH, save_text


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load the dataset, save exploration files, and return X and y."""
    data = pd.read_csv(DATA_PATH)

    save_text("head.txt", data.head().to_string())
    save_text("columns.txt", "\n".join(data.columns.tolist()))
    save_text("dtypes.txt", data.dtypes.to_string())
    save_text(
        "dimensions.txt",
        "\n".join(
            [
                f"Nombre d'observations : {data.shape[0]}",
                f"Nombre de colonnes brutes : {data.shape[1]}",
            ]
        ),
    )
    save_text("missing_values.txt", data.isna().sum().to_string())

    x_raw = data.drop(columns=["wage", "logwage"])
    x = pd.get_dummies(x_raw, drop_first=True, dtype=float)
    y = data["wage"].astype(float).to_numpy()

    save_text(
        "design_matrix_info.txt",
        "\n".join(
            [
                f"Variables explicatives brutes : {x_raw.shape[1]}",
                f"Variables explicatives après encodage : {x.shape[1]}",
                f"Forme de X : {x.shape}",
                f"Forme de X avec intercept : ({len(y)}, {x.shape[1] + 1})",
                "",
                "Colonnes utilisées :",
                *x.columns.tolist(),
            ]
        ),
    )
    save_text(
        "encoding_notes.txt",
        "\n".join(
            [
                "Les variables catégorielles sont encodées avec pd.get_dummies(drop_first=True).",
                "La variable 'region' ne génère aucune colonne supplémentaire car elle ne contient qu'une seule modalité.",
            ]
        ),
    )
    return data, x, y


def build_design_matrix(x: pd.DataFrame) -> np.ndarray:
    """Return the design matrix with an intercept column."""
    return np.column_stack([np.ones(len(x), dtype=float), x.to_numpy(dtype=float)])


def coefficient_names(x: pd.DataFrame) -> list[str]:
    """Return coefficient names in manual and library order."""
    return ["Intercept"] + x.columns.tolist()
