from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, RANDOM_STATE, TEST_SIZE


TARGET_CANDIDATES = ("Target", "target", "Status", "status", "Class", "class")
CATEGORICAL_CODE_FEATURES = {
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
}


@dataclass(frozen=True)
class PreparedData:
    x_train: Any
    x_test: Any
    y_train: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    feature_names: list[str] | None


def find_csv(input_path: str | None = None) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return path

    csv_files = sorted(DATA_RAW_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {DATA_RAW_DIR}. Download the Kaggle dataset and put it there."
        )
    return csv_files[0]


def infer_target_column(df: pd.DataFrame, target: str | None = None) -> str:
    if target:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")
        return target

    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not infer target column. Pass --target explicitly. "
        f"Available columns: {list(df.columns)}"
    )


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    return df


def build_preprocessor(df: pd.DataFrame, use_pca: bool = False, pca_components: float | int = 0.95) -> Pipeline:
    categorical_features = [
        column
        for column in df.columns
        if column in CATEGORICAL_CODE_FEATURES or not pd.api.types.is_numeric_dtype(df[column])
    ]
    numeric_features = [column for column in df.columns if column not in categorical_features]

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    steps: list[tuple[str, Any]] = [("transformer", transformer)]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)))
    return Pipeline(steps)


def prepare_dataset(
    input_path: str | None = None,
    target: str | None = None,
    use_pca: bool = False,
    apply_smote: bool = False,
) -> PreparedData:
    csv_path = find_csv(input_path)
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = clean_column_names(df)
    target_col = infer_target_column(df, target)

    df = df.drop_duplicates().copy()
    y_raw = df[target_col].astype(str)
    x = df.drop(columns=[target_col])

    class_names = sorted(y_raw.unique().tolist())
    class_to_id = {label: index for index, label in enumerate(class_names)}
    y = y_raw.map(class_to_id).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(x_train, use_pca=use_pca)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    if apply_smote:
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError("Install imbalanced-learn to use --smote.") from exc

        smote = SMOTE(random_state=RANDOM_STATE)
        x_train_processed, y_train = smote.fit_resample(x_train_processed, y_train)

    feature_names = None
    if not use_pca:
        try:
            feature_names = preprocessor.named_steps["transformer"].get_feature_names_out().tolist()
        except Exception:
            feature_names = None

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, DATA_PROCESSED_DIR / "preprocessor.joblib")
    joblib.dump(
        {
            "x_train": x_train_processed,
            "x_test": x_test_processed,
            "y_train": y_train,
            "y_test": y_test,
            "class_names": class_names,
            "feature_names": feature_names,
            "source_csv": str(csv_path),
            "target_column": target_col,
            "categorical_features": preprocessor.named_steps["transformer"].transformers_[1][2],
            "numeric_features": preprocessor.named_steps["transformer"].transformers_[0][2],
        },
        DATA_PROCESSED_DIR / "prepared_data.joblib",
    )

    return PreparedData(
        x_train=x_train_processed,
        x_test=x_test_processed,
        y_train=y_train,
        y_test=y_test,
        class_names=class_names,
        feature_names=feature_names,
    )


def load_prepared_data() -> PreparedData:
    path = DATA_PROCESSED_DIR / "prepared_data.joblib"
    if not path.exists():
        raise FileNotFoundError("Prepared data not found. Run: python scripts/prepare_data.py")

    data = joblib.load(path)
    return PreparedData(
        x_train=data["x_train"],
        x_test=data["x_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        class_names=data["class_names"],
        feature_names=data.get("feature_names"),
    )
