from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR, MODELS_DIR, RANDOM_STATE
from src.data import load_prepared_data


def write_confusion_matrices(class_names: list[str]) -> Path:
    output_path = METRICS_DIR / "confusion_matrices.csv"
    rows = []
    for path in sorted(METRICS_DIR.glob("*.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        matrix = metrics.get("confusion_matrix")
        if matrix is None:
            continue
        for actual_index, actual_name in enumerate(class_names):
            for predicted_index, predicted_name in enumerate(class_names):
                rows.append(
                    {
                        "model": path.stem,
                        "actual": actual_name,
                        "predicted": predicted_name,
                        "count": matrix[actual_index][predicted_index],
                    }
                )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "actual", "predicted", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_feature_importance(model_name: str, top_n: int = 20) -> Path:
    data = load_prepared_data()
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data.feature_names:
        raise ValueError("Feature names not available. Prepare data without PCA first.")

    model = joblib.load(model_path)
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
        std = np.zeros_like(importance)
        method = "native"
    else:
        result = permutation_importance(
            model,
            data.x_test,
            data.y_test,
            scoring="f1_macro",
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        importance = result.importances_mean
        std = result.importances_std
        method = "permutation"

    if len(importance) != len(data.feature_names):
        raise ValueError(
            f"Feature count mismatch for {model_name}: "
            f"model has {len(importance)} features, prepared data has {len(data.feature_names)} features. "
            "Retrain the model after running scripts/prepare_data.py."
        )

    ranked = sorted(
        zip(data.feature_names, importance, std, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    output_path = METRICS_DIR / f"{model_name}_feature_importance.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["rank", "feature", "importance", "std", "method"])
        writer.writeheader()
        for rank, (feature, value, deviation) in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "feature": feature,
                    "importance": value,
                    "std": deviation,
                    "method": method,
                }
            )
    return output_path


def model_feature_count(model_name: str) -> int | None:
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        return None

    model = joblib.load(model_path)
    if hasattr(model, "feature_importances_"):
        return len(model.feature_importances_)
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    return None


def choose_importance_model(feature_count: int) -> str:
    for model_name in ("random_forest_tuned", "random_forest"):
        count = model_feature_count(model_name)
        if count == feature_count:
            return model_name

    raise ValueError(
        "No compatible Random Forest model found for the prepared data. "
        "Run scripts/train_ml.py or scripts/tune_ml.py after scripts/prepare_data.py."
    )


def main() -> None:
    data = load_prepared_data()
    matrix_path = write_confusion_matrices(data.class_names)
    if not data.feature_names:
        raise ValueError("Feature names not available. Prepare data without PCA first.")

    importance_model = choose_importance_model(len(data.feature_names))
    importance_path = write_feature_importance(importance_model)
    print(f"saved: {matrix_path}")
    print(f"saved: {importance_path}")


if __name__ == "__main__":
    main()
