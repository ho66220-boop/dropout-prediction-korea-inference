from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR, PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: format(value, floatfmt))
        else:
            formatted[column] = formatted[column].astype(str)

    headers = list(formatted.columns)
    rows = formatted.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *row_lines])


def load_summary() -> pd.DataFrame:
    summary_path = METRICS_DIR / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Run python scripts/summarize_metrics.py first.")
    return pd.read_csv(summary_path)


def plot_model_performance(summary: pd.DataFrame) -> Path:
    top = summary.sort_values("macro_f1", ascending=False).head(10).copy()
    top = top.sort_values("macro_f1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["model"], top["macro_f1"], color="#4C78A8")
    ax.set_xlabel("Macro F1")
    ax.set_title("Top Model Performance by Macro F1")
    ax.set_xlim(0, max(0.8, top["macro_f1"].max() + 0.04))
    for index, value in enumerate(top["macro_f1"]):
        ax.text(value + 0.005, index, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()

    output_path = FIGURES_DIR / "model_performance.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_experiment_comparison(summary: pd.DataFrame) -> Path:
    experiment_rows = summary[
        summary["model"].str.contains("_base|_smote|_pca|_pca_smote", regex=True)
        & ~summary["model"].str.contains("_tuned", regex=False)
    ].copy()

    def split_model(value: str) -> pd.Series:
        for suffix in ("_pca_smote", "_smote", "_pca", "_base"):
            if value.endswith(suffix):
                return pd.Series({"algorithm": value[: -len(suffix)], "experiment": suffix[1:]})
        return pd.Series({"algorithm": value, "experiment": "other"})

    experiment_rows = pd.concat([experiment_rows, experiment_rows["model"].apply(split_model)], axis=1)
    pivot = experiment_rows.pivot_table(index="algorithm", columns="experiment", values="macro_f1")
    pivot = pivot[[column for column in ["base", "smote", "pca", "pca_smote"] if column in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("Algorithm")
    ax.set_title("Effect of PCA and SMOTE")
    ax.set_ylim(0.55, 0.75)
    ax.legend(title="Experiment")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()

    output_path = FIGURES_DIR / "experiment_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_feature_importance() -> Path:
    path = METRICS_DIR / "random_forest_tuned_feature_importance.csv"
    if not path.exists():
        raise FileNotFoundError("Run python scripts/analyze_results.py first.")

    importance = pd.read_csv(path).head(12).copy()
    importance["feature"] = importance["feature"].map(clean_feature_name)
    importance = importance.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importance["feature"], importance["importance"], color="#59A14F")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    for index, value in enumerate(importance["importance"]):
        ax.text(value + 0.003, index, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()

    output_path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(best_model: str, class_names: list[str]) -> Path:
    metrics = json.loads((METRICS_DIR / f"{best_model}.json").read_text(encoding="utf-8"))
    matrix = metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {best_model}")

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="#111111")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path = FIGURES_DIR / "best_confusion_matrix.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def build_report(summary: pd.DataFrame, best_model: str, figure_paths: list[Path]) -> Path:
    best = summary.sort_values("macro_f1", ascending=False).iloc[0]
    baseline = summary[summary["model"] == "mlp_sgd"].iloc[0]
    rf_smote = summary[summary["model"] == "random_forest_smote"].iloc[0]
    rf_base = summary[summary["model"] == "random_forest_base"].iloc[0]

    importance = pd.read_csv(METRICS_DIR / "random_forest_tuned_feature_importance.csv").head(10)
    top_features = "\n".join(
        f"{int(row.rank)}. {clean_feature_name(row.feature)}: {row.importance:.4f}"
        for row in importance.itertuples(index=False)
    )

    metrics_table = summary.sort_values("macro_f1", ascending=False).head(8)[
        ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    ]
    metrics_table = dataframe_to_markdown(metrics_table)

    figure_links = "\n".join(f"- [{path.name}](figures/{path.name})" for path in figure_paths)

    report = f"""# Final Project Report Draft

## Project Information

- Type: Solo individual project
- Title: Predicting Student Dropout from Overseas Education Data, and Inferring South Korea's Surge in 10th-Grade Voluntary Withdrawal
- Dataset: Predict Students' Dropout and Academic Success (Portuguese higher education)
- Task: Multi-class classification of `Dropout`, `Enrolled`, and `Graduate`

## 1. Objective

This project predicts student dropout and academic success using higher-education tabular data. The main goal is not only to maximize predictive performance, but also to compare classical machine learning models and a neural network model, then interpret which student-related factors are most associated with the prediction. As a social framing, the interpretation is connected to South Korea's recent surge in 10th-grade voluntary withdrawal (which exceeded 10,000 students for the first time). Since Korean data is hard to obtain directly, insights from this overseas (Portuguese higher-education) dataset are cautiously extrapolated to the Korean context — treated as analogical interpretation, not prediction. A key nuance is that many Korean withdrawals are "strategic" (leaving to focus on the national college entrance exam), whose motivation may differ from the failure-driven dropout this dataset captures.

## 2. Data and Preprocessing

The dataset contains 4,424 student records. The target variable has three classes: `Dropout`, `Enrolled`, and `Graduate`.

The preprocessing pipeline includes duplicate removal, stratified train/test split, standardization for numerical features, one-hot encoding for categorical features, and optional PCA or SMOTE experiments. A fixed random seed was used for reproducibility.

## 3. Models

The following models were tested:

- Logistic Regression
- Support Vector Machine with RBF kernel
- Random Forest
- Multi-Layer Perceptron trained with SGD and backpropagation

GridSearchCV was applied to the classical ML models. Separate comparison experiments were also conducted for PCA and SMOTE.

## 4. Main Results

Top model results by macro F1:

{metrics_table}

The best model by macro F1 was `{best_model}` with macro F1 `{best.macro_f1:.4f}` and accuracy `{best.accuracy:.4f}`.

The MLP baseline reached macro F1 `{baseline.macro_f1:.4f}` and accuracy `{baseline.accuracy:.4f}`. In this dataset, the tree-based model performed better than the neural network. This is a reasonable result because the dataset is small and tabular, where ensemble tree methods often capture non-linear feature interactions effectively without requiring large-scale representation learning.

## 5. PCA and SMOTE Findings

SMOTE slightly improved Random Forest macro F1 from `{rf_base.macro_f1:.4f}` to `{rf_smote.macro_f1:.4f}`. However, PCA generally reduced performance across models. This suggests that the original feature dimensions contain interpretable and useful predictive information, and compressing them may remove signals that are important for distinguishing `Enrolled` from the other classes.

## 6. Feature Importance

The most important Random Forest features were:

{top_features}

The strongest predictors are mostly semester performance variables such as approved curricular units and grades. Tuition payment status and age at enrollment also appear among the top features. This indicates that academic progress after enrollment is more predictive than many demographic or application-stage variables.

## 7. Error Analysis

The confusion matrix shows that the hardest class to classify is generally `Enrolled`. This is expected because `Enrolled` is an intermediate state: some enrolled students may later graduate, while others may eventually drop out. Therefore, the class boundary is less stable than the boundary between clear `Dropout` and clear `Graduate` cases.

## 8. Limitations

- The dataset has only 4,424 rows, which limits the advantage of deep learning.
- The task is based on historical tabular data and may not capture personal, institutional, or temporal factors outside the dataset.
- Some high-importance features are semester outcome variables, which may be unavailable at the earliest admission stage.
- The model should be used as a decision-support tool, not as a final judgment about individual students.

## 9. Conclusion

Classical machine learning, especially Random Forest with SMOTE, produced the strongest performance in this project. The MLP model was competitive but did not outperform the best ML model. From an educational insight perspective, academic progress indicators and tuition-payment status were the most influential predictors. For practical use, the model is most valuable as an early-warning system that helps institutions identify students who may need timely support.

## Figures

{figure_links}
"""

    output_path = REPORTS_DIR / "final_report.md"
    output_path.write_text(report, encoding="utf-8")
    return output_path


def build_html_report(summary: pd.DataFrame, best_model: str) -> Path:
    best = summary.sort_values("macro_f1", ascending=False).iloc[0]
    top_table = summary.sort_values("macro_f1", ascending=False).head(8)[
        ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    ].copy()
    for column in top_table.columns[1:]:
        top_table[column] = top_table[column].map(lambda value: f"{value:.4f}")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Student Success and Dropout Prediction Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.55; margin: 40px auto; max-width: 1040px; color: #202124; }}
    h1, h2 {{ color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    figure {{ margin: 28px 0; }}
    img {{ max-width: 100%; border: 1px solid #d0d7de; }}
    .metric {{ font-weight: 700; color: #1d4ed8; }}
    @media print {{ body {{ margin: 24px; }} figure {{ break-inside: avoid; }} }}
  </style>
</head>
<body>
  <h1>A Comparative Study of ML and DL for Predicting Student Success and Dropout</h1>
  <p><strong>Solo Individual Project</strong> | Final Project Report</p>

  <h2>Objective</h2>
  <p>This project predicts whether students are likely to drop out, remain enrolled, or graduate. The study compares classical machine learning models with a neural network model and interprets the key features behind the predictions.</p>

  <h2>Dataset and Method</h2>
  <p>The dataset contains 4,424 higher-education student records. Preprocessing included duplicate removal, stratified train/test split, standardization, one-hot encoding, and optional PCA/SMOTE experiments.</p>

  <h2>Main Result</h2>
  <p>The best model was <span class="metric">{best_model}</span>, with macro F1 <span class="metric">{best.macro_f1:.4f}</span> and accuracy <span class="metric">{best.accuracy:.4f}</span>.</p>
  {top_table.to_html(index=False, escape=False)}

  <figure>
    <img src="figures/model_performance.png" alt="Model performance">
    <figcaption>Top model performance by macro F1.</figcaption>
  </figure>
  <figure>
    <img src="figures/experiment_comparison.png" alt="PCA and SMOTE comparison">
    <figcaption>Effect of PCA and SMOTE on model performance.</figcaption>
  </figure>
  <figure>
    <img src="figures/feature_importance.png" alt="Feature importance">
    <figcaption>Most important predictors from the Random Forest model.</figcaption>
  </figure>
  <figure>
    <img src="figures/best_confusion_matrix.png" alt="Confusion matrix">
    <figcaption>Confusion matrix for the best-performing model.</figcaption>
  </figure>

  <h2>Interpretation</h2>
  <p>Random Forest performed best overall, which fits the characteristics of a small tabular dataset. The MLP model was competitive but did not outperform the strongest classical ML model. Semester-level academic progress variables, especially approved curricular units and grades, were the most influential predictors.</p>

  <h2>Conclusion</h2>
  <p>The model is best understood as a decision-support early-warning system. It can help identify students who may need support, but should not be used as a final decision-making authority for individual students.</p>
</body>
</html>
"""
    output_path = REPORTS_DIR / "final_report.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_presentation_outline(summary: pd.DataFrame, best_model: str) -> Path:
    best = summary.sort_values("macro_f1", ascending=False).iloc[0]
    outline = f"""# Presentation Outline

## Slide 1. Title
A Comparative Study of ML and DL for Predicting Student Success and Dropout

## Slide 2. Problem and Goal
- Goal: Predict Dropout, Enrolled, Graduate
- Practical value: early warning and student support

## Slide 3. Dataset
- 4,424 rows
- 36 input features after preprocessing
- Target classes: Dropout, Enrolled, Graduate

## Slide 4. Preprocessing
- Duplicate removal
- Stratified train/test split
- Standardization and one-hot encoding
- PCA and SMOTE comparison experiments

## Slide 5. Models
- Logistic Regression
- SVM RBF
- Random Forest
- MLP with SGD/backpropagation

## Slide 6. Performance
- Best model: {best_model}
- Macro F1: {best.macro_f1:.4f}
- Accuracy: {best.accuracy:.4f}
- Show `model_performance.png`

## Slide 7. PCA and SMOTE
- SMOTE slightly improved Random Forest
- PCA generally reduced performance
- Show `experiment_comparison.png`

## Slide 8. Feature Importance
- Semester approved units and grades were strongest predictors
- Tuition fees up to date was also important
- Show `feature_importance.png`

## Slide 9. Error Analysis
- Enrolled class is hardest to classify
- It is an intermediate and unstable academic state
- Show `best_confusion_matrix.png`

## Slide 10. Conclusion
- Random Forest performed best for this small tabular dataset
- MLP was competitive but not superior
- Best use case: decision-support early-warning system
"""
    output_path = REPORTS_DIR / "presentation_outline.md"
    output_path.write_text(outline, encoding="utf-8")
    return output_path


def build_slide_deck(summary: pd.DataFrame, best_model: str) -> Path:
    best = summary.sort_values("macro_f1", ascending=False).iloc[0]
    slides = [
        ("Predicting Dropout from Overseas Data to Read Korea's 10th-Grade Withdrawal Surge", ["Solo Individual Project", "Final Project Presentation"]),
        ("Problem and Goal", ["Predict Dropout, Enrolled, and Graduate", "Read Korea's 10th-grade withdrawal surge by analogy", "Support early intervention for students at risk"]),
        ("Dataset", ["4,424 student records", "Tabular higher-education data", "Three target classes"]),
        ("Preprocessing", ["Stratified train/test split", "Standardization and one-hot encoding", "PCA and SMOTE experiments"]),
        ("Models", ["Logistic Regression", "SVM with RBF kernel", "Random Forest", "MLP with SGD/backpropagation"]),
        ("Performance", [f"Best model: {best_model}", f"Macro F1: {best.macro_f1:.4f}", f"Accuracy: {best.accuracy:.4f}", '<img src="figures/model_performance.png" alt="performance">']),
        ("PCA and SMOTE", ["SMOTE slightly improved Random Forest", "PCA generally reduced performance", '<img src="figures/experiment_comparison.png" alt="experiments">']),
        ("Feature Importance", ["Semester approved units and grades were strongest", "Tuition payment status was also important", '<img src="figures/feature_importance.png" alt="importance">']),
        ("Error Analysis", ["The Enrolled class was hardest to classify", "It is an intermediate academic state", '<img src="figures/best_confusion_matrix.png" alt="confusion matrix">']),
        ("Korea Analogy", ["Data signals failure-driven dropout (low grades, unpaid tuition)", "Korean 10th-grade withdrawal is often strategic (exam-focused)", "Motivations may be opposite — applied as hypothesis, not policy"]),
        ("Conclusion", ["Random Forest performed best for this dataset", "MLP was competitive but not superior", "Best practical use: decision-support early-warning system", "Korea insights are analogical only, not direct prediction"]),
    ]
    sections = []
    for title, bullets in slides:
        rendered = "\n".join(
            item if item.startswith("<img") else f"<li>{item}</li>"
            for item in bullets
        )
        if "<li>" in rendered:
            rendered = f"<ul>{rendered}</ul>"
        sections.append(f"<section><h1>{title}</h1>{rendered}</section>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Student Success Prediction Slides</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #111827; color: #f9fafb; }}
    section {{ min-height: 100vh; box-sizing: border-box; padding: 64px 84px; display: flex; flex-direction: column; justify-content: center; border-bottom: 1px solid #374151; }}
    h1 {{ font-size: 44px; margin: 0 0 28px; max-width: 980px; }}
    ul {{ font-size: 28px; line-height: 1.5; }}
    img {{ max-width: 860px; max-height: 520px; background: white; padding: 10px; border-radius: 4px; }}
    @media print {{ section {{ min-height: 720px; page-break-after: always; }} body {{ background: white; color: black; }} }}
  </style>
</head>
<body>
  {"".join(sections)}
</body>
</html>
"""
    output_path = REPORTS_DIR / "presentation_slides.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    best_model = summary.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    class_names = ["Dropout", "Enrolled", "Graduate"]

    figure_paths = [
        plot_model_performance(summary),
        plot_experiment_comparison(summary),
        plot_feature_importance(),
        plot_confusion_matrix(best_model, class_names),
    ]
    report_path = build_report(summary, best_model, figure_paths)
    html_report_path = build_html_report(summary, best_model)
    outline_path = build_presentation_outline(summary, best_model)
    slides_path = build_slide_deck(summary, best_model)

    print(f"best model: {best_model}")
    for path in figure_paths:
        print(f"saved: {path}")
    print(f"saved: {report_path}")
    print(f"saved: {html_report_path}")
    print(f"saved: {outline_path}")
    print(f"saved: {slides_path}")


if __name__ == "__main__":
    main()
