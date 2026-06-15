"""추가 시각화 자료 생성 스크립트.

기존 figure(클래스 분포, 모델 성능, feature importance, confusion matrix 등)에
더해, 데이터 탐색 깊이와 한국 고1 자퇴 유추 해석을 뒷받침하는 4종의 figure를
`reports/figures/` 아래에 생성한다.

- correlation_heatmap.png       : 핵심 수치형 변수 간 상관관계
- pca_projection.png            : 2D PCA projection (클래스별 색상)
- per_class_metrics.png         : 최고 모델의 클래스별 precision/recall/F1
- dropout_signal_by_class.png   : 클래스별 핵심 변수 평균 (실패형 이탈 신호 시각화)

전제: scripts/prepare_data.py, scripts/train_ml.py 등으로 데이터/메트릭이 이미
생성되어 있어야 한다.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR, PROJECT_ROOT
from src.data import find_csv, infer_target_column, load_prepared_data

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def configure_korean_font() -> bool:
    """한글 캡션이 깨지지 않도록 사용 가능한 한글 폰트를 등록한다.

    Windows의 맑은 고딕(malgun.ttf) 등을 우선 탐색한다. 찾지 못하면 False를
    반환하고, 호출부에서 한글 캡션을 영어로 대체한다.
    """
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunsl.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


KOREAN_FONT_AVAILABLE = configure_korean_font()

# 클래스별 색상 (직관적으로 Dropout=빨강, Graduate=초록)
CLASS_COLORS = {"Dropout": "#E45756", "Enrolled": "#F2A93B", "Graduate": "#54A24B"}

KEY_NUMERIC_FEATURES = [
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Admission grade",
    "Previous qualification (grade)",
    "Age at enrollment",
    "Tuition fees up to date",
    "Scholarship holder",
]


def short_label(name: str) -> str:
    """긴 컬럼명을 그래프용 짧은 라벨로 축약."""
    replacements = {
        "Curricular units 1st sem (approved)": "1st sem approved",
        "Curricular units 1st sem (grade)": "1st sem grade",
        "Curricular units 2nd sem (approved)": "2nd sem approved",
        "Curricular units 2nd sem (grade)": "2nd sem grade",
        "Admission grade": "Admission grade",
        "Previous qualification (grade)": "Prev qual grade",
        "Age at enrollment": "Age at enroll",
        "Tuition fees up to date": "Tuition up-to-date",
        "Scholarship holder": "Scholarship",
    }
    return replacements.get(name, name)


def class_color(label: str) -> str:
    return CLASS_COLORS.get(label, "#4C78A8")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    available = [c for c in KEY_NUMERIC_FEATURES if c in df.columns]
    corr = df[available].corr()
    labels = [short_label(c) for c in available]

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Correlation of Key Numeric Features")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="#111111" if abs(value) < 0.6 else "#ffffff",
                fontsize=7,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path = FIGURES_DIR / "correlation_heatmap.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_pca_projection() -> Path:
    data = load_prepared_data()
    x = data.x_train
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=float)

    coords = PCA(n_components=2, random_state=42).fit_transform(x)

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.asarray(data.y_train)
    for class_id, label in enumerate(data.class_names):
        mask = y == class_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.5,
            label=label,
            color=class_color(label),
            edgecolors="none",
        )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("2D PCA Projection of Students by Class")
    ax.legend(title="Class", markerscale=2)
    fig.tight_layout()

    output_path = FIGURES_DIR / "pca_projection.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def choose_best_model() -> str:
    summary = pd.read_csv(METRICS_DIR / "summary.csv")
    for model in summary.sort_values("macro_f1", ascending=False)["model"]:
        path = METRICS_DIR / f"{model}.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8")).get("classification_report")
            if isinstance(report, dict):
                return model
    raise FileNotFoundError("classification_report를 가진 모델 JSON을 찾을 수 없습니다.")


def plot_per_class_metrics() -> Path:
    best_model = choose_best_model()
    report = json.loads((METRICS_DIR / f"{best_model}.json").read_text(encoding="utf-8"))[
        "classification_report"
    ]
    classes = [c for c in ("Dropout", "Enrolled", "Graduate") if c in report]
    metrics = ["precision", "recall", "f1-score"]
    metric_colors = {"precision": "#4C78A8", "recall": "#F58518", "f1-score": "#54A24B"}

    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 6))
    for offset, metric in enumerate(metrics):
        values = [report[c][metric] for c in classes]
        bars = ax.bar(x + (offset - 1) * width, values, width, label=metric, color=metric_colors[metric])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Precision / Recall / F1: {best_model}")
    ax.legend()
    fig.tight_layout()

    output_path = FIGURES_DIR / "per_class_metrics.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_dropout_signal_by_class(df: pd.DataFrame, target: str) -> Path:
    """클래스별 핵심 변수 평균을 0~1로 정규화하여 비교.

    데이터의 Dropout은 학업·재정 지표가 전반적으로 낮은 '실패형 이탈'임을
    시각적으로 보여준다. 이는 성적이 양호한 한국형 전략적 자퇴와 대비된다.
    """
    available = [c for c in KEY_NUMERIC_FEATURES if c in df.columns]
    classes = [c for c in ("Dropout", "Enrolled", "Graduate") if c in df[target].unique()]

    means = df.groupby(target)[available].mean()
    # 변수별 min-max 정규화 (클래스 간 상대 위치 비교용)
    normalized = (means - means.min()) / (means.max() - means.min()).replace(0, np.nan)
    normalized = normalized.fillna(0.5)

    labels = [short_label(c) for c in available]
    x = np.arange(len(available))
    width = 0.8 / max(len(classes), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, label in enumerate(classes):
        values = normalized.loc[label, available].values
        ax.bar(x + (offset - (len(classes) - 1) / 2) * width, values, width, label=label, color=class_color(label))
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Class-mean (min-max normalized)")
    ax.set_title("Failure-Driven Dropout Signal: Key Features by Class")
    ax.legend(title="Class")
    if KOREAN_FONT_AVAILABLE:
        caption = (
            "Dropout 학생은 학업·재정 지표가 전반적으로 낮은 '실패형 이탈' 패턴을 보인다.\n"
            "반면 한국 고1의 전략적 자퇴는 성적이 양호한 자발적 이탈이 많아 동기 구조가 다를 수 있다 (유추 해석)."
        )
    else:
        caption = (
            "Dropout students score low on academic/financial indicators — a failure-driven exit.\n"
            "Korea's 10th-grade withdrawal is often a strategic, exam-focused choice (analogy, not prediction)."
        )
    ax.text(0.0, -0.32, caption, transform=ax.transAxes, fontsize=9, color="#444444")
    fig.subplots_adjust(bottom=0.32)

    output_path = FIGURES_DIR / "dropout_signal_by_class.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(find_csv(), sep=None, engine="python")
    target = infer_target_column(df)

    outputs = [
        plot_correlation_heatmap(df),
        plot_pca_projection(),
        plot_per_class_metrics(),
        plot_dropout_signal_by_class(df, target),
    ]
    for path in outputs:
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
