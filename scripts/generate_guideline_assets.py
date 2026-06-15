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
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR, PROJECT_ROOT, RANDOM_STATE
from src.data import find_csv, infer_target_column, load_prepared_data


REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

BASELINE_CODES = [
    {
        "name": "ML Algorithms Usage and Prediction",
        "url": "https://www.kaggle.com/code/sunayanagawde/ml-algorithms-usage-and-prediction",
        "role": "전통적 ML workflow와 비교를 위한 public-code baseline.",
    },
    {
        "name": "Dropout Graduate Analysis",
        "url": "https://www.kaggle.com/code/satyaprakashshukl/droput-graduate-analysis",
        "role": "EDA 중심의 dropout/graduate 분석을 참고하기 위한 public-code baseline.",
    },
    {
        "name": "Student Dropout Analysis for School Education",
        "url": "https://www.kaggle.com/code/jeevabharathis/student-dropout-analysis-for-school-education",
        "role": "교육 분야 dropout 분석을 참고하기 위한 public-code baseline.",
    },
]


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


def load_raw_dataset() -> tuple[pd.DataFrame, str]:
    csv_path = find_csv()
    df = pd.read_csv(csv_path, sep=None, engine="python")
    target = infer_target_column(df)
    return df, target


def plot_class_distribution(df: pd.DataFrame, target: str) -> Path:
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index.astype(str), counts.values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Target Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for index, value in enumerate(counts.values):
        ax.text(index, value + max(counts.values) * 0.015, str(value), ha="center")
    fig.tight_layout()
    output_path = FIGURES_DIR / "class_distribution.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_key_feature_boxplots(df: pd.DataFrame, target: str) -> Path:
    features = [
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
    ]
    available = [feature for feature in features if feature in df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for ax, feature in zip(axes, available, strict=False):
        groups = [df.loc[df[target] == label, feature].dropna() for label in sorted(df[target].unique())]
        ax.boxplot(groups, tick_labels=sorted(df[target].unique()), showfliers=False)
        ax.set_title(feature)
        ax.tick_params(axis="x", rotation=15)
    for ax in axes[len(available) :]:
        ax.axis("off")
    fig.tight_layout()
    output_path = FIGURES_DIR / "key_feature_distributions.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run_unsupervised_analysis() -> dict[str, float | int]:
    data = load_prepared_data()
    model = KMeans(n_clusters=len(data.class_names), random_state=RANDOM_STATE, n_init=20)
    cluster_labels = model.fit_predict(data.x_train)
    metrics = {
        "n_clusters": len(data.class_names),
        "normalized_mutual_info": normalized_mutual_info_score(data.y_train, cluster_labels),
        "adjusted_rand_index": adjusted_rand_score(data.y_train, cluster_labels),
        "inertia": float(model.inertia_),
    }
    output_path = METRICS_DIR / "unsupervised_kmeans.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_activity_appendix() -> Path:
    content = """# 활동 Appendix

## 프로포절의 기존 활동 계획

| 기간 | 계획 |
|---|---|
| 5/11-5/17 | 결측치 처리, 스케일링, PCA 등 ML/DL 공통 전처리 파이프라인 구축 |
| 5/18-5/31 | Logistic Regression, SVM, Random Forest, MLP 모델 개발 |
| 6/1-6/09 | GridSearchCV, SGD, Backpropagation 기반 모델 검증 및 튜닝 |
| 6/10-6/30 | 통합 성능 비교, 오류 분석, 변수 중요도 해석, 한국 자퇴 유추 해석, 최종 보고서 및 발표 자료 작성 |

## 실제 수행한 수정 활동 계획

| 단계 | 수행 내용 | 산출물 |
|---|---|---|
| 데이터 준비 | 학생 중도 포기 데이터셋을 준비하고 재현 가능한 폴더 구조를 생성 | `data/raw`, `data/processed` |
| 전처리 | stratified train/test split, 표준화, one-hot encoding, PCA/SMOTE 옵션 구현 | `scripts/prepare_data.py` |
| ML baseline | Logistic Regression, SVM, Random Forest 학습 | `scripts/train_ml.py` |
| DL baseline | SGD/backpropagation 기반 MLP 학습 | `scripts/train_mlp.py` |
| 튜닝 | GridSearchCV로 전통적 ML 모델 튜닝 | `scripts/tune_ml.py` |
| Ablation 실험 | base, SMOTE, PCA, PCA+SMOTE 설정 비교 | `scripts/run_experiments.py` |
| 분석 | confusion matrix, feature importance, EDA 그래프, KMeans 비지도 분석 생성 | `reports/metrics`, `reports/figures` |
| 최종 산출물 | 가이드라인 맞춤 보고서, appendix, 발표 자료 생성 | `reports/final_report_guideline_aligned.md`, `reports/presentation_slides.html` |

## 수행 범위 (1인 프로젝트)

| 영역 | 수행 내용 |
|---|---|
| 데이터·전처리 | 전처리 전략 수립, 표준화/PCA 실험, SMOTE 실험 설계, EDA 해석 |
| 모델링 | Logistic Regression·SVM·Random Forest 학습 및 튜닝, MLP 구조 설계와 SGD/backpropagation 학습 구성 |
| 분석·해석 | ML/DL 오류 비교, confusion matrix 분석, feature importance 해석, 한국 고1 자퇴에 대한 유추적 해석, 최종 검증 |

## 재현성 체크리스트

- 고정 random seed: 42
- 비교 가능한 실험에는 동일한 train/test split 사용
- 평가 지표: accuracy, macro precision, macro recall, macro F1, weighted F1
- 생성 파일은 `reports/metrics`와 `reports/figures`에 저장
"""
    output_path = REPORTS_DIR / "activity_appendix.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def build_guideline_report(unsupervised_metrics: dict[str, float | int], figure_paths: list[Path]) -> Path:
    summary = pd.read_csv(METRICS_DIR / "summary.csv")
    best = summary.sort_values("macro_f1", ascending=False).iloc[0]
    mlp = summary[summary["model"] == "mlp_sgd"].iloc[0]
    importance = pd.read_csv(METRICS_DIR / "random_forest_tuned_feature_importance.csv").head(10)

    metrics_table = summary.sort_values("macro_f1", ascending=False).head(10)[
        ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    ]
    metrics_table = dataframe_to_markdown(metrics_table)
    feature_lines = "\n".join(
        f"{int(row.rank)}. {clean_feature_name(row.feature)}: {row.importance:.4f}"
        for row in importance.itertuples(index=False)
    )
    baseline_lines = "\n".join(
        f"- {item['name']}: {item['url']}  \n  본 프로젝트에서의 역할: {item['role']}"
        for item in BASELINE_CODES
    )
    figure_lines = "\n".join(f"- [{path.name}](figures/{path.name})" for path in figure_paths)

    report = f"""# 과제 가이드라인 맞춤 최종 보고서

## 프로젝트 개요

- 과목: NOVA50101 Introduction to Artificial Intelligence for Industrial AI
- 프로젝트 유형: 1인 개인 프로젝트 (Application project)
- 프로젝트 제목: 해외 교육 데이터 기반 중도 이탈 예측과 한국 고1 자퇴 급증에 대한 유추적 해석
- 데이터셋: Predict Students' Dropout and Academic Success (포르투갈 고등교육, Kaggle/UCI)

본 프로젝트는 학생이 중도 포기할지, 계속 재학 중일지, 졸업할지를 예측하는 교육 AI 문제를 다룬다. 과제 가이드라인에 맞춰 전통적 머신러닝, 딥러닝, 전처리 실험, 오류 분석, 결과 해석을 함께 수행했다. 더불어 최근 연 1만 명을 넘어선 한국 고1 자퇴 급증 현상과 연결하되, 한국 데이터를 직접 구하기 어려워 해외(포르투갈 고등교육) 데이터의 이탈 구조를 학습한 뒤 한국 상황에 조심스럽게 유추한다. 한국 관련 서술은 모두 "예측"이 아닌 "유추적 해석"으로 제한하며, 한국형 전략적 자퇴(수능 집중을 위한 자발적 이탈)는 데이터가 학습한 실패형 이탈과 동기가 다를 수 있다는 점을 명시한다.

## Public Code Baseline 및 Originality

프로포절에서는 다음 Kaggle public code를 외부 baseline/reference로 제시했다.

{baseline_lines}

Kaggle notebook의 출력값과 split 방식은 정적 페이지에서 항상 동일하게 재현하기 어렵기 때문에, 본 보고서는 해당 public code보다 절대적으로 성능이 우수하다고 주장하지 않는다. 대신 이 public code들을 참고 baseline으로 삼고, 모든 모델을 동일한 전처리와 train/test protocol 안에서 비교하는 재현 가능한 파이프라인을 구축했다.

본 프로젝트의 originality는 다음 지점에 있다.

- 하나의 공통 전처리 파이프라인 아래에서 ML/DL 모델을 통제된 방식으로 비교
- PCA와 SMOTE ablation 실험 수행
- 전통적 ML 모델에 대한 GridSearchCV 튜닝
- SGD/backpropagation 기반 MLP 학습
- feature importance와 confusion matrix 기반 오류 분석
- KMeans 비지도 분석을 통해 자연 cluster와 target label의 정렬 정도 확인

## 데이터셋 및 EDA

데이터셋은 4,424개의 학생 record와 target을 포함한 36개의 raw column으로 구성되어 있다. target class는 `Dropout`, `Enrolled`, `Graduate` 세 가지이다.

EDA 결과 target class가 불균형하며, 학기별 학업 성과 변수들이 class별로 뚜렷한 차이를 보였다. 따라서 stratified split, class balancing 실험, macro F1 중심 평가가 필요하다고 판단했다.

## 방법론

전처리:

- 중복 record 제거
- stratified train/test split 적용
- 수치형 feature 표준화
- 범주형 feature one-hot encoding
- PCA와 SMOTE를 전처리 변형 실험으로 적용

모델:

- Logistic Regression
- SVM with RBF kernel
- Random Forest
- MLP trained with SGD/backpropagation
- KMeans for unsupervised analysis

평가 지표:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Confusion matrix

class distribution이 불균형하고 `Enrolled` class가 다른 class보다 분류하기 어렵기 때문에, accuracy뿐 아니라 macro F1을 핵심 지표로 사용했다.

## 실험 결과

macro F1 기준 상위 결과는 다음과 같다.

{metrics_table}

가장 좋은 모델은 `{best.model}`이며, macro F1은 `{best.macro_f1:.4f}`, accuracy는 `{best.accuracy:.4f}`이다. MLP baseline은 macro F1 `{mlp.macro_f1:.4f}`, accuracy `{mlp.accuracy:.4f}`를 기록했다.

가장 강한 성능을 보인 모델은 SMOTE를 적용한 tree-based ensemble이었다. 이 결과는 데이터셋이 작고 tabular 형태라는 점에서 타당하다. Random Forest는 대규모 representation learning이 필요한 neural network보다 작은 tabular dataset에서 비선형 feature interaction을 효율적으로 포착할 수 있다.

## Ablation 분석

SMOTE는 Random Forest의 macro F1을 소폭 개선했지만, PCA는 전반적으로 성능을 낮췄다. 이는 차원 축소 과정에서 tabular feature가 가진 유용한 신호가 일부 손실되었을 가능성을 시사한다. 반면 SMOTE는 minority class 패턴을 조금 더 반영하도록 돕는 효과가 있었다.

## 비지도 학습 분석

target class 수와 동일하게 KMeans를 3개 cluster로 실행했다. 결과는 다음과 같다.

- Normalized Mutual Information: `{unsupervised_metrics['normalized_mutual_info']:.4f}`
- Adjusted Rand Index: `{unsupervised_metrics['adjusted_rand_index']:.4f}`
- Inertia: `{unsupervised_metrics['inertia']:.2f}`

비지도 cluster와 실제 label의 정렬 정도가 낮다는 점은, 세 target class가 단순한 거리 기반 clustering만으로 명확히 분리되지 않음을 보여준다. 따라서 supervised learning model이 필요하다는 해석이 가능하다.

## 변수 중요도 및 인사이트

Random Forest 기준 상위 feature는 다음과 같다.

{feature_lines}

가장 중요한 예측 변수는 학기별 이수/승인 과목 수와 성적이었다. 등록금 납부 여부도 중요한 변수로 나타났다. 이는 중도 포기와 졸업 여부가 입학 당시 정보뿐 아니라, 입학 후 학업 진행 상황과 강하게 연결되어 있음을 의미한다.

### 한국 고1 자퇴에 대한 유추적 해석

본 데이터가 가리키는 이탈 신호는 학업·재정 누적 실패(과목 이수 부진, 낮은 성적, 등록금 미납)에 가깝다. 반면 한국 고1의 전략적 자퇴는 오히려 성적이 양호한 학생이 더 나은 입시 전략(검정고시·수능 집중)을 위해 선택하는 경우가 많아, 동기 구조가 반대일 수 있다. 따라서 해외 데이터의 신호를 한국에 곧바로 적용하면 정반대 해석을 낳을 위험이 있다. 유추가 유효한 지점은 "재정·행정 상태와 초기 학업 성과가 지속 여부와 연결된다"는 일반 구조이며, 이는 경제적 사유나 학업 부적응으로 인한 비전략적 자퇴 이해에 참고가 된다. 이 간극을 명시적으로 드러내는 것이 본 프로젝트의 핵심 기여이며, 한국 관련 모든 해석은 정책 제언이 아닌 가설 수준으로 한정한다.

## 오류 분석

confusion matrix를 보면 `Enrolled` class가 가장 분류하기 어려웠다. 이는 `Enrolled`가 중간 상태이기 때문이다. 현재 재학 중인 학생은 이후 졸업할 수도 있고 중도 포기할 수도 있으므로, `Dropout` 및 `Graduate`와 decision boundary가 겹칠 수 있다.

## 한계점

- 데이터셋 규모가 딥러닝에 비해 작아 MLP가 representation learning의 장점을 충분히 활용하기 어렵다.
- 일부 중요 feature는 학기별 성과 변수이므로, 입학 직후의 매우 이른 시점에는 사용할 수 없을 수 있다.
- Kaggle public-code baseline은 reference로 사용했지만, 정적 notebook 페이지에서 정확한 metric을 동일하게 재현하지는 못했다.
- 모델은 과거 tabular data에 기반하므로, 제도 변화나 개인적 상황을 모두 반영하지 못한다.
- 데이터는 포르투갈 고등교육 맥락이므로, 한국 고1 자퇴에 대한 해석은 유추 수준이며 직접적인 예측·정책 근거로 사용할 수 없다. 특히 한국형 전략적 자퇴는 데이터가 학습한 실패형 이탈과 동기가 다를 수 있다.
- 예측 결과는 학생 지원을 위한 decision-support 도구로 사용해야 하며, 학생에 대한 최종 판단이나 불이익 부여에 사용되어서는 안 된다.

## 향후 연구

- Top public Kaggle notebook을 동일 split과 metric으로 local에서 재실행하여 직접 비교한다.
- 과목 범위가 허용된다면 XGBoost, LightGBM, CatBoost 등 gradient boosting model을 추가한다.
- 입학 시점 feature 또는 1학기 feature만 사용하는 early-warning model을 별도로 구축한다.
- dropout recall을 높이기 위해 calibration 및 threshold tuning을 수행한다.
- gender, scholarship status, international status 등에 대한 fairness analysis를 수행한다.
- 여러 학년도 데이터가 확보되면 temporal validation을 수행한다.

## 가이드라인 충족 여부

- Technical soundness: 공통 전처리 파이프라인, ML/DL 비교, tuning, ablation, EDA, 오류 분석 포함
- Limitation and future work: 한계점과 구체적인 향후 실험 방향 제시
- Activities: 기존/수정 활동 계획 및 수행 내역을 appendix로 제시

## 그림

{figure_lines}

## Appendix

[activity_appendix.md](activity_appendix.md)를 참고한다.
"""
    output_path = REPORTS_DIR / "final_report_guideline_aligned.md"
    output_path.write_text(report, encoding="utf-8")
    return output_path


def build_guideline_html(markdown_path: Path) -> Path:
    text = markdown_path.read_text(encoding="utf-8")
    html_body = text
    replacements = [
        ("# ", "<h1>", "</h1>"),
        ("## ", "<h2>", "</h2>"),
    ]
    lines = []
    in_list = False
    for raw_line in html_body.splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                lines.append("</ul>")
                in_list = False
            continue
        if line.startswith("## "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("- "):
            if not in_list:
                lines.append("<ul>")
                in_list = True
            lines.append(f"<li>{line[2:]}</li>")
        elif line.startswith("|"):
            lines.append(f"<pre>{line}</pre>")
        else:
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<p>{line}</p>")
    if in_list:
        lines.append("</ul>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>과제 가이드라인 맞춤 최종 보고서</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1060px; margin: 40px auto; line-height: 1.55; color: #202124; }}
    h1, h2 {{ color: #1f2937; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 4px 8px; margin: 2px 0; }}
    li {{ margin: 4px 0; }}
    @media print {{ body {{ margin: 24px; }} }}
  </style>
</head>
<body>
{chr(10).join(lines)}
</body>
</html>
"""
    output_path = REPORTS_DIR / "final_report_guideline_aligned.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df, target = load_raw_dataset()
    figure_paths = [
        plot_class_distribution(df, target),
        plot_key_feature_boxplots(df, target),
        FIGURES_DIR / "model_performance.png",
        FIGURES_DIR / "experiment_comparison.png",
        FIGURES_DIR / "feature_importance.png",
        FIGURES_DIR / "best_confusion_matrix.png",
    ]
    unsupervised_metrics = run_unsupervised_analysis()
    appendix_path = build_activity_appendix()
    report_path = build_guideline_report(unsupervised_metrics, figure_paths)
    html_path = build_guideline_html(report_path)

    print(f"saved: {appendix_path}")
    print(f"saved: {report_path}")
    print(f"saved: {html_path}")
    for path in figure_paths:
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
