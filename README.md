# 학생 학업 성취도 및 중도 포기 예측 프로젝트

Team 11의 NOVA50101 Introduction to Artificial Intelligence for Industrial AI 최종 프로젝트입니다. Kaggle/UCI의 `Predict students' dropout and academic success` 데이터셋을 사용하여 전통적 머신러닝 모델과 MLP 딥러닝 모델을 비교하고, 학생 상태 예측에 중요한 요인을 해석합니다.

## Project Overview

이 프로젝트는 학생 데이터를 바탕으로 `Dropout`, `Enrolled`, `Graduate` 세 클래스를 분류하는 tabular classification 문제를 다룹니다. 단순히 성능이 높은 모델을 찾는 데 그치지 않고, 어떤 변수들이 학생의 학업 지속 가능성과 관련되어 있는지 feature importance를 통해 함께 분석합니다.

핵심 질문은 다음과 같습니다.

- 전통적 머신러닝 모델과 MLP 중 어떤 모델이 이 데이터 구조에 더 적합한가?
- 모델 성능뿐 아니라, 어떤 요인이 예측에 중요하게 작용하는가?
- 교육 현장에서 학생 지원 또는 조기경보 관점으로 활용할 수 있는 해석 가능한 단서를 얻을 수 있는가?

## Dataset

- Dataset name: `Predict students' dropout and academic success`
- Kaggle URL: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
- UCI mirror: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Dataset size: 4,424 rows / 36 input columns + target
- Target classes: `Dropout`, `Enrolled`, `Graduate`

원본 CSV는 용량과 재배포 조건을 고려하여 Git에는 포함하지 않습니다. 실행하려면 Kaggle 또는 UCI에서 받은 CSV 파일을 `data/raw/` 아래에 넣어 주세요.

```text
data/raw/student_dropout_success/data.csv
```

## Methodology

전처리는 `scripts/prepare_data.py`에서 수행합니다.

- CSV 자동 탐색 또는 `--input`으로 지정한 CSV 로드
- BOM, 앞뒤 공백, 탭 등 컬럼명 정리
- target 컬럼 자동 추론
- 중복 행 제거
- `stratify=y`를 적용한 train/test split
- 숫자형 변수 표준화
- 명목형 코드 변수 one-hot encoding
- 선택 옵션으로 PCA 및 SMOTE 적용
- 전처리된 데이터와 preprocessor를 `data/processed/`에 저장

비교한 모델은 Logistic Regression, SVM with RBF kernel, Random Forest, MLP trained with SGD/backpropagation입니다. KMeans는 비지도 분석 참고용으로 사용했습니다.

## Key Results

현재 결과 파일 기준 가장 좋은 모델은 `random_forest`입니다.

```text
macro_f1 = 0.7032
accuracy = 0.7627
weighted_f1 = 0.7609
```

MLP는 `macro_f1 = 0.6898`, `accuracy = 0.7525`를 기록했습니다. PCA/SMOTE ablation은 보조 실험으로 유지했으며, 전체 결론은 동일한 전처리와 split 안에서 모델들을 비교한 결과를 중심으로 해석했습니다.

## Key Insights

### Why Random Forest Outperformed MLP

이 프로젝트의 데이터는 이미지나 텍스트가 아니라 4,424행 규모의 구조화된 tabular dataset입니다. 이런 표형 데이터에서는 Random Forest 같은 tree-based ensemble이 비선형 관계와 변수 간 상호작용을 비교적 안정적으로 포착할 수 있습니다.

반면 MLP는 더 많은 데이터와 세심한 튜닝이 필요하며, 소규모 tabular dataset에서는 항상 우위를 보장하지 않습니다. 따라서 본 결과는 딥러닝이 항상 더 좋은 선택은 아니며, 산업 현장의 tabular prediction 문제에서는 데이터 구조에 맞는 모델 선택이 중요하다는 점을 보여줍니다.

### Feature Importance Interpretation

이 프로젝트의 핵심 가치는 단순 성능 비교뿐 아니라, feature importance를 통해 예측에 영향을 미치는 주요 변수를 확인했다는 점입니다. Random Forest 기준으로 상위 변수에는 다음과 같은 항목이 포함되었습니다.

- 학기별 승인 과목 수
- 학기별 성적
- 학기별 평가 횟수
- 등록금 납부 여부
- 장학금 여부
- 입학 시 나이 및 입학 성적

이 변수들은 학생의 학업 지속 가능성과 밀접한 관련이 있을 수 있습니다. 따라서 본 프로젝트는 단순 분류 모델이 아니라, 학생 지원 또는 조기경보 관점에서 해석 가능한 분석 결과를 제공합니다.

다만 feature importance는 인과관계를 의미하지 않습니다. 실제 개입 정책을 설계하려면 학교 현장의 도메인 검증, 추가 데이터, 공정성 검토가 필요합니다.

### Implication for Student Dropout Early-Warning

본 모델은 학생의 최종 상태를 단정하는 도구가 아니라, 지원이 필요할 가능성이 있는 학생을 더 빨리 살펴보기 위한 decision-support 관점에서 해석하는 것이 적절합니다. 특히 학업 성과와 재정 관련 지표가 함께 중요하게 나타났다는 점은 학생 지원 전략을 설계할 때 학업 지도와 행정적 지원을 함께 고려해야 함을 시사합니다.

## Limitations

- 현재 기본 실험은 1학기 및 2학기 학업 성과 변수를 모두 사용하므로, 결과는 "최종 상태 분류" 관점으로 해석하는 것이 적절합니다.
- 일부 중요 변수는 학기별 성과 변수이므로 입학 직후의 조기 예측에는 사용할 수 없을 수 있습니다.
- feature importance는 상관적 신호를 보여줄 뿐 인과관계를 증명하지 않습니다.
- 데이터셋 규모가 비교적 작기 때문에 MLP가 충분한 representation learning 이점을 얻기 어려울 수 있습니다.
- 예측 결과는 학생 지원을 위한 참고 자료로 사용해야 하며, 학생에 대한 최종 판단이나 불이익 부여에 사용되어서는 안 됩니다.

## How to Run

1. 필요한 패키지를 설치합니다.

```powershell
pip install -r requirements.txt
```

2. 데이터 CSV를 `data/raw/` 아래에 넣고 전처리를 실행합니다.

```powershell
python scripts/prepare_data.py
```

target 컬럼을 직접 지정하려면 다음처럼 실행합니다.

```powershell
python scripts/prepare_data.py --target Target
```

3. 기본 머신러닝 모델을 학습합니다.

```powershell
python scripts/train_ml.py
```

4. MLP 모델을 학습합니다.

```powershell
python scripts/train_mlp.py
```

5. GridSearchCV로 머신러닝 모델을 튜닝합니다.

```powershell
python scripts/tune_ml.py
```

6. PCA/SMOTE 비교 실험을 실행하고 결과를 요약합니다.

```powershell
python scripts/run_experiments.py
python scripts/summarize_metrics.py
```

7. Confusion matrix와 feature importance 분석 파일을 생성합니다.

```powershell
python scripts/analyze_results.py
```

8. 최종 보고서, 발표 자료, 가이드라인 맞춤 산출물을 생성합니다.

```powershell
python scripts/generate_final_assets.py
python scripts/generate_guideline_assets.py
```

## Repository Structure

- `src/data.py`: 데이터 로드, 컬럼명 정리, target 추론, 전처리, split 저장
- `src/evaluation.py`: 공통 평가 지표 계산
- `scripts/prepare_data.py`: 전처리 실행
- `scripts/train_ml.py`: 기본 ML 모델 학습
- `scripts/train_mlp.py`: MLP 학습
- `scripts/tune_ml.py`: GridSearchCV 튜닝
- `scripts/run_experiments.py`: PCA/SMOTE 실험
- `scripts/analyze_results.py`: confusion matrix 및 feature importance 생성
- `scripts/generate_final_assets.py`: 일반 최종 보고서/슬라이드 생성
- `scripts/generate_guideline_assets.py`: 과제 가이드라인 맞춤 보고서/appendix 생성
- `reports/metrics/summary.csv`: 전체 모델 성능 요약
- `reports/metrics/random_forest_tuned_feature_importance.csv`: 변수 중요도
- `reports/final_report_guideline_aligned.md`: 과제 가이드라인에 맞춘 최종 보고서
- `reports/presentation_slides.html`: 발표 슬라이드 HTML

## Public Code Baseline

프로포절에 적은 Kaggle public code 3개는 외부 baseline/reference로 사용했습니다. 다만 Kaggle notebook의 정확한 split과 metric을 동일하게 재현한 것은 아니므로, 본 보고서에서는 "public code보다 절대적으로 우수하다"고 주장하지 않습니다. 대신 동일한 전처리와 train/test split 안에서 ML/DL 모델을 공정하게 비교하고, PCA/SMOTE ablation과 오류 분석을 추가한 점을 originality로 정리했습니다.
