# 학생 학업 성취도 및 중도 포기 예측 프로젝트

Team 11의 NOVA50101 Introduction to Artificial Intelligence for Industrial AI 최종 프로젝트 작업 공간입니다. Kaggle/UCI의 `Predict students' dropout and academic success` 데이터셋을 사용하여 전통적 머신러닝 모델과 MLP 딥러닝 모델을 비교합니다.

## 프로젝트 목표

학생 데이터를 바탕으로 `Dropout`, `Enrolled`, `Graduate` 세 클래스를 예측하고, 어떤 요인이 학업 성공과 중도 포기 예측에 중요한지 분석합니다. 단순 성능 비교에 그치지 않고 EDA, 전처리 실험, ML/DL 비교, 오류 분석, 한계점 및 향후 연구를 함께 정리합니다.

## 데이터셋

- Dataset name: `Predict students' dropout and academic success`
- Kaggle URL: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
- UCI mirror: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Dataset size: 4,424 rows / 36 input columns + target
- Target classes: `Dropout`, `Enrolled`, `Graduate`

원본 CSV는 용량과 재배포 조건을 고려하여 Git에는 포함하지 않습니다. 실행하려면 Kaggle 또는 UCI에서 받은 CSV 파일을 `data/raw/` 아래에 넣어 주세요. 예시는 다음과 같습니다.

```text
data/raw/student_dropout_success/data.csv
```

## 예측 시점 주의

현재 기본 실험은 데이터셋의 1학기 및 2학기 학업 성과 변수를 모두 사용하여 최종 학생 상태를 분류합니다. 따라서 이 결과는 "최종 상태 분류" 관점의 모델로 해석하는 것이 적절합니다.

만약 보고서에서 "조기 중도탈락 예측"을 강조한다면, 2학기 성적 관련 변수는 미래 정보에 가까울 수 있습니다. 이 경우 입학 정보와 1학기 변수만 사용하는 별도 실험을 추가하는 것이 더 엄밀합니다.

## 전처리

`scripts/prepare_data.py`는 다음 과정을 수행합니다.

- CSV 자동 탐색 또는 `--input`으로 지정한 CSV 로드
- BOM, 앞뒤 공백, 탭 등 컬럼명 정리
- target 컬럼 자동 추론
- 중복 행 제거
- `stratify=y`를 적용한 train/test split
- 숫자형 변수 표준화
- 명목형 코드 변수 one-hot encoding
- 선택 옵션으로 PCA 및 SMOTE 적용
- 전처리된 데이터와 preprocessor를 `data/processed/`에 저장

SMOTE는 train split에만 적용되며, test split에는 적용하지 않습니다.

## 실행 순서

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

VS Code의 `.venv`를 사용할 경우 다음처럼 실행할 수 있습니다.

```powershell
& "c:/Users/Administrator/project/AI team/.venv/Scripts/python.exe" "c:/Users/Administrator/project/AI team/scripts/generate_guideline_assets.py"
```

## 주요 산출물

- `data/processed/`: 전처리된 train/test split 및 전처리 파이프라인
- `models/`: 학습된 모델 파일
- `reports/metrics/summary.csv`: 전체 모델 성능 요약
- `reports/metrics/confusion_matrices.csv`: 모델별 confusion matrix
- `reports/metrics/random_forest_tuned_feature_importance.csv`: 변수 중요도
- `reports/final_report_guideline_aligned.md`: 과제 가이드라인에 맞춘 최종 보고서
- `reports/final_report_guideline_aligned.html`: HTML 보고서
- `reports/activity_appendix.md`: 기존/수정 활동 계획 및 팀원 기여도
- `reports/presentation_slides.html`: 발표 슬라이드 HTML

## 사용 모델

- Logistic Regression
- SVM with RBF kernel
- Random Forest
- MLP trained with SGD/backpropagation
- KMeans for unsupervised analysis

## 현재 최고 결과

현재 실험에서 가장 좋은 모델은 `random_forest`입니다.

```text
macro_f1 = 0.7032
accuracy = 0.7627
weighted_f1 = 0.7609
```

MLP는 `macro_f1 = 0.6898`, `accuracy = 0.7525`를 기록했습니다. 작은 tabular dataset에서는 Random Forest 계열 모델이 MLP보다 더 안정적으로 좋은 성능을 보였습니다.

## Public Code Baseline 관련 주의점

프로포절에 적은 Kaggle public code 3개는 외부 baseline/reference로 사용했습니다. 다만 Kaggle notebook의 정확한 split과 metric을 동일하게 재현한 것은 아니므로, 본 보고서에서는 "public code보다 절대적으로 우수하다"고 주장하지 않습니다. 대신 동일한 전처리와 train/test split 안에서 ML/DL 모델을 공정하게 비교하고, PCA/SMOTE ablation과 오류 분석을 추가한 점을 originality로 정리했습니다.

## 코드 구성

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
