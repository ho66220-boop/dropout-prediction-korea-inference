# 최종 프로젝트 보고서

## 프로젝트 정보

- 유형: 1인 개인 프로젝트
- 제목: 해외 교육 데이터 기반 중도 이탈 예측과 한국 고1 자퇴 급증에 대한 유추적 해석
- 데이터셋: Predict Students' Dropout and Academic Success (포르투갈 고등교육)
- 과제 유형: Application project

## 1. 연구 목표

본 프로젝트의 목표는 고등교육 학생 데이터를 활용하여 학생의 상태를 `Dropout`, `Enrolled`, `Graduate` 세 class로 예측하는 것이다. 또한 단순히 높은 성능을 내는 모델을 찾는 데 그치지 않고, 전통적 머신러닝 모델과 MLP 딥러닝 모델을 비교하고, 예측에 중요한 변수와 오류 양상을 분석하는 데 중점을 두었다.

나아가 이 분석을 사회적 배경과 연결한다. 최근 한국에서 고등학교 1학년 자퇴자 수가 처음으로 연 1만 명을 넘어서며 사회 문제로 부상했다. 다만 그 상당수는 학업 실패형 이탈이 아니라 검정고시·수능 집중을 위한 "전략적 자퇴"라는 한국적 특수성을 가진다. 한국 고1 자퇴를 직접 담은 데이터를 구하기 어렵기 때문에, 본 프로젝트는 공개된 해외(포르투갈 고등교육) 중도 이탈 데이터에서 이탈의 일반적 구조를 학습한 뒤, 그 해석을 한국 상황에 조심스럽게 유추하는 우회 전략을 택했다. 따라서 한국 관련 서술은 모두 "예측"이 아닌 "유추적 해석"으로 제한한다.

## 2. 데이터 및 전처리

데이터셋은 총 4,424개의 record로 구성되어 있으며, 학생의 입학 정보, 학업 성과, 경제적/사회적 변수, target label을 포함한다.

전처리 과정은 다음과 같다.

- 중복 record 제거
- stratified train/test split
- 수치형 변수 표준화
- 범주형 변수 one-hot encoding
- PCA 적용 여부 비교
- SMOTE 적용 여부 비교

class imbalance가 존재하고 `Enrolled` class가 중간 상태라는 특성이 있으므로, accuracy뿐 아니라 macro F1을 핵심 지표로 사용했다.

## 3. 사용 모델

본 프로젝트에서는 수업에서 다룬 전통적 ML 모델과 딥러닝 모델을 함께 사용했다.

- Logistic Regression
- SVM with RBF kernel
- Random Forest
- MLP with SGD/backpropagation
- KMeans 비지도 분석

전통적 ML 모델에는 GridSearchCV를 적용했고, PCA/SMOTE ablation 실험을 통해 전처리 방식이 성능에 미치는 영향도 비교했다.

## 4. 실험 결과

macro F1 기준 최고 성능 모델은 `random_forest_smote`였다.

```text
best model: random_forest_smote
accuracy: 0.7559
macro_f1: 0.7035
weighted_f1: 0.7582
```

MLP baseline은 다음 성능을 보였다.

```text
model: mlp_sgd
accuracy: 0.7424
macro_f1: 0.6792
weighted_f1: 0.7401
```

작은 tabular dataset에서는 MLP보다 Random Forest 계열 모델이 더 안정적인 성능을 보였다. 이는 tree-based ensemble이 적은 데이터에서도 비선형 feature interaction을 효과적으로 포착할 수 있기 때문으로 해석된다.

## 5. PCA/SMOTE 분석

SMOTE는 Random Forest의 macro F1을 소폭 향상시켰다. 반면 PCA는 대부분의 모델에서 성능을 낮추는 경향을 보였다. 이는 원래 feature들이 해석 가능하고 예측에 필요한 정보를 직접 담고 있으며, 차원 축소 과정에서 일부 유용한 tabular signal이 손실되었을 가능성을 의미한다.

## 6. 변수 중요도

Random Forest 기준 주요 변수는 다음과 같다.

1. Curricular units 2nd sem (approved)
2. Curricular units 2nd sem (grade)
3. Curricular units 1st sem (approved)
4. Curricular units 1st sem (grade)
5. Tuition fees up to date

가장 중요한 변수들은 대부분 학기별 승인 과목 수와 성적이었다. 이는 학생의 중도 포기나 졸업 여부가 입학 당시 정보보다 실제 학업 진행 상황과 강하게 연결되어 있음을 보여준다.

### 6-1. 한국 고1 자퇴에 대한 유추적 해석

데이터가 가리키는 이탈 신호는 "학업·재정 누적 실패"에 가깝다. 즉 과목 이수 부진, 낮은 성적, 등록금 미납이 이탈 위험과 연결된다. 반면 한국 고1의 전략적 자퇴는 오히려 성적이 양호한 학생이 더 나은 입시 전략(검정고시·수능 집중)을 위해 선택하는 경우가 많다. 따라서 동일한 "이탈"이라도 한국형 전략적 자퇴는 모델이 학습한 실패형 이탈과 동기가 반대일 수 있다.

이 간극 자체가 본 프로젝트의 핵심 발견이다. 해외 데이터의 신호 구조를 한국에 곧바로 적용하면 정반대 해석을 낳을 수 있으므로, 유추가 유효한 지점(경제적 사유·학업 부적응에 의한 비전략적 자퇴 이해)과 위험한 지점(성적 양호한 자발적 이탈)을 구분해야 한다. 한국 관련 모든 해석은 정책 제언이 아닌 가설 수준이며, 실제 적용에는 한국 자체 데이터와 도메인 검증이 필요하다.

## 7. 오류 분석

confusion matrix를 보면 `Enrolled` class가 가장 분류하기 어려웠다. `Enrolled`는 최종 결과가 아니라 중간 상태이므로, 이후 졸업할 학생과 중도 포기할 학생이 함께 포함될 수 있다. 따라서 `Dropout` 및 `Graduate` class와 decision boundary가 겹치는 것이 자연스럽다.

## 8. Public Code Baseline과 Originality

프로포절에서 제시한 Kaggle public code 3개는 외부 baseline/reference로 사용했다.

- ML Algorithms Usage and Prediction
- Dropout Graduate Analysis
- Student Dropout Analysis for School Education

다만 해당 notebook들의 정확한 split과 metric을 동일하게 재현한 것은 아니므로, 본 보고서에서는 public code보다 절대적으로 우수하다고 주장하지 않는다. 본 프로젝트의 차별점은 동일한 파이프라인 안에서 ML/DL 모델을 공정하게 비교하고, PCA/SMOTE ablation, KMeans 비지도 분석, feature importance, 오류 분석까지 통합했다는 점이다.

## 9. 한계점

- 데이터 크기가 딥러닝 모델을 충분히 학습시키기에는 크지 않다.
- 일부 중요 feature는 학기 이후에 관측되는 성과 변수이므로, 입학 직후 조기 예측에는 사용할 수 없을 수 있다.
- public Kaggle code의 성능을 동일 split으로 직접 재현하지는 못했다.
- 데이터에 포함되지 않은 개인적/제도적 요인은 반영하기 어렵다.
- 데이터는 포르투갈 고등교육 맥락이므로, 한국 고1 자퇴에 대한 해석은 유추 수준이며 직접적인 예측·정책 근거로 사용할 수 없다.
- 예측 결과는 학생 지원을 위한 참고 자료로 사용해야 하며, 학생에 대한 최종 판단 기준으로 사용해서는 안 된다.

## 10. 향후 연구

- Kaggle public code를 동일 split과 metric으로 재실행하여 직접 비교
- admission-only 또는 first-semester-only feature 기반 early-warning model 구축
- dropout recall 향상을 위한 threshold tuning
- gender, scholarship status, international status에 대한 fairness analysis
- XGBoost/LightGBM/CatBoost 등 추가 ensemble 모델 비교

## 산출물

- `reports/metrics/summary.csv`: 전체 모델 성능 요약
- `reports/figures/model_performance.png`: 모델 성능 비교 그래프
- `reports/figures/experiment_comparison.png`: PCA/SMOTE 실험 비교
- `reports/figures/feature_importance.png`: 변수 중요도
- `reports/figures/best_confusion_matrix.png`: 최고 모델 confusion matrix
- `reports/final_report_guideline_aligned.md`: 과제 가이드라인 맞춤 보고서
- `reports/activity_appendix.md`: 활동 계획 및 수행 내역
