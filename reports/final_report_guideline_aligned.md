# 과제 가이드라인 맞춤 최종 보고서

## 프로젝트 개요

- 과목: NOVA50101 Introduction to Artificial Intelligence for Industrial AI
- 팀: Team 11
- 팀원: 박재우, 염지훈, 오형우
- 프로젝트 유형: Application project
- 프로젝트 제목: 학업 성취도 및 중도 포기 예측을 위한 머신러닝과 딥러닝 모델의 비교 연구
- 데이터셋: Predict Students' Dropout and Academic Success from Kaggle/UCI

본 프로젝트는 학생이 중도 포기할지, 계속 재학 중일지, 졸업할지를 예측하는 교육 AI 문제를 다룬다. 과제 가이드라인에 맞춰 전통적 머신러닝, 딥러닝, 전처리 실험, 오류 분석, 결과 해석을 함께 수행했다.

## Public Code Baseline 및 Originality

프로포절에서는 다음 Kaggle public code를 외부 baseline/reference로 제시했다.

- ML Algorithms Usage and Prediction: https://www.kaggle.com/code/sunayanagawde/ml-algorithms-usage-and-prediction  
  본 프로젝트에서의 역할: 전통적 ML workflow와 비교를 위한 public-code baseline.
- Dropout Graduate Analysis: https://www.kaggle.com/code/satyaprakashshukl/droput-graduate-analysis  
  본 프로젝트에서의 역할: EDA 중심의 dropout/graduate 분석을 참고하기 위한 public-code baseline.
- Student Dropout Analysis for School Education: https://www.kaggle.com/code/jeevabharathis/student-dropout-analysis-for-school-education  
  본 프로젝트에서의 역할: 교육 분야 dropout 분석을 참고하기 위한 public-code baseline.

Kaggle notebook의 출력값과 split 방식은 정적 페이지에서 항상 동일하게 재현하기 어렵기 때문에, 본 보고서는 해당 public code보다 절대적으로 성능이 우수하다고 주장하지 않는다. 대신 이 public code들을 참고 baseline으로 삼고, 모든 모델을 동일한 전처리와 train/test protocol 안에서 비교하는 재현 가능한 파이프라인을 구축했다.

본 프로젝트의 originality는 다음 지점에 있다.

- 하나의 공통 전처리 파이프라인 아래에서 ML/DL 모델을 통제된 방식으로 비교
- PCA와 SMOTE ablation 실험 수행
- 전통적 ML 모델에 대한 GridSearchCV 튜닝
- SGD/backpropagation 기반 MLP 학습
- feature importance와 confusion matrix 기반 오류 분석
- feature importance를 통해 학생 상태 예측에 중요한 학업·행정 요인을 해석
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

| model                       | accuracy | macro_precision | macro_recall | macro_f1 | weighted_f1 |
| --------------------------- | -------- | --------------- | ------------ | -------- | ----------- |
| random_forest               | 0.7627   | 0.7091          | 0.6993       | 0.7032   | 0.7609      |
| logistic_regression_tuned   | 0.7379   | 0.7113          | 0.7166       | 0.7024   | 0.7510      |
| random_forest_tuned         | 0.7458   | 0.7029          | 0.7052       | 0.6994   | 0.7531      |
| svm_rbf                     | 0.7412   | 0.7010          | 0.6972       | 0.6982   | 0.7493      |
| logistic_regression_pca_smote | 0.7299 | 0.7039          | 0.7094       | 0.6959   | 0.7430      |
| random_forest_base          | 0.7412   | 0.6962          | 0.6963       | 0.6949   | 0.7479      |
| random_forest_smote         | 0.7492   | 0.6965          | 0.6890       | 0.6914   | 0.7495      |
| mlp_sgd                     | 0.7525   | 0.7017          | 0.6833       | 0.6898   | 0.7490      |

가장 좋은 모델은 `random_forest`이며, macro F1은 `0.7032`, accuracy는 `0.7627`이다. MLP baseline은 macro F1 `0.6898`, accuracy `0.7525`를 기록했다. 두 모델의 차이가 매우 크다고 볼 수는 없지만, 동일한 split과 전처리 조건에서 Random Forest 계열 모델이 더 안정적인 macro F1을 보였다.

이 프로젝트의 데이터는 이미지나 텍스트가 아니라 4,424행 규모의 구조화된 tabular dataset이다. 이러한 표형 데이터에서는 Random Forest 같은 tree-based ensemble이 비선형 관계와 변수 간 상호작용을 비교적 안정적으로 포착할 수 있다. 특히 학업 성과, 등록금 납부 여부, 장학금 여부처럼 서로 다른 성격의 변수들이 섞인 데이터에서는 tree 기반 모델이 복잡한 feature interaction을 명시적 가정 없이 다룰 수 있다는 장점이 있다.

반면 MLP는 더 많은 데이터와 세심한 구조·학습률·정규화 튜닝이 필요하며, 소규모 tabular dataset에서는 반드시 우위를 보장하지 않는다. 따라서 본 결과는 딥러닝이 항상 더 좋은 것은 아니며, 산업 현장의 tabular prediction 문제에서는 데이터 구조에 맞는 모델 선택이 중요하다는 시사점을 준다. 즉, 실제 산업 AI 적용에서는 최신 모델을 사용하는 것보다 문제의 데이터 형태, 표본 수, 해석 가능성 요구를 함께 고려하는 것이 중요하다.

## 변수 중요도 및 인사이트

성능 비교 다음으로 중요한 해석 축은 feature importance이다. 본 프로젝트의 핵심 인사이트는 어떤 모델이 더 높은 점수를 냈는지뿐 아니라, 학생 상태 예측에서 어떤 학업·행정 변수가 중요한 신호로 작용했는지를 확인하는 데 있다.

Random Forest 기준 상위 feature는 다음과 같다.

1. Curricular units 2nd sem (approved): 0.1304
2. Curricular units 2nd sem (grade): 0.1075
3. Curricular units 1st sem (approved): 0.0925
4. Curricular units 1st sem (grade): 0.0706
5. Curricular units 2nd sem (evaluations): 0.0472
6. Curricular units 1st sem (evaluations): 0.0441
7. Tuition fees up to date = 1: 0.0401
8. Age at enrollment: 0.0323
9. Tuition fees up to date = 0: 0.0323
10. Admission grade: 0.0266

이 프로젝트의 중요한 가치는 단순히 모델 성능을 비교하는 데 그치지 않고, feature importance를 통해 학생 상태 예측에 영향을 미치는 주요 변수를 확인했다는 점이다. 상위 변수에는 학기별 승인 과목 수, 학기별 성적, 평가 횟수, 등록금 납부 여부, 입학 시 나이와 입학 성적이 포함되었다. 또한 전체 feature importance 결과에서는 장학금 여부도 주요 변수군 안에 포함되어, 재정·지원 관련 변수가 학생의 학업 지속 가능성과 연결될 수 있는 맥락을 제공한다.

이 결과는 학생의 중도탈락 또는 졸업 여부가 입학 당시 정보만으로 결정되는 것이 아니라, 입학 후 학업 진행 상황과 행정적·재정적 상태가 함께 반영되는 문제임을 시사한다. 따라서 본 프로젝트는 단순 분류 모델이라기보다, 학생 지원 또는 조기경보 관점에서 해석 가능한 분석 결과를 제공한다. 다만 변수 중요도는 인과관계를 의미하지 않으며, 실제 개입 정책 설계에는 학교 현장의 도메인 검증과 추가 분석이 필요하다. 또한 실제 조기 개입 모델로 활용하려면 2학기 정보를 포함한 전체 데이터가 아니라, 입학 정보 또는 1학기 정보만 사용하는 별도 실험을 수행해야 한다.

산업 AI 관점에서 본 프로젝트는 데이터 구조와 의사결정 목적에 맞는 모델 선택의 중요성을 보여준다. 교육 현장의 학생 상태 예측 문제는 고차원 이미지나 자연어 문제가 아니라, 학업 이력과 행정 정보가 결합된 tabular prediction 문제에 가깝다. 따라서 해석 가능성과 안정성이 중요한 상황에서는 Random Forest와 같은 tree-based ensemble이 실용적인 baseline이 될 수 있다.

또한 feature importance 결과는 모델을 단순히 "맞히는 도구"로 사용하는 것보다, 어떤 영역에서 학생 지원 신호가 나타나는지 파악하는 데 활용될 가능성을 보여준다. 예를 들어 학기별 승인 과목 수와 성적은 학업 진행 상태를, 등록금 납부 여부와 장학금 여부는 재정적·행정적 지원 필요성을 간접적으로 시사할 수 있다. 단, 이러한 결과는 학생을 자동으로 분류하거나 낙인찍기 위한 근거가 아니라, 추가 상담과 지원 여부를 검토하기 위한 decision-support 정보로 해석해야 한다.

## 오류 분석

confusion matrix를 보면 `Enrolled` class가 가장 분류하기 어려웠다. 이는 `Enrolled`가 중간 상태이기 때문이다. 현재 재학 중인 학생은 이후 졸업할 수도 있고 중도 포기할 수도 있으므로, `Dropout` 및 `Graduate`와 decision boundary가 겹칠 수 있다.

## Ablation 분석

PCA와 SMOTE는 보조적인 전처리 변형 실험으로 수행했다. SMOTE와 PCA가 일부 모델에서 성능 변화를 만들었지만, 전체 결론을 바꿀 정도의 큰 개선으로 해석하기는 어렵다. 특히 PCA는 tabular feature가 가진 해석 가능한 신호를 압축하는 과정에서 성능 또는 해석력을 낮출 수 있으므로, 본 프로젝트에서는 원본 feature 기반의 모델 비교와 feature importance 해석을 중심 결과로 보았다.

## 비지도 학습 분석

target class 수와 동일하게 KMeans를 3개 cluster로 실행했다. 결과는 다음과 같다.

- Normalized Mutual Information: `0.1559`
- Adjusted Rand Index: `0.1519`
- Inertia: `102807.14`

비지도 cluster와 실제 label의 정렬 정도가 낮다는 점은, 세 target class가 단순한 거리 기반 clustering만으로 명확히 분리되지 않음을 보여준다. 따라서 supervised learning model이 필요하다는 해석이 가능하다.

## 한계점

- 데이터셋 규모가 딥러닝에 비해 작아 MLP가 representation learning의 장점을 충분히 활용하기 어렵다.
- 일부 중요 feature는 학기별 성과 변수이므로, 입학 직후의 매우 이른 시점에는 사용할 수 없을 수 있다.
- feature importance는 예측에 사용된 변수의 상대적 기여도를 보여주지만, 해당 변수가 학생 상태를 직접적으로 유발한다는 인과관계를 의미하지 않는다.
- Kaggle public-code baseline은 reference로 사용했지만, 정적 notebook 페이지에서 정확한 metric을 동일하게 재현하지는 못했다.
- 모델은 과거 tabular data에 기반하므로, 제도 변화나 개인적 상황을 모두 반영하지 못한다.
- 실제 조기개입 정책에 적용하려면 학교 현장의 도메인 검증, 공정성 검토, 추가 데이터 기반 검증이 필요하다.
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
- Activities: 기존/수정 활동 계획 및 팀원 기여도를 appendix로 제시

## 그림

- [class_distribution.png](figures/class_distribution.png)
- [key_feature_distributions.png](figures/key_feature_distributions.png)
- [model_performance.png](figures/model_performance.png)
- [experiment_comparison.png](figures/experiment_comparison.png)
- [feature_importance.png](figures/feature_importance.png)
- [best_confusion_matrix.png](figures/best_confusion_matrix.png)

## Appendix

[activity_appendix.md](activity_appendix.md)를 참고한다.
