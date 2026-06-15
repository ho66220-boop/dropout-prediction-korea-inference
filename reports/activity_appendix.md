# 활동 Appendix

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
