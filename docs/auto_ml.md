# AutoML
AutoML(Automated Machine Learning)은 머신러닝 모델 개발 과정을 자동화하는 기술로, 데이터 전처리, 특성 선택, 모델 선택, 하이퍼파라미터 튜닝 등을 자동으로 수행합니다. AutoML은 전문가 수준의 머신러닝 모델을 빠르게 개발하고, 최적화된 모델을 생성하는 데 도움이 됩니다.

## AutoML 라이브러리
AutoML 라이브러리는 머신러닝 및 딥러닝 모델의 개발 과정을 자동화하는 오픈소스 도구로, 주로 데이터 전처리, 모델 선택, 하이퍼파라미터 튜닝 등을 지원합니다.

- **Auto-sklearn**: Scikit-learn 기반으로, 모델 선택 및 하이퍼파라미터 최적화를 자동화하며 Bayesian Optimization을 사용(https://automl.github.io/auto-sklearn/)
```python
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
```

- **TPOT**: 유전 알고리즘을 활용하여 머신러닝 파이프라인을 최적화하며, Scikit-learn과 호환(https://epistasislab.github.io/tpot/)
```python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
```

- **H2O AutoML**: 분산 환경에서 다양한 알고리즘(GBM, Random Forest 등)을 지원하며, R과 Python 인터페이스 제공.(https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

- **AutoKeras**: Keras 기반으로 신경망 구조 탐색(NAS)을 지원하는 딥러닝 AutoML 도구.(https://autokeras.com/)

- **PyCaret**: 저코드 기반의 ML 워크플로우 자동화 도구로, 분류, 회귀, 클러스터링 등을 지원.(https://pycaret.gitbook.io/docs)

- **FLAML**: 경량화된 AutoML 라이브러리로 빠르고 효율적인 모델 학습에 초점.

- **AutoGluon**: 딥러닝 및 머신러닝 작업을 위한 간단한 API 제공.(https://auto.gluon.ai/)
```python
from autogluon.tabular import TabularDataset, TabularPredictor

data_root = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
train_data = TabularDataset(data_root + 'train.csv')
test_data = TabularDataset(data_root + 'test.csv')

predictor = TabularPredictor(label='class').fit(train_data=train_data)
predictions = predictor.predict(test_data)
```

## AutoML 서비스
AutoML 서비스는 사용자가 복잡한 코딩 없이도 머신러닝 모델을 개발하고 배포할 수 있도록 지원합니다.

- 클라우드 기반 AutoML 서비스
  - **Google Cloud AutoML (Vertex AI)**:
    - URL: https://cloud.google.com/automl
    - 비전, 자연어 처리(NLP), 표 데이터 등 다양한 도메인에서 커스텀 모델 생성 가능.
    - 그래픽 인터페이스와 Python API 제공.
  - **Amazon SageMaker Autopilot**:
    - URL: https://aws.amazon.com/sagemaker/autopilot
    - 데이터 전처리부터 모델 배포까지 자동화.
    - SHAP 기반 변수 중요도 분석 등 모델 해석 기능 포함.
  - **Microsoft Azure AutoML**:
    - URL: https://azure.microsoft.com/ko-kr/solutions/automated-machine-learning/
    - 분류, 회귀, 시계열 예측 등 다양한 작업 지원.
    - Python SDK를 통해 ML 파이프라인 자동화 및 모델 해석 기능 제공.

- 솔루션 기반 AutoML 서비스
  - **H2O Driverless AI**:
    - URL: https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/
    - 자동 특징 공학 및 하이퍼파라미터 튜닝.
    - 설명 가능한 AI(XAI) 기능 제공.
  - **DataRobot**:
    - URL: https://www.datarobot.com/product/ai-platform/
    - Video: https://youtu.be/vyi_0D-rJ1A?feature=shared
    - 데이터 준비부터 모델 배포까지 엔드투엔드 솔루션.
    - 민감한 데이터를 다룰 수 있는 온프레미스 설치 옵션 제공.
