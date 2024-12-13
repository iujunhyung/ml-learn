# 데이터 전처리 (Data Preprocessing)

데이터 전처리는 머신러닝 파이프라인에서 매우 중요한 단계로, 정확하고 효율적인 모델 학습을 위해 반드시 거쳐야 하는 과정입니다. 적절한 전처리가 없다면 모델이 노이즈나 불균형, 잘못된 스케일의 데이터에 의해 오차가 커지거나 편향된 결과를 내놓게 됩니다. 데이터 전처리는 일반적으로 다음과 같은 단계를 포함하며, 각 단계를 신중하고 체계적으로 수행하는 것이 중요합니다.



## 1. 데이터 정제 (Data Cleaning)

데이터 정제는 수집된 원시 데이터(raw data)에서 발생할 수 있는 불완전하거나 오류가 있는 부분을 손질하는 과정입니다. 이 단계는 모델이 “깨끗한” 데이터를 바탕으로 학습할 수 있도록 돕습니다.

### 일관성 검사(Consistency Check):
- 데이터의 형식이 일관되지 않거나, 범주형 변수의 값이 잘못된 경우 수정합니다.
- 예를 들어, 날짜 데이터의 형식이 다른 경우 통일하거나, 범주형 변수의 오타를 수정하고 타입을 통일합니다.

```python
# 데이터 형식 확인
df.info()
```

### 결측치 처리(Missing Values):
- 데이터에 빈값이 포함된 경우, 해당 결측치를 적절히 처리해야 합니다.
- 결측치가 적은 경우 해당 행(또는 열)을 제거하지만 데이터 손실을 최소화하기 위해 대체(Imputation) 기법을 사용하여 채울 수 있습니다.
- 대체(Imputation):
  - 평균(mean), 중앙값(median), 최빈값(mode)으로 채우기
  - KNN Imputation, MICE(Multiple Imputation by Chained Equations), Regression Imputation 등 고급 기법 활용

```python
# 결측치 확인
df.isnull().sum()

# 결측치 제거
df = df.dropna()

# 평균값으로 결측치 대체
df = df.fillna(df.mean())

# 중앙값으로 결측치 대체
df = df.fillna(df.median())

# 최빈값으로 결측치 대체
df = df.fillna(df.mode().iloc[0])

# KNN Imputation

# MICE Imputation

# Regression Imputation

```

### 중복 데이터 제거(Duplication):
- 동일한 레코드가 여러 번 등장하는 경우 제거
- 유니크한 ID또는 날짜, 시간 등을 기준으로 중복 여부를 확인하여 제거

```python
# 중복 데이터 확인
df.duplicated().sum()

# 중복 데이터 제거
df = df.drop_duplicates()
```

### 이상치 처리(Outliers):
- 이상치는 데이터 분포에 비정상적으로 높거나 낮은 값으로, 모델의 성능을 저하시키므로 처리가 필요합니다.
- 이상치 감지 기법: Boxplot, Z-score, IQR(사분위 범위)
- 이상치 처리 방법: 제거, 변환(로그 스케일), 혹은 이상치를 별도 모델링

```python
# 이상치 확인
sns.boxplot(x=df['feature'])

# 이상치 제거
```


## 2. 데이터 변환 (Data Transformation)

데이터 변환은 모델이 데이터를 더 잘 이해하고 처리할 수 있도록 형식, 스케일, 구조 등을 변환하는 과정입니다.

### 스케일링(Scaling):
데이터가 다양한 단위나 범위로 구성되어 있을 때, 스케일링을 통해 데이터의 범위를 일정하게 조정합니다.

- 표준화(Standardization): 평균을 0, 표준편차를 1로 변환 (예: StandardScaler)
- 정규화(Normalization): 데이터의 값을 0~1 사이로 변환 (예: MinMaxScaler)
- 로버스트 스케일링(Robust Scaling): 이상치에 민감하지 않도록 변환 (예: RobustScaler)

### 인코딩(Encoding):
데이터를 모델이 이해할 수 있는 형태로 변환하는 과정으로, 범주형 데이터 또는 텍스트 데이터를 수치형 데이터로 변환합니다.

- 명목형 범주 변수: One-hot Encoding, Dummy Encoding
- 순서형 범주 변수: Label Encoding, Ordinal Encoding
- 텍스트 데이터: TF-IDF, Word2Vec, GloVe, BERT 임베딩 등 자연어처리 기법 활용

### 정규화(Normalization) & 변환:
데이터가 정규분포를 따르지 않거나, 특정 변수가 치우친 경우 변환을 통해 데이터의 분포를 조정합니다.

- 로그 변환(log transform): 분포가 한쪽으로 치우친(skewed) 데이터를 정규분포 형태에 가깝게 변환
- Box-Cox, Yeo-Johnson 변환: 로그 변환이 어려운 경우 이용



## 3. 데이터 분할 (Data Splitting)

데이터 분할은 모델 성능을 정확히 평가하기 위한 핵심적인 단계입니다. 이를 통해 모델은 학습에 사용되지 않은 데이터(테스트 세트)로 검증되어, 과적합 여부를 점검할 수 있습니다.

### 기본 분할:
- 일반적으로 70-80%를 학습(train) 데이터, 20-30%를 테스트(test) 데이터로 분할

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 검증 세트(Validation set):
- 학습 과정 중 하이퍼파라미터 튜닝이나 모델 선택을 위해 별도의 검증 세트를 둘 수도 있음
- K-Fold 교차 검증(K-fold cross validation) 기법을 통해 학습 데이터를 훈련 및 검증 세트로 반복 분할하여 사용



## 4. 피처 엔지니어링 (Feature Engineering)

피처 엔지니어링은 단순한 전처리 과정을 넘어, 데이터에 내재된 패턴을 더욱 잘 모델링하기 위해 새로운 변수를 만들거나 기존 변수를 가공하는 과정입니다.

- 시간 데이터에서 요일, 월, 계절 등의 추출
- 텍스트 데이터에서 단어 수, 문장 길이, 키워드 빈도 등의 메타피처 생성
- 이미지 데이터에서 색상 히스토그램, 에지(edge) 특징, 특정 객체 유무 등 추출
- 다양한 통계량(평균, 분산, 최대값, 최소값 등)을 활용한 피처 생성
- 다양한 데이터 간의 상호작용(interaction) 및 다항식(polynomial) 피처 생성
