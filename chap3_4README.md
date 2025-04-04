# 인공신경망 핵심 개념 정리

TensorFlow와 Keras를 활용한 인공신경망 학습, 최적화, 활성화 함수 정리

## 목차
1. [학습 최적화 (Fine Tuning)](#1-학습-최적화-fine-tuning)
2. [활성화 함수 (Activation Functions)](#2-활성화-함수-activation-functions)
3. [배치 정규화 (Batch Normalization)](#3-배치-정규화-batch-normalization)
4. [가중치 초기화 (Weight Initialization)](#4-가중치-초기화-weight-initialization)

## 1. 학습 최적화 (Fine Tuning)

### 학습률(Learning Rate)
- **정의**: 경사 하강법에서 한 번의 스텝에서 얼마나 이동할지 결정하는 하이퍼파라미터
- **중요성**: 너무 크면 발산하고, 너무 작으면 수렴이 느림
- **최적화 방법**:
  - Learning Rate Scheduler: 학습 중 점진적으로 학습률 감소
  - 지수적으로 학습률을 증가시키며 loss가 급증하는 지점 찾기

```python
# 학습률 스케줄러 예시
optimizer = tf.keras.optimizers.SGD(
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=10000,
        decay_rate=0.9
    )
)
```

### Early Stopping
- **목적**: 과적합 방지, 학습 시간 단축
- **작동 방식**: 검증 세트의 성능이 특정 에포크 동안 개선되지 않으면 학습 중단
- **구현**:

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

### 모델 체크포인트
- **목적**: 학습 과정 중 최고 성능의 모델 저장
- **구현**:

```python
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    save_best_only=True
)
```

### 평가 지표 (Metrics)

#### 정확도 (Accuracy)
- 올바르게 예측된 샘플의 비율
- 불균형 데이터에서는 오해의 소지가 있음

#### 정밀도 (Precision)
- 양성으로 예측한 것 중 실제 양성인 비율
- `TP/(TP+FP)`
- 거짓 양성을 최소화하고 싶을 때 중요

#### 재현율 (Recall)
- 실제 양성 중 양성으로 예측한 비율
- `TP/(TP+FN)`
- 거짓 음성을 최소화하고 싶을 때 중요

#### F1 점수
- 정밀도와 재현율의 조화 평균
- `2 * (precision * recall) / (precision + recall)`

### 교차 검증 (Cross Validation)
- **목적**: 모델의 일반화 성능을 더 정확히 평가
- **k-fold 교차 검증**: 데이터를 k개의 폴드로 나누고, 각 폴드를 한 번씩 검증 세트로 사용

### 하이퍼파라미터 튜닝
- **그리드 탐색(Grid Search)**: 모든 하이퍼파라미터 조합을 시도
- **랜덤 탐색(Random Search)**: 무작위로 하이퍼파라미터 조합을 시도
- **베이지안 최적화**: 이전 결과를 바탕으로 유망한 조합 탐색

### 규제화 (Regularization)
- **L1 규제**: 가중치의 절대값 합에 페널티 (희소성 촉진)
- **L2 규제**: 가중치의 제곱값 합에 페널티 (가중치 분산 감소)
- **드롭아웃(Dropout)**: 학습 중 무작위로 뉴런을 비활성화

## 2. 활성화 함수 (Activation Functions)

### ReLU (Rectified Linear Unit)
- **수식**: `f(x) = max(0, x)`
- **장점**: 계산 효율성, 기울기 소실 문제 완화
- **단점**: Dead Neuron 문제 (음수 입력에 대해 항상 0 출력)

### Leaky ReLU
- **수식**: `f(x) = max(αx, x)` (일반적으로 α=0.01)
- **장점**: Dead Neuron 문제 해결
- **단점**: α는 추가 하이퍼파라미터

```python
# Keras에서 구현
tf.keras.layers.LeakyReLU(alpha=0.01)
```

### PReLU (Parametric ReLU)
- **수식**: `f(x) = max(αx, x)` (α는 학습 가능한 파라미터)
- **장점**: α를 데이터로부터 학습
- **구현**:

```python
tf.keras.layers.PReLU()
```

### ELU (Exponential Linear Unit)
- **수식**: `f(x) = x if x > 0 else α(e^x - 1)` (일반적으로 α=1)
- **장점**: 평균 활성화가 0에 가까움, 음수 입력에 대한 완만한 기울기
- **구현**:

```python
tf.keras.layers.Dense(units, activation='elu')
```

### SELU (Scaled ELU)
- **특징**: 자기 정규화 특성, 심층 네트워크에서 안정적인 학습
- **사용 조건**: 완전 연결 계층에서 'lecun_normal' 초기화와 함께 사용
- **구현**:

```python
tf.keras.layers.Dense(units, activation='selu', kernel_initializer='lecun_normal')
```

### Swish 및 Mish
- **최신 활성화 함수**: 성능이 뛰어난 최신 활성화 함수들
- **Swish**: `f(x) = x * sigmoid(x)`
- **Mish**: `f(x) = x * tanh(softplus(x))`

## 3. 배치 정규화 (Batch Normalization)

### 개념과 효과
- **목적**: 내부 공변량 이동(Internal Covariate Shift) 감소
- **작동 방식**: 각 층의 입력을 평균 0, 분산 1로 정규화
- **효과**:
  - 학습 속도 향상
  - 초기화에 대한 의존성 감소
  - 규제 효과
  - 더 높은 학습률 사용 가능

### 구현

```python
# 배치 정규화 적용
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    # 더 많은 층...
])

# 또는 배치 정규화를 활성화 함수 이전에 적용 (대체 방법)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, kernel_initializer='he_normal', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    # 더 많은 층...
])
```

## 4. 가중치 초기화 (Weight Initialization)

### 기울기 소실/폭발 문제
- **문제**: 심층 네트워크에서 기울기가 전파되는 과정에서 사라지거나 폭발함
- **해결책**: 적절한 가중치 초기화와 활성화 함수 선택

### Xavier/Glorot 초기화
- **목적**: 시그모이드, 탄젠트 활성화 함수에 적합
- **방법**: 이전 층과 다음 층의 노드 수에 기반하여 가중치 초기화
- **Keras에서의 구현**: `glorot_uniform` (기본값), `glorot_normal`

### He 초기화
- **목적**: ReLU 및 변형 활성화 함수에 적합
- **방법**: Glorot 초기화를 수정하여 ReLU에 맞게 조정
- **Keras에서의 구현**: `he_uniform`, `he_normal`

### LeCun 초기화
- **목적**: SELU와 같은 자기 정규화 활성화 함수에 적합
- **Keras에서의 구현**: `lecun_uniform`, `lecun_normal`

## 그래디언트 클리핑 (Gradient Clipping)
- **목적**: 그래디언트 폭발 문제 방지
- **구현**:

```python
optimizer = tf.keras.optimizers.SGD(clipnorm=1.0)  # 그래디언트 벡터의 L2 노름을 1.0으로 제한
# 또는
optimizer = tf.keras.optimizers.SGD(clipvalue=0.5)  # 그래디언트 값을 -0.5~0.5 범위로 제한
```

## 비교 및 권장사항

| 활성화 함수 | 초기화 방법 | 규제 방법 | 적합한 상황 |
|----------|------------|---------|----------|
| ReLU     | He         | Dropout | 일반적인 상황, CNN |
| Leaky ReLU/PReLU | He  | Dropout | Dead Neuron 문제가 우려될 때 |
| SELU     | LeCun      | AlphaDropout | 매우 깊은 완전 연결 네트워크 |
| Sigmoid/Tanh | Glorot | L2 규제화 | 이진 분류, RNN |

## 실전 구현 예시

```python
# 가중치 초기화와 배치 정규화를 적용한 모델
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(300, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(100, kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, clipnorm=1.0),
    metrics=['accuracy']
)

# 완전한 학습 파이프라인 예시
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("my_model.h5", save_best_only=True)
    ]
)
```
