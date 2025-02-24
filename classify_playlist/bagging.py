import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# CSV 파일 읽기
csv1 = pd.read_csv('favorite_songs.csv')  # 좋아하는 노래 CSV
csv2 = pd.read_csv('soso_songs.csv')   # 쏘쏘
csv3 = pd.read_csv('disliked_songs.csv')   # 싫어하는 노래 CSV

# 원하는 열의 이름 리스트
selected = ['valence', 'acousticness', 'energy', 'instrumentalness']  # 내가 뽑은 특징들

# 레이블과 특징 벡터 분리
X1 = csv1[selected]  # 좋아하는 노래에서 선택한 특징 벡터 추출
y1 = csv1.iloc[:, 0]  # 좋아하는 노래의 레이블

X2 = csv2[selected]  # 쏘쏘 노래에서 선택한 특징 벡터 추출
y2 = csv2.iloc[:, 0]  # 쏘쏘 노래의 레이블

X3 = csv3[selected]  # 싫어하는 노래에서 선택한 특징 벡터 추출
y3 = csv3.iloc[:, 0]  # 싫어하는 노래의 레이블


# 데이터 분할
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)

# 학습 및 테스트 데이터 결합
X_train = pd.concat([X_train1, X_train2, X_train3])
y_train = pd.concat([y_train1, y_train2, y_train3])
X_test = pd.concat([X_test1, X_test2, X_test3])
y_test = pd.concat([y_test1, y_test2, y_test3])


# 모델들 정의
model_1 = SVC(probability=True)
model_2 = DecisionTreeClassifier()
model_3 = LogisticRegression()
model_4 = RandomForestClassifier(random_state=42)
model_5 = GaussianNB()
model_6 = KNeighborsClassifier(n_neighbors=5)

# 모델들 훈련
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)
model_3.fit(X_train, y_train)
model_4.fit(X_train, y_train)
model_5.fit(X_train, y_train)
model_6.fit(X_train, y_train)

# 각 모델의 예측 확률
model_1_probs = model_1.predict_proba(X_test)  # 확률 반환
model_2_probs = model_2.predict_proba(X_test)
model_3_probs = model_3.predict_proba(X_test)
model_4_probs = model_4.predict_proba(X_test)
model_5_probs = model_3.predict_proba(X_test)
model_6_probs = model_4.predict_proba(X_test)

# 실제 라벨
y_true = y_test

# 초기 가중치 (모든 모델에 대해 동일한 가중치 부여)
weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# 경사하강법 파라미터
learning_rate = 0.0001
epochs = 1000

# 경사하강법을 사용하여 가중치 최적화
for epoch in range (epochs):
    # 예측 확률의 가중 평균 계산
    weighted_probs = (weights[0] * model_1_probs +
                      weights[1] * model_2_probs +
                      weights[2] * model_3_probs +
                      weights[3] * model_4_probs +
                      weights[4] * model_5_probs +
                      weights[5] * model_6_probs
                      ) / np.sum(weights)

    # Cross-entropy loss 계산
    loss = log_loss(y_true, weighted_probs)

    # 손실에 대한 가중치의 기울기 계산
    gradients = np.zeros_like(weights)
    for i in range(4):
        gradients[i] = np.sum(
            (weighted_probs[:, 1] - y_true) * (eval(f'model_{i + 1}_probs')[:, 1] - model_1_probs[:, 1]))  # 기울기 계산 (예시)

    # 가중치 업데이트 (경사하강법)
    weights -= learning_rate * gradients

    # 주기적으로 손실과 가중치 출력
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Weights: {weights}')

# 최종 가중치로 예측 수행
final_weighted_probs = (weights[0] * model_1_probs +
                        weights[1] * model_2_probs +
                        weights[2] * model_3_probs +
                        weights[3] * model_4_probs +
                        weights[4] * model_5_probs +
                        weights[5] * model_6_probs
                        ) / np.sum(weights)
final_predictions = np.argmax(final_weighted_probs, axis=1)

# 정확도 평가
accuracy = np.mean(final_predictions == y_true)
print(f'Accuracy with optimized weights: {accuracy:.4f}')

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼돈 행렬 계산
conf_matrix = confusion_matrix(y_true, final_predictions)

# 혼돈 행렬 출력
print("Confusion Matrix:")
print(conf_matrix)

# 혼돈 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
