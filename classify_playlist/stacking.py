from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

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

# 기본 모델 정의
base_learners = [
    ('svc', SVC(probability=True, C=100, class_weight='balanced', degree=4, gamma='scale', kernel='poly')),
    ('dt', DecisionTreeClassifier(criterion='gini', max_depth=7, max_features='sqrt', min_samples_leaf=1, min_samples_split=7, splitter='random')),
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier(max_depth=8, max_features='sqrt', min_samples_leaf=2, min_samples_split=7, n_estimators=100)),
    ('nb', GaussianNB(var_smoothing=1e-9)),
    ('knn', KNeighborsClassifier(algorithm='auto', leaf_size=20, n_neighbors=11, p=2, weights='distance'))
]

# 최종 모델로 Logistic Regression 사용
final_model = LogisticRegression(max_iter=200)

# 스태킹 모델 정의
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=final_model)

# 모델 훈련
stacking_model.fit(X_train, y_train)

# 예측 확률
stacking_model_probs = stacking_model.predict_proba(X_test)

# 실제 라벨
y_true = y_test

# Cross-entropy loss 계산
loss = log_loss(y_true, stacking_model_probs)
print(f'Log Loss: {loss:.4f}')

# 예측 수행
final_predictions = np.argmax(stacking_model_probs, axis=1)

# 정확도 평가
accuracy = np.mean(final_predictions == y_true)
print(f'Accuracy: {accuracy:.4f}')

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
