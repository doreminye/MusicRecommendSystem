import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


# X1, X2, X3 데이터프레임에서 10번째와 5번째 열에만 로그 변환 적용
X1 = X1.copy()  # 원본 데이터프레임을 복사
X2 = X2.copy()  # 원본 데이터프레임을 복사
X3 = X3.copy()  # 원본 데이터프레임을 복사

X1.iloc[:, [0, 1, 2,3]] = np.log1p(X1.iloc[:, [0, 1, 2,3]])
X2.iloc[:, [0, 1, 2,3]] = np.log1p(X2.iloc[:, [0, 1, 2,3]])
X3.iloc[:, [0, 1, 2,3]] = np.log1p(X3.iloc[:, [0, 1, 2,3]])


# 데이터 분할
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)

# 학습 및 테스트 데이터 결합
X_train = pd.concat([X_train1, X_train2, X_train3])
y_train = pd.concat([y_train1, y_train2, y_train3])
X_test = pd.concat([X_test1, X_test2, X_test3])
y_test = pd.concat([y_test1, y_test2, y_test3])


#Random Forest 분류기 초기화
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 정확도: {accuracy:.2f}")

# 추가 평가 지표 출력
print("분류 리포트:")
print(classification_report(y_test, y_pred))

# 혼동 행렬 출력
confusion = confusion_matrix(y_test, y_pred)
print("혼동 행렬:")
print(confusion)

# 잘못 분류된 샘플을 출력
incorrect_indices = (y_test != y_pred)
incorrect_samples = X_test[incorrect_indices]
incorrect_labels = y_test[incorrect_indices]
predicted_labels = y_pred[incorrect_indices]
