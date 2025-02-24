import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CSV 파일 읽기
csv1 = pd.read_csv('../dimension/favorite_songs_pca.csv')  # 좋아하는 노래 CSV
csv2 = pd.read_csv('../dimension/disliked_songs_pca.csv')   # 싫어하는 노래 CSV
csv3 = pd.read_csv('../dimension/soso_songs.csv')

# 레이블과 특징 벡터 분리
X1 = csv1.iloc[:, 1:4]  # 좋아하는 노래에서 특징 벡터 추출 (PCA1, PCA2, PCA3)
y1 = csv1.iloc[:, 0]    # 좋아하는 노래의 레이블 (첫 번째 열)

X2 = csv2.iloc[:, 1:4]  # 싫어하는 노래에서 특징 벡터 추출 (PCA1, PCA2, PCA3)
y2 = csv2.iloc[:, 0]    # 싫어하는 노래의 레이블 (첫 번째 열)

X3 = csv3.iloc[:, 1:4]  # 싫어하는 노래에서 특징 벡터 추출 (PCA1, PCA2, PCA3)
y3 = csv3.iloc[:, 0]    # 싫어하는 노래의 레이블 (첫 번째 열)

# 각 데이터셋을 70% 학습, 30% 테스트로 분할
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# 학습 데이터와 테스트 데이터 결합
X_train = pd.concat([X_train1, X_train2, X_train3])  # 학습 데이터 결합
y_train = pd.concat([y_train1, y_train2, y_train3])  # 레이블 결합

X_test = pd.concat([X_test1, X_test2, X_test3])      # 테스트 데이터 결합
y_test = pd.concat([y_test1, y_test2, y_test3])      # 레이블 결합

# Random Forest 분류기 초기화
model = RandomForestClassifier(n_estimators=5000, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# feature importance 추출
print("피쳐 중요도:\n\t{0}".format(np.round(model.feature_importances_, 2)))

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
