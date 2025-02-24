import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 추가된 임포트
import matplotlib.pyplot as plt

# CSV 파일 읽기
csv1 = pd.read_csv('..\\classify_playlist\\favorite_songs.csv')  # 좋아하는 노래 CSV
csv2 = pd.read_csv('..\\classify_playlist\\disliked_songs.csv')   # 싫어하는 노래 CSV

# 레이블과 특징 벡터 분리
X1 = csv1.iloc[:, 4:]  # 좋아하는 노래에서 특징 벡터 추출 (첫 4개 열 제외)
y1 = csv1.iloc[:, 0]   # 좋아하는 노래의 레이블

X2 = csv2.iloc[:, 4:]  # 싫어하는 노래에서 특징 벡터 추출 (첫 4개 열 제외)
y2 = csv2.iloc[:, 0]   # 싫어하는 노래의 레이블

# 레이블 합치기 (좋아하는 노래: 1, 싫어하는 노래: 0)
X = pd.concat([X1, X2], ignore_index=True)
y = pd.concat([y1.apply(lambda x: 1), y2.apply(lambda x: 0)], ignore_index=True)

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 성능 평가
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 특성 중요도 추출
feature_importances = model.feature_importances_

# 특성 중요도 시각화
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance for Favorite vs Disliked Songs')
plt.show()
