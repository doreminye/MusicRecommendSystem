import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CSV 파일 읽기
csv1 = pd.read_csv('favorite_songs.csv')  # 좋아하는 노래 CSV
csv2 = pd.read_csv('soso_songs.csv')   # 쏘쏘 노래 CSV
csv3 = pd.read_csv('disliked_songs.csv')   # 싫어하는 노래 CSV
# 원하는 열의 인덱스 리스트
selected = [10,11,13, 5]  # 내가 뽑은 특징들

# 레이블과 특징 벡터 분리
X1 = csv1.iloc[:, selected]  # 좋아하는 노래에서 특징 벡터 추출 (첫 4개 열 제외)
y1 = csv1.iloc[:, 0]   # 좋아하는 노래의 레이블

X2 = csv2.iloc[:, selected]  # 싫어하는 노래에서 특징 벡터 추출 (첫 4개 열 제외)
y2 = csv2.iloc[:, 0]   # 싫어하는 노래의 레이블

X3 = csv3.iloc[:, selected]  # 싫어하는 노래에서 특징 벡터 추출 (첫 4개 열 제외)
y3 = csv3.iloc[:, 0]   # 싫어하는 노래의 레이블


# 각 데이터셋을 80% 학습, 30% 테스트로 분할
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)

# 학습 데이터와 테스트 데이터 결합
X_train = pd.concat([X_train1, X_train2, X_train3])  # 학습 데이터 결합
y_train = pd.concat([y_train1, y_train2, y_train3])  # 레이블 결합

X_test = pd.concat([X_test1, X_test2, X_test3])      # 테스트 데이터 결합
y_test = pd.concat([y_test1, y_test2, y_test3])      # 레이블 결합

# Random Forest 분류기 초기화
model = RandomForestClassifier(n_estimators=100, random_state=42)

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

####잘못 분류된 샘플을 출력###
import matplotlib.pyplot as plt
# X_test: 테스트 데이터셋, y_test: 실제 레이블, y_pred: 예측된 레이블
incorrect_indices = (y_test != y_pred)

incorrect_samples = X_test[incorrect_indices]
incorrect_labels = y_test[incorrect_indices]
predicted_labels = y_pred[incorrect_indices]

# 잘못된 예측을 DataFrame으로 만듭니다.
wrong_predictions = pd.DataFrame({
    'Actual': incorrect_labels,
    'Predicted': predicted_labels,
    'Features': list(incorrect_samples.values)
})

print(wrong_predictions)

# 특징 중요도 추출
feature_importances = model.feature_importances_

# 특징 이름
features = X_train.columns

# 중요도를 정렬합니다.
indices = np.argsort(feature_importances)[::-1]

# 시각화
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()