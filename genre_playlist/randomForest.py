import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# CSV 파일 읽기
csv1 = pd.read_csv('rock2.csv')
csv2 = pd.read_csv('kpop2.csv')
csv3 = pd.read_csv('rap2.csv')
csv4 = pd.read_csv('ballad2.csv')
csv5 = pd.read_csv('classic2.csv')

# 원하는 열의 이름 리스트
selected = ['valence', 'acousticness', 'energy', 'instrumentalness']  # 내가 뽑은 특징들

# 각 CSV에서 X (특징)와 y (레이블) 분리 함수화
def prepare_data(csv):
    X = csv[selected]
    y = csv.iloc[:, 0]
    return X, y

# 데이터 준비
X1, y1 = prepare_data(csv1)
X2, y2 = prepare_data(csv2)
X3, y3 = prepare_data(csv3)
X4, y4 = prepare_data(csv4)
X5, y5 = prepare_data(csv5)

# 로그 변환 함수화
def apply_log_transformation(X):
    return np.log1p(X)

# 로그 변환 적용
X1 = apply_log_transformation(X1)
X2 = apply_log_transformation(X2)
X3 = apply_log_transformation(X3)
X4 = apply_log_transformation(X4)
X5 = apply_log_transformation(X5)

# 데이터 분할
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.2, random_state=42)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, test_size=0.2, random_state=42)

# 학습 및 테스트 데이터 결합
X_train = pd.concat([X_train1, X_train2, X_train3, X_train4, X_train5])
y_train = pd.concat([y_train1, y_train2, y_train3, y_train4, y_train5])
X_test = pd.concat([X_test1, X_test2, X_test3, X_test4, X_test5])
y_test = pd.concat([y_test1, y_test2, y_test3, y_test4, y_test5])

# Random Forest 분류기 초기화
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

# 잘못 분류된 샘플 출력
incorrect_indices = (y_test != y_pred)
incorrect_samples = X_test[incorrect_indices]
incorrect_labels = y_test[incorrect_indices]
predicted_labels = y_pred[incorrect_indices]

print("\n잘못 분류된 샘플:")
for i in range(len(incorrect_samples)):
    print(f"잘못 분류된 샘플 {i+1}:\n특징: {incorrect_samples.iloc[i].values}\n실제 레이블: {incorrect_labels.iloc[i]}\n예측된 레이블: {predicted_labels[i]}\n")
