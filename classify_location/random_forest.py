import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# CSV 파일 읽기
data = pd.read_csv('six_location_data.csv')

# 특징(X)와 라벨(y) 분리
X = data[['Latitude', 'Longitude', 'Elevation', 'Time']]
y = data['Label']

# 학습용 데이터와 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 분류기 초기화 및 학습
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = clf.predict(X_test)

# 정확도 및 평가 출력
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(6)]))
