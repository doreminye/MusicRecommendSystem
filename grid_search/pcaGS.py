
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

# CSV 파일 읽기
csv1 = pd.read_csv('../dimension/favorite_songs_pca.csv')  # 좋아하는 노래 CSV
csv2 = pd.read_csv('../dimension/disliked_songs_pca.csv')   # 싫어하는 노래 CSV

# 레이블과 특징 벡터 분리
X1 = csv1.iloc[:, 1:4]  # 좋아하는 노래에서 특징 벡터 추출 (PCA1, PCA2, PCA3)
y1 = csv1.iloc[:, 0]    # 좋아하는 노래의 레이블 (첫 번째 열)

X2 = csv2.iloc[:, 1:4]  # 싫어하는 노래에서 특징 벡터 추출 (PCA1, PCA2, PCA3)
y2 = csv2.iloc[:, 0]    # 싫어하는 노래의 레이블 (첫 번째 열)

# 각 데이터셋을 70% 학습, 30% 테스트로 분할
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# 학습 데이터와 테스트 데이터 결합
X_train = pd.concat([X_train1, X_train2])  # 학습 데이터 결합
y_train = pd.concat([y_train1, y_train2])  # 레이블 결합

X_test = pd.concat([X_test1, X_test2])      # 테스트 데이터 결합
y_test = pd.concat([y_test1, y_test2])      # 레이블 결합

# Random Forest 분류기 초기화
model = RandomForestClassifier(random_state=42)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [1,2,3,4,5,6,7],       # 트리 개수
    'max_depth': [None, 5,10,15,20],       # 트리의 최대 깊이
    'max_features': ['sqrt', 'log2'],      # 분기 시 고려할 피처 수
    'min_samples_split': [2, 5, 10],       # 분할을 위한 최소 샘플 수
    'min_samples_leaf': [1, 2, 4]          # 리프 노드의 최소 샘플 수
}
# 랜덤 포레스트 모델 초기화
rf_model = RandomForestClassifier(random_state=42)

# GridSearchCV 설정
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,                                  # 5-폴드 교차 검증
    scoring='accuracy',                    # 평가 척도
    verbose=2,                             # 출력 단계 조절
    n_jobs=-1                              # 모든 CPU 코어 사용
)

# 학습 및 최적 하이퍼파라미터 탐색
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 및 최고 성능 확인
print("최적 하이퍼파라미터: ", grid_search.best_params_)
print("최고 성능(Accuracy): ", grid_search.best_score_)