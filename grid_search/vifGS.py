from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# CSV 파일 읽기
csv1 = pd.read_csv('../classify_playlist/favorite_songs.csv')  # 좋아하는 노래 CSV
csv2 = pd.read_csv('../classify_playlist/soso_songs.csv')   # 싫어하는 노래 CSV
csv3 = pd.read_csv('../classify_playlist/disliked_songs.csv')   # 싫어하는 노래 CSV
# 원하는 열의 인덱스 리스트
selected = [5, 10, 11, 13]  # 내가 뽑은 특징들

# 레이블과 특징 벡터 분리
X1 = csv1.iloc[:, selected]  # 좋아하는 노래에서 특징 벡터 추출 (첫 4개 열 제외)
y1 = csv1.iloc[:, 0]   # 좋아하는 노래의 레이블

X2 = csv2.iloc[:, selected]
y2 = csv2.iloc[:, 0]

X3 = csv2.iloc[:, selected]  # 싫어하는 노래에서 특징 벡터 추출 (첫 4개 열 제외)
y3 = csv2.iloc[:, 0]   # 싫어하는 노래의 레이블

# 각 데이터셋을 80% 학습, 30% 테스트로 분할
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# 학습 데이터와 테스트 데이터 결합
X_train = pd.concat([X_train1, X_train2])  # 학습 데이터 결합
y_train = pd.concat([y_train1, y_train2])  # 레이블 결합

X_test = pd.concat([X_test1, X_test2])      # 테스트 데이터 결합
y_test = pd.concat([y_test1, y_test2])      # 레이블 결합

# 하이퍼파라미터 그리드 설정
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],                # 이웃의 수
    'weights': ['uniform', 'distance'],              # 가중치 설정
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # 알고리즘
    'leaf_size': [20, 30, 40],                      # leaf_size (트리 알고리즘을 사용할 경우)
    'p': [1, 2]                                     # 거리 계산 방식 (맨하탄 거리, 유클리드 거리)
}

# KNeighborsClassifier 모델 초기화
knn_model = KNeighborsClassifier()

# GridSearchCV 설정
knn_grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid_knn,
    cv=5,                                  # 5-폴드 교차 검증
    scoring='accuracy',                    # 평가 척도
    verbose=1,                             # 출력 단계 조절
    n_jobs=-1                              # 모든 CPU 코어 사용
)

# 학습 및 최적 하이퍼파라미터 탐색
knn_grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 및 최고 성능 확인
print("최적 하이퍼파라미터: ", knn_grid_search.best_params_)
print("최고 성능(Accuracy): ", knn_grid_search.best_score_)


"""
랜덤 포레스트 결과
RandomForestClassifier(max_depth=8, max_features='sqrt', min_samples_leaf=2, min_samples_split=7, n_estimators=100)
최고 성능(Accuracy):  0.6833333333333333

SVC 결과
SVC(probability=True, C= 100, class_weight= 'balanced', degree= 4, gamma= 'scale', kernel= 'poly')
SVC 최고 성능(Accuracy):  0.7

Decision model 결과
DecisionTreeClassifier(criterion= 'gini', max_depth=7, max_features='sqrt', min_samples_leaf= 1, min_samples_split=7, splitter= 'random')
DecisionTree 최고 성능(Accuracy):  0.6733333333333333

로지스틱 회귀
LogisticRegression(C=10, class_weight=None, max_iter=200, penalty='l2', solver='newton-cg')
LogisticRegression 최고 성능(Accuracy):  0.67

knn_grid_search
KNeighborsClassifier(algorithm='auto', leaf_size= 20, n_neighbors=11, p=2, weights='distance')
최고 성능(Accuracy):  0.7066666666666667
"""