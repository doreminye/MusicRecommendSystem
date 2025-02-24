import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# CSV 파일 읽기
csv1 = pd.read_csv('../classify_playlist/favorite_songs.csv')  # 좋아하는 노래 CSV
csv2 = pd.read_csv('../classify_playlist/disliked_songs.csv')   # 싫어하는 노래 CSV
csv3 = pd.read_csv('../classify_playlist/soso_songs.csv')   # 싫어하는 노래 CSV

# 원하는 열의 인덱스 리스트

desired_columns = [5, 10, 11, 13]

# 원하는 열 선택
X1 = csv1.iloc[:, desired_columns]
X2 = csv2.iloc[:, desired_columns]
X3 = csv2.iloc[:, desired_columns]

# X1과 X2를 하나의 데이터프레임으로 결합
df = pd.concat([X1, X2, X3], ignore_index=True)

# VIF 계산
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

# VIF 값을 출력
print(vif_data)

# 상관행렬 계산
correlation_matrix = df.corr()

# 상관행렬 히트맵으로 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title("특징 변수간 상관 행렬")
plt.show()
