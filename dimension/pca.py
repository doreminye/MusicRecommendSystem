import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# CSV 파일 읽기
csv1 = pd.read_csv('../classify_playlist/favorite_songs.csv')  # 좋아하는 노래 CSV
csv2 = pd.read_csv('../classify_playlist/disliked_songs.csv')   # 싫어하는 노래 CSV
csv3 = pd.read_csv('../classify_playlist/soso_songs.csv')       # 쏘쏘 csv

# 원하는 열의 인덱스 리스트
selected = [6, 8, 9, 10, 11, 12, 13]  # 내가 뽑은 7개의 특징들

# 원하는 열 선택
X1 = csv1.iloc[:, selected]
X2 = csv2.iloc[:, selected]
X3 = csv3.iloc[:, selected]

# X1과 X2를 하나의 데이터프레임으로 결합
df = pd.concat([X1, X2, X3], ignore_index=True)

# PCA 수행
# 3차원으로 축소
pca = PCA(n_components=3)
X_pca = pca.fit_transform(df)

# PCA 결과를 데이터프레임으로 변환
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])

# 레이블 추가 (좋아하는 노래: 2, 그냥 그런 노래: 1, 싫어하는 노래: 0)
labels = [2] * len(X1) + [0] * len(X2) + [1] * len(X1) # 레이블 리스트 생성
pca_df.insert(0, 'label', labels)  # 첫 번째 열에 레이블 추가

# 좋아하는 노래와 싫어하는 노래로 데이터프레임 나누기
favorite_songs_pca = pca_df[pca_df['label'] == 2]
soso_songs_pca = pca_df[pca_df['label'] == 1]
disliked_songs_pca = pca_df[pca_df['label'] == 0]

# 각 데이터프레임을 CSV 파일로 저장
favorite_songs_pca.to_csv('favorite_songs_pca.csv', index=False)
disliked_songs_pca.to_csv('disliked_songs_pca.csv', index=False)
soso_songs_pca.to_csv('soso_songs_pca.csv', index=False)

# 3D PCA 결과 시각화
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'],
                     c=pca_df['label'], cmap='coolwarm', alpha=0.7)

# 축 레이블 및 제목 설정
ax.set_title("PCA")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")

# 레이블 범례 추가
legend1 = ax.legend(*scatter.legend_elements())
plt.show()

# PCA 결과를 CSV 파일로 저장 (이미 위에서 저장됨)
