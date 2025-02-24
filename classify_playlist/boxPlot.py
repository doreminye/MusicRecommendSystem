import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 읽기
csv1 = pd.read_csv('favorite_songs.csv')  # 좋아하는 노래 CSV
csv2 = pd.read_csv('soso_songs.csv')
csv3 = pd.read_csv('disliked_songs.csv')  # 싫어하는 노래

# 원하는 열의 이름 리스트
selected = ['valence', 'acousticness', 'energy', 'instrumentalness']  # 특징들

# 원하는 열 선택
X1 = csv1[selected].copy()  # .copy()를 사용하여 원본 데이터를 안전하게 복사
X2 = csv2[selected].copy()
X3 = csv3[selected].copy()

# 좋아하는 노래와 싫어하는 노래 데이터프레임에 그룹 레이블 추가
X1.loc[:, 'Group'] = 'Favorite'
X2.loc[:, 'Group'] = 'Soso'
X3.loc[:, 'Group'] = 'Disliked'

# 두 그룹을 합친다
df = pd.concat([X1, X2, X3])

# 로그 변환 적용 (log1p는 0에 대해 안전하게 적용)
df[selected] = np.log10(df[selected])

# 데이터를 긴 형태로 변환 (melt)
df_melt = pd.melt(df, id_vars=['Group'], var_name='Feature', value_name='Value')

# 개별 특징별로 박스플롯과 점을 함께 그리기
features = df_melt['Feature'].unique()

for feature in features:
    plt.figure(figsize=(8, 6))

    # 박스플롯
    sns.boxplot(x='Group', y='Value', data=df_melt[df_melt['Feature'] == feature],
                palette="Set2", showfliers=False)  # 이상치는 제외

    # 점 표시
    sns.stripplot(x='Group', y='Value', data=df_melt[df_melt['Feature'] == feature],
                  color='black', alpha=0.6, jitter=True)  # 점을 약간 퍼지게

    plt.title(f'Comparison of Log-Transformed {feature} between Favorite and Disliked Songs')
    plt.tight_layout()
    plt.show()
