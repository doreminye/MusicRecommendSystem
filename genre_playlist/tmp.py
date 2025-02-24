import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 읽기
csv1 = pd.read_csv('rock.csv')
csv2 = pd.read_csv('kpop.csv')
csv3 = pd.read_csv('rap.csv')
csv4 = pd.read_csv('ballad.csv')
csv5 = pd.read_csv('classic.csv')

# 원하는 열의 이름 리스트
selected = ['valence', 'acousticness', 'energy', 'instrumentalness']  # 내가 뽑은 특징들

# 각 CSV에서 X (특징)와 y (레이블), artist, title 분리 함수화
def prepare_data(csv):
    X = csv[selected]
    y = csv.iloc[:, 0]  # 첫 번째 열이 레이블
    artists = csv['artist']  # 'artist' 열을 추가
    titles = csv['title']  # 'title' 열을 추가
    return X, y, artists, titles

# 데이터 준비
X1, y1, artists1, titles1 = prepare_data(csv1)
X2, y2, artists2, titles2 = prepare_data(csv2)
X3, y3, artists3, titles3 = prepare_data(csv3)
X4, y4, artists4, titles4 = prepare_data(csv4)
X5, y5, artists5, titles5 = prepare_data(csv5)

# 데이터 분할
X_train1, X_test1, y_train1, y_test1, artists_train1, artists_test1, titles_train1, titles_test1 = train_test_split(X1, y1, artists1, titles1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2, artists_train2, artists_test2, titles_train2, titles_test2 = train_test_split(X2, y2, artists2, titles2, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3, artists_train3, artists_test3, titles_train3, titles_test3 = train_test_split(X3, y3, artists3, titles3, test_size=0.2, random_state=42)
X_train4, X_test4, y_train4, y_test4, artists_train4, artists_test4, titles_train4, titles_test4 = train_test_split(X4, y4, artists4, titles4, test_size=0.2, random_state=42)
X_train5, X_test5, y_train5, y_test5, artists_train5, artists_test5, titles_train5, titles_test5 = train_test_split(X5, y5, artists5, titles5, test_size=0.2, random_state=42)

# 학습 및 테스트 데이터 결합
X_train = pd.concat([X_train1, X_train2, X_train3, X_train4, X_train5])
y_train = pd.concat([y_train1, y_train2, y_train3, y_train4, y_train5])
X_test = pd.concat([X_test1, X_test2, X_test3, X_test4, X_test5])
y_test = pd.concat([y_test1, y_test2, y_test3, y_test4, y_test5])
artists_test = pd.concat([artists_test1, artists_test2, artists_test3, artists_test4, artists_test5])
titles_test = pd.concat([titles_test1, titles_test2, titles_test3, titles_test4, titles_test5])

# t-SNE 적용: 학습 데이터와 테스트 데이터를 합친 후 적용
combined_X = pd.concat([X_train, X_test])

# t-SNE 모델 학습
tsne = TSNE(n_components=2, random_state=42)
X_combined_tsne = tsne.fit_transform(combined_X)

# 학습 데이터와 테스트 데이터를 나누어 다시 분리
X_train_tsne = X_combined_tsne[:len(X_train)]
X_test_tsne = X_combined_tsne[len(X_train):]

# Random Forest 분류기 초기화
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
model.fit(X_train_tsne, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test_tsne)

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
incorrect_artists = artists_test[incorrect_indices]
incorrect_titles = titles_test[incorrect_indices]
incorrect_features = incorrect_samples[selected]

print("\n잘못 분류된 샘플:")
for i in range(len(incorrect_samples)):
    print(f"잘못 분류된 샘플 {i+1}:")
    print(f"곡: {incorrect_artists.iloc[i]}, {incorrect_titles.iloc[i]}")
    print(f"실제 레이블: {incorrect_labels.iloc[i]}")
    print(f"예측된 레이블: {predicted_labels[i]}")
    print(f"특징: {incorrect_features.iloc[i].to_frame().T}")
    print("-" * 50)

import matplotlib.patches as mpatches
# 고유 클래스 (1~5)
unique_labels = sorted(y_test.unique())

# 무지개 색상 컬러맵 생성
cmap = plt.cm.get_cmap("tab10", len(unique_labels))
color_map = {label: cmap((label - 1) / len(unique_labels)) for label in unique_labels}  # 1~5에 맞게 매핑

# 시각화
plt.figure(figsize=(10, 8))

# 샘플별로 중심과 테두리 색상 지정
for i in range(len(X_test_tsne)):
    face_color = color_map[y_pred[i]]  # 예측된 클래스의 색상
    edge_color = color_map[y_test.iloc[i]] if y_test.iloc[i] != y_pred[i] else face_color  # 잘못 분류된 경우만 테두리 색 변경

    plt.scatter(
        X_test_tsne[i, 0], X_test_tsne[i, 1],
        facecolor=face_color,
        edgecolor=edge_color,
        alpha=0.8, s=100, linewidths=1.5
    )

# 범례 추가
legend_patches = [mpatches.Patch(color=color_map[label], label=f"Class {label}") for label in unique_labels]
plt.legend(handles=legend_patches, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# 그래프 제목 및 레이블
plt.title("t-SNE Visualization (Rainbow Colors, Misclassified Borders)", fontsize=16)
plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)
plt.tight_layout()

# 그래프 표시
plt.show()
