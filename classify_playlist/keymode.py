import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# CSV 파일 읽기
csv1 = pd.read_csv('favorite_songs.csv')  # 좋아하는 노래 CSV
csv2 = pd.read_csv('disliked_songs.csv')   # 싫어하는 노래 CSV
csv3 = pd.read_csv('soso_songs.csv')   # 싫어하는 노래 CSV


# 키와 모드 결합하기
def combine_key_mode(row):
    key = row[6]  # 6번째 열: key
    mode = row[8]  # 8번째 열: mode
    if mode == 1:  # 1은 장조
        return f"{key} Major"
    else:  # 0은 단조
        return f"{key} Minor"

# 각 CSV 파일에 대해 키와 모드 결합 적용
csv1['Key_Mode'] = csv1.apply(combine_key_mode, axis=1)
csv2['Key_Mode'] = csv2.apply(combine_key_mode, axis=1)
csv3['Key_Mode'] = csv3.apply(combine_key_mode, axis=1)

# 키와 모드의 비율 계산하기
def calculate_key_mode_distribution(df):
    key_mode_counts = df['Key_Mode'].value_counts()
    total_songs = len(df)
    key_mode_percentage = (key_mode_counts / total_songs) * 100
    return key_mode_percentage

# 각 CSV 파일에서 키와 모드 비율 계산
distribution_csv1 = calculate_key_mode_distribution(csv1)
distribution_csv2 = calculate_key_mode_distribution(csv2)
distribution_csv3 = calculate_key_mode_distribution(csv3)

# 결과 출력
print("좋아하는 노래의 키와 모드 비율:")
print(distribution_csv1)

print("\n쏘쏘 노래의 키와 모드 비율:")
print(distribution_csv2)

print("\n싫어하는 노래의 키와 모드 비율:")
print(distribution_csv3)
