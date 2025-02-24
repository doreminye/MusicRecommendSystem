import os
import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# CSV 파일에서 위치 데이터를 추출하는 함수
def extract_location_data(csv_file):
    data = pd.read_csv(csv_file)
    # 필요한 열만 선택
    locations = data[['Latitude', 'Longitude', 'Elevation', 'Time']].values
    return locations

# 데이터 전처리 및 정규화
def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# DBSCAN 클러스터링 수행
def dbscan_clustering(data):
    dbscan = DBSCAN(eps=0.25, min_samples=10)
    labels = dbscan.fit_predict(data)
    return labels

# CSV 파일 경로 설정
csv_file_path = 'six_location_data.csv'  # CSV 파일 경로

# 데이터 처리
input_data = extract_location_data(csv_file_path)

# 클러스터링 수행
preprocessed_data = preprocess_data(input_data[:, :4])  # 위도, 경도, 고도, 시간 사용
labels = dbscan_clustering(preprocessed_data)

# 결과를 지도에 시각화
map_center = [np.mean(input_data[:, 0]), np.mean(input_data[:, 1])]
m = folium.Map(location=map_center, zoom_start=13)

# 클러스터 색상 매핑
colors = {
    -1: 'black',  # 이상치
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange',
    4: 'purple',
    5: 'cyan'
}

# 각 클러스터의 포인트 추가
for idx in range(len(input_data)):
    lat = input_data[idx, 0]
    lon = input_data[idx, 1]
    elevation = input_data[idx, 2]
    time_only = input_data[idx, 3]
    folium.CircleMarker(
        location=(lat, lon),
        radius=5,
        color=colors.get(labels[idx], 'black'),  # 클러스터 색상, 이상치는 검은색
        fill=True,
        fill_color=colors.get(labels[idx], 'black'),
        fill_opacity=0.6,
        popup=f'Cluster: {labels[idx]}, Elevation: {elevation:.2f} m, Time: {time_only:.2f} hours'
    ).add_to(m)

# 결과 지도 저장
m.save('DBSCAN.html')
