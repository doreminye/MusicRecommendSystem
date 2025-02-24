import os
import gpxpy
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium

# GPX 파일에서 위치 및 시간 데이터를 추출하는 함수
def extract_location_data(gpx_file):
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)

    locations = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time:
                    timestamp = time.mktime(point.time.timetuple())
                    locations.append([point.latitude, point.longitude, point.elevation, timestamp])
    return np.array(locations)

# 거리 계산 (Haversine 공식을 사용)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (단위: km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 1000  # 거리 (단위: meters)

# 속도 계산 (시간 차이 가중치 추가)
def calculate_weighted_speed(data):
    speeds = []
    for i in range(1, len(data)):
        lat1, lon1 = data[i - 1][:2]
        time1 = data[i - 1][3]
        lat2, lon2 = data[i][:2]
        time2 = data[i][3]
        distance = haversine(lat1, lon1, lat2, lon2)
        time_diff = time2 - time1

        if time_diff > 0:
            speed = distance / time_diff
            weight = 1 / time_diff
            speeds.append(speed * weight)
        else:
            speeds.append(0)
    return np.array(speeds)

# 위치와 속도를 결합한 데이터를 생성하는 함수
def location_speed_with_weighted_time(input_gpx):
    location_vec = extract_location_data(input_gpx)
    weighted_speed_vec = calculate_weighted_speed(location_vec)
    weighted_speed_vec = np.insert(weighted_speed_vec, 0, 0)  # 첫 번째 데이터 속도를 0으로 설정
    return np.column_stack((location_vec, weighted_speed_vec))

# 데이터 전처리 및 정규화
def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

# K-Means 클러스터링 수행
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

# GPX 파일 경로 설정
gpx_file_path = '../data_making/merged.gpx'  # GPX 파일 경로

# 데이터 처리
input_data = location_speed_with_weighted_time(gpx_file_path)
preprocessed_data = preprocess_data(input_data[:, :4])  # 위도, 경도, 고도, 속도 모두 사용

# K-Means 클러스터링 수행
n_clusters = 3  # 원하는 클러스터 수
labels, centers = kmeans_clustering(preprocessed_data, n_clusters)

# 클러스터링 결과를 지도에 시각화
map_center = [np.mean(input_data[:, 0]), np.mean(input_data[:, 1])]
m = folium.Map(location=map_center, zoom_start=13)

# 클러스터 색상 매핑
colors = {
    0: 'red',
    1: 'blue',
    2: 'green',
#    3: 'orange',
#    4: 'purple',
#    5: 'cyan'
}

# 각 클러스터의 포인트 추가
for idx in range(len(input_data)):
    lat = input_data[idx, 0]
    lon = input_data[idx, 1]
    elevation = input_data[idx, 2]
    folium.CircleMarker(
        location=(lat, lon),
        radius=5,
        color=colors[labels[idx]],  # 클러스터 색상
        fill=True,
        fill_color=colors[labels[idx]],
        fill_opacity=0.6,
        popup=f'Cluster: {labels[idx]}, Elevation: {elevation:.2f} m'
    ).add_to(m)

# 클러스터 중심점 추가
for center in centers:
    folium.Marker(
        location=(center[0], center[1]),
        icon=folium.Icon(color='red', icon='info-sign'),
        popup='Centroid'
    ).add_to(m)

# 결과 지도 저장
m.save('K-means.html')
