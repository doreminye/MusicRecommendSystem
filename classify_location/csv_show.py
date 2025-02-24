
import pandas as pd
import folium

# CSV 파일 읽기
data = pd.read_csv('six_location_data.csv')

# 지도 생성 (초기 위치: 위도, 경도)
m = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=12)

# 클래스 색상 설정
colors = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange',
    4: 'purple',
    5: 'pink'
}

# 각 점을 지도에 추가
for index, row in data.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=5,
        color=colors[row['Label']],
        fill=True,
        fill_opacity=0.6,
        popup=f"Label: {row['Label']}, Elevation: {row['Elevation']}, Time: {row['Time']}"
    ).add_to(m)

# 지도 저장
m.save('csv_show.html')
