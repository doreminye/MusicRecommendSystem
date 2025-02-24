import folium
from xml.etree import ElementTree as ET

# GPX 파일 경로
gpx_file_path = 'raw_gpx/hom_sc1.gpx'  # 실제 파일 경로로 변경하세요.

# GPX 파일을 파싱
tree = ET.parse(gpx_file_path)
root = tree.getroot()

# 네임스페이스 등록
ns = {'default': "http://www.topografix.com/GPX/1/1"}

# 트랙포인트(trkpt) 요소들을 순회하며 위도, 경도, 시간을 리스트에 저장
trackpoints = []
for trkpt in root.findall('.//default:trkpt', ns):
    lat = float(trkpt.get('lat'))
    lon = float(trkpt.get('lon'))

    # 시간 정보 추출
    time_elem = trkpt.find('default:time', ns)
    time_str = time_elem.text if time_elem is not None else "No Time Info"

    trackpoints.append((lat, lon, time_str))

# Folium 지도 생성
# 첫 번째 트랙포인트를 지도 중심으로 설정
map_center = trackpoints[0][:2] if trackpoints else (0, 0)
mymap = folium.Map(location=map_center, zoom_start=15)

# 트랙포인트들을 지도에 빨간 점으로 표시하고 클릭 시 좌표와 시간 정보를 팝업으로 보여줌
for lat, lon, time_str in trackpoints:
    folium.CircleMarker(
        location=(lat, lon),
        radius=3,
        color='red',
        fill=True,
        fill_color='red',
        popup=f"Latitude: {lat}, Longitude: {lon}<br>Time: {time_str}"  # 클릭 시 좌표와 시간 팝업 표시
    ).add_to(mymap)

# 지도를 HTML 파일로 저장
map_file_path = 'hom_sc1.html'  # 저장할 파일 경로
mymap.save(map_file_path)

print(f"지도 HTML 파일이 생성되었습니다: {map_file_path}")