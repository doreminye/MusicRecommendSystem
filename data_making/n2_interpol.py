import xml.etree.ElementTree as ET
import numpy as np
import folium


def load_gpx(file_path):
    # GPX 파일 읽기
    tree = ET.parse(file_path)
    root = tree.getroot()

    # GPX 네임스페이스
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

    points = []
    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        ele = float(trkpt.find('gpx:ele', ns).text)
        time = trkpt.find('gpx:time', ns).text
        points.append((lat, lon, ele, time))

    return points


def save_gpx(file_path, points):
    # 보간된 데이터를 새로운 GPX 파일로 저장
    root = ET.Element('gpx', xmlns='http://www.topografix.com/GPX/1/1', version='1.1', creator='Python')
    trk = ET.SubElement(root, 'trk')
    trkseg = ET.SubElement(trk, 'trkseg')

    for lat, lon, ele, time in points:
        trkpt = ET.SubElement(trkseg, 'trkpt', lat=str(lat), lon=str(lon))
        ET.SubElement(trkpt, 'ele').text = str(ele)
        ET.SubElement(trkpt, 'time').text = time

    tree = ET.ElementTree(root)
    tree.write(file_path, xml_declaration=True, encoding='utf-8')


def moving_average(data, window_size):
    # 이동 평균 필터 적용
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def interpolate_gpx(input_file, output_file, html_file, window_size=5):
    # GPX 파일 로드
    points = load_gpx(input_file)

    # 데이터 추출
    latitudes = np.array([p[0] for p in points])
    longitudes = np.array([p[1] for p in points])
    elevations = np.array([p[2] for p in points])
    times = [p[3] for p in points]

    # 이동 평균 필터 적용
    lat_smoothed = moving_average(latitudes, window_size)
    lon_smoothed = moving_average(longitudes, window_size)
    ele_smoothed = moving_average(elevations, window_size)

    # 보간된 데이터 포인트 생성
    interpolated_points = []
    for i in range(len(lat_smoothed)):
        interpolated_points.append((lat_smoothed[i], lon_smoothed[i], ele_smoothed[i], times[i]))

    # 보간된 데이터를 새로운 GPX 파일로 저장
    save_gpx(output_file, interpolated_points)

    # HTML 지도 생성
    map_center = [latitudes.mean(), longitudes.mean()]
    m = folium.Map(location=map_center, zoom_start=15)

    # 원본 데이터 포인트 추가
    for lat, lon, ele, time in points:
        folium.CircleMarker([lat, lon], color='red', radius=4, fill=True, fill_color='red').add_to(m)

    # 보간된 데이터 포인트 추가
    for lat, lon, ele, time in interpolated_points:
        folium.CircleMarker([lat, lon], color='blue', radius=4, fill=True, fill_color='blue').add_to(m)

    # HTML 파일로 저장
    m.save(html_file)


# 사용 예제
interpolate_gpx('../raw_gpx/sc_hom3.gpx', 'interpol_gpx/sc_hom.gpx', 'interpol_html/sc_hom2.html', window_size=30)
