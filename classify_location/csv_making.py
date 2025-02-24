import gpxpy
import pandas as pd

def parse_gpx_and_label(gpx_file, label):
    with open(gpx_file, 'r') as file:
        gpx = gpxpy.parse(file)

    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                time_seconds = point.time.hour * 3600 + point.time.minute * 60 + point.time.second
                data.append([point.latitude, point.longitude, point.elevation, time_seconds, label])
    return data

# 예시: 6개의 GPX 파일을 라벨링하여 하나의 데이터프레임으로 저장
gpx_files = ['interpol_gpx/dr_hom.gpx', 'interpol_gpx/hom_dr.gpx',
             'interpol_gpx/sc_hom.gpx', 'interpol_gpx/hom_sc.gpx',
             'interpol_gpx/wk_hom.gpx', 'interpol_gpx/hom_wk.gpx']
labels = [0, 1, 2, 3, 4, 5]  # 각 파일에 대한 라벨

all_data = []
for gpx_file, label in zip(gpx_files, labels):
    all_data.extend(parse_gpx_and_label(gpx_file, label))

# 데이터프레임으로 변환하고 CSV 파일로 저장
df = pd.DataFrame(all_data, columns=['Latitude', 'Longitude', 'Elevation', 'Time', 'Label'])
df.to_csv('six_location_data.csv', index=False)
