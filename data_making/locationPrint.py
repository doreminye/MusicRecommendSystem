import gpxpy
import gpxpy.gpx


def parse_gpx(file_path):
    # GPX 파일을 읽어들입니다.
    with open(file_path, 'r') as file:
        gpx = gpxpy.parse(file)

    # GPX 파일에서 모든 트랙을 가져옵니다.
    for track in gpx.tracks:
        print(f"Track Name: {track.name}")
        for segment in track.segments:
            for point in segment.points:
                # 시간대와 위치 정보를 출력합니다.
                print(f"Time: {point.time}, Latitude: {point.latitude}, Longitude: {point.longitude}, Elevation: {point.elevation}")


if __name__ == "__main__":
    # GPX 파일 경로를 지정합니다.
    file_path = 'merged.gpx'
    parse_gpx(file_path)

