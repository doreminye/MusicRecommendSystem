import xml.etree.ElementTree as ET
from datetime import datetime


# Spotify 로그 파싱
def parse_spotify_log(file_path):
    songs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split(' - ')
            if len(parts) < 5:
                continue
            start_time = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(parts[1], '%Y-%m-%d %H:%M:%S')
            artist = parts[2]
            title = parts[3].strip()
            duration_str = parts[4].strip()

            # 재생 시간 파싱
            minutes, seconds = 0, 0
            if '시간' in duration_str:
                time_parts = duration_str.split('시간')
                minutes = int(time_parts[0].strip())
                remainder = time_parts[1].strip()
                if '분' in remainder:
                    min_parts = remainder.split('분')
                    minutes += int(min_parts[0].strip())
                    if '초' in min_parts[1]:
                        seconds = int(min_parts[1].replace('초', '').strip())
                else:
                    if '초' in remainder:
                        seconds = int(remainder.replace('초', '').strip())
            elif '분' in duration_str:
                time_parts = duration_str.split('분')
                minutes = int(time_parts[0].strip())
                if '초' in time_parts[1]:
                    seconds = int(time_parts[1].replace('초', '').strip())
            else:
                if '초' in duration_str:
                    seconds = int(duration_str.replace('초', '').strip())

            duration = minutes * 60 + seconds
            # 30초 미만인 곡은 제외
            if duration >= 30:
                songs.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'artist': artist,
                    'title': title,
                    'duration': duration
                })
    return songs


# GPX 파일 파싱
def parse_gpx(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    times = []
    for trkpt in root.findall('.//gpx:trkpt', ns):
        time_str = trkpt.find('gpx:time', ns).text
        time = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ')
        times.append(time)
    return min(times), max(times)


# 시간 범위에 따른 노래 찾기
def find_songs_in_time_range(start_time, end_time, songs):
    song_dict = {}
    for song in songs:
        if song['end_time'] >= start_time and song['start_time'] <= end_time:
            key = (song['artist'], song['title'])  # (artist, title)로 중복 체크
            if key not in song_dict:
                song_dict[key] = {
                    'artist': song['artist'],
                    'title': song['title'],
                    'total_duration': song['duration']
                }
            else:
                # 총 재생 시간 합산
                song_dict[key]['total_duration'] += song['duration']

    # 결과를 리스트로 변환
    songs_in_range = []
    for song in song_dict.values():
        minutes, seconds = divmod(song['total_duration'], 60)
        hours, minutes = divmod(minutes, 60)
        duration_str = f"{hours}시간 {minutes}분 {seconds}초"
        songs_in_range.append({
            'artist': song['artist'],
            'title': song['title'],
            'total_duration': duration_str
        })

    return songs_in_range


# 결과를 반환하는 함수
def get_music_list(spotify_log_file, gpx_file):
    # 데이터 파싱
    spotify_songs = parse_spotify_log(spotify_log_file)
    gpx_start_time, gpx_end_time = parse_gpx(gpx_file)

    # 노래 찾기
    songs_played = find_songs_in_time_range(gpx_start_time, gpx_end_time, spotify_songs)

    return songs_played


# 이 파일을 직접 실행할 때만 아래 코드가 동작
if __name__ == '__main__':
    # 테스트용 파일 경로 설정
    spotify_log_file = 'spotify_tracks_log.txt'
    gpx_file = 'interpol_gpx/hom_sc2.gpx'
    # main 함수 호출
    songs_played = get_music_list(spotify_log_file, gpx_file)

    # 결과 출력
    if not songs_played:
        print("GPX 시간 범위 내에서 재생된 노래가 없습니다.")
    else:
        # 최대 길이에 맞춰 포맷 조정
        artist_width = max(len(song['artist']) for song in songs_played) + 2
        title_width = max(len(song['title']) for song in songs_played) + 2
        duration_width = max(len(song['total_duration']) for song in songs_played) + 2

        # 헤더 출력
        print(f"{'아티스트'.ljust(artist_width)}{'제목'.ljust(title_width)}{'총 재생 시간'}")
        print("=" * (artist_width + title_width + duration_width))

        # 데이터 출력
        for entry in songs_played:
            print(f"{entry['artist'].ljust(artist_width)}{entry['title'].ljust(title_width)}{entry['total_duration']}")
