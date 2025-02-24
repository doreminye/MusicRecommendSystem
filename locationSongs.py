import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv

# Spotify API 인증 설정
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='bbb60dd02ca3480f8da2af6e4d0c4b14',
                                                           client_secret='8bed2c31363740e7b1d4fecee42c14c4'))


# Spotify에서 곡 정보 가져오기
def fetch_song_details(artist, title):
    query = f"{artist} {title}"
    results = sp.search(q=query, type='track', limit=1)
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        track_id = track['id']
        audio_features = sp.audio_features(track_id)[0]
        artist_id = track['artists'][0]['id']
        artist_info = sp.artist(artist_id)
        if audio_features:
            return {
                'track_id': track_id,  # Track ID 추가
                'artist': track['artists'][0]['name'],
                'title': track['name'],
                'album': track['album']['name'],
                'release_date': track['album']['release_date'],
                'duration_ms': track['duration_ms'],  # Duration 추가
                'popularity': track['popularity'],
                'genre': ', '.join(artist_info['genres']),
                **audio_features  # 오디오 피처들을 직접 추가
            }
    return None


# 폴더와 파일 설정
gpx_folder = 'raw_gpx'
prefix = 'sc_hom'
csv_file_path = 'track_features.csv'  # 결과를 저장할 CSV 파일 경로

# CSV 파일 열기 및 헤더 작성
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow([
        'label', 'track_id', 'artist', 'title', 'danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'duration_ms', 'time_signature',
        'timbre_1', 'timbre_2', 'timbre_3', 'timbre_4',
        'timbre_5', 'timbre_6', 'timbre_7', 'timbre_8',
        'timbre_9', 'timbre_10', 'timbre_11', 'timbre_12'
    ])

# 폴더 내의 GPX 파일들에 대해 분석 수행
for file_name in os.listdir(gpx_folder):
    if file_name.startswith(prefix) and file_name.endswith('.gpx'):
        gpx_file_path = os.path.join(gpx_folder, file_name)
        print(f"\nAnalyzing {file_name}...")
        music_list = listening.get_music_list('spotify_tracks_log.txt', gpx_file_path)

        # 데이터 수집 및 CSV 파일에 기록
        for entry in music_list:
            artist = entry['artist']
            title = entry['title']
            details = fetch_song_details(artist, title)
            if details:
                # 오디오 분석 요청
                audio_analysis = sp.audio_analysis(details['track_id'])
                timbre = audio_analysis['segments'][0]['timbre'] if audio_analysis['segments'] else [
                                                                                                        0] * 12  # timbre 값 가져오기

                # 한 줄에 트랙 ID, 가수, 제목, 오디오 피처를 기록
                row = [
                          0,  # 레이블 정답 (예시로 0을 사용, 필요 시 수정 가능)
                          details['track_id'],  # 트랙 ID
                          details['artist'],  # 가수
                          details['title'],  # 제목
                          details['danceability'],
                          details['energy'],
                          details['key'],
                          details['loudness'],
                          details['mode'],
                          details['speechiness'],
                          details['acousticness'],
                          details['instrumentalness'],
                          details['liveness'],
                          details['valence'],
                          details['tempo'],
                          details['duration_ms'],
                          details['time_signature'],
                      ] + timbre  # timbre 벡터 추가

                writer.writerow(row)

# 결과 출력
print(f"\nTrack features have been saved to {csv_file_path}.")
