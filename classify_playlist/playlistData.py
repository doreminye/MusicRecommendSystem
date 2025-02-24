import spotipy
from spotipy.oauth2 import SpotifyOAuth
import csv
import time

# Spotify Client 정보
CLIENT_ID = 'bbb60dd02ca3480f8da2af6e4d0c4b14'
CLIENT_SECRET = '8bed2c31363740e7b1d4fecee42c14c4'
REDIRECT_URI = 'http://localhost:8080/callback'

# Spotify OAuth 설정
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="playlist-read-private playlist-read-collaborative"
))

# 처리할 플레이리스트 ID 목록
playlist_ids = [
    '37i9dQZF1EIgk9iRvW064r',
    '2ps1BOANiV1uyKwdelBQ0g',
    '37i9dQZF1EIdDyy28MYSyS',
    '37i9dQZF1EQn1VBR3CMMWb'
]

try:
    # CSV 파일 생성 및 헤더 작성
    with open('soso_songs.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = [
            "prefer", "track_id", "artist", "title", "danceability", "energy", "key", "loudness",
            "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "duration_ms", "time_signature", "genre"
        ]
        writer.writerow(header)

        # 각 플레이리스트 처리
        for playlist_id in playlist_ids:
            # Spotify API 요청을 통해 플레이리스트의 모든 트랙 가져오기
            playlist = sp.playlist(playlist_id)
            print(f"플레이리스트 이름: {playlist['name']}")
            print(f"총 트랙 수: {playlist['tracks']['total']}")

            # 모든 트랙을 한 번에 가져오기 위해 페이징 처리
            results = playlist['tracks']
            tracks = results['items']
            while results['next']:
                results = sp.next(results)
                tracks.extend(results['items'])

            # 트랙 정보와 ID를 리스트에 저장
            track_data = []
            track_ids = []

            for item in tracks:
                track = item['track']
                if track and track['id']:  # 트랙 정보와 ID가 있는 경우에만 추가
                    track_data.append({
                        "artist": track['artists'][0]['name'],
                        "title": track['name'],
                        "id": track['id']
                    })
                    track_ids.append(track['id'])

            # API 호출을 줄이기 위해 오디오 피처를 100개씩 요청
            for i in range(0, len(track_ids), 100):
                audio_features = sp.audio_features(track_ids[i:i + 100])

                # 오디오 피처와 트랙 정보를 CSV 파일에 기록
                for j, features in enumerate(audio_features):
                    if features:
                        track_info = track_data[i + j]
                        artist_name = track_info['artist']

                        # 아티스트의 장르 정보 요청
                        artist_info = sp.artist(sp.artist_uri(track_info['id']))
                        genre = ', '.join(artist_info.get('genres', []))  # 장르는 여러 개일 수 있기 때문에 쉼표로 구분

                        row = [
                            1,  # 레이블 정답
                            track_info['id'],  # 트랙 ID
                            track_info['artist'],  # 아티스트
                            track_info['title'],  # 제목
                            features['danceability'],
                            features['energy'],
                            features['key'],
                            features['loudness'],
                            features['mode'],
                            features['speechiness'],
                            features['acousticness'],
                            features['instrumentalness'],
                            features['liveness'],
                            features['valence'],
                            features['tempo'],
                            features['duration_ms'],
                            features['time_signature'],
                            genre  # 아티스트 장르 추가
                        ]
                        writer.writerow(row)

            print(f"{playlist['name']} 플레이리스트의 데이터가 성공적으로 저장되었습니다.")

    print("모든 플레이리스트 데이터를 CSV 파일에 저장 완료!")

except spotipy.exceptions.SpotifyException as e:
    print(f"Spotify API 오류: {e}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
