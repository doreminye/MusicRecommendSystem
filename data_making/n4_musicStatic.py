import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

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
                'artist': track['artists'][0]['name'],
                'title': track['name'],
                'album': track['album']['name'],
                'release_date': track['album']['release_date'],
                'duration_ms': track['duration_ms'],
                'popularity': track['popularity'],
                'genre': ', '.join(artist_info['genres']),
                'tempo': audio_features['tempo'],
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'valence': audio_features['valence'],
                'acousticness': audio_features['acousticness'],
                'instrumentalness': audio_features['instrumentalness'],
                'liveness': audio_features['liveness'],
                'loudness': audio_features['loudness']
            }
    return None


# 속성값들의 평균 계산
def calculate_mean_values(df):
    features_df = df[
        ['tempo', 'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'loudness']]
    return features_df.mean()


# 폴더와 파일 설정
gpx_folder = 'raw_gpx'
prefix = 'sc_hom'

# 모든 파일의 데이터를 저장할 리스트
all_data = []

# 폴더 내의 GPX 파일들에 대해 분석 수행
for file_name in os.listdir(gpx_folder):
    if file_name.startswith(prefix) and file_name.endswith('.gpx'):
        gpx_file_path = os.path.join(gpx_folder, file_name)
        print(f"\nAnalyzing {file_name}...")
        music_list = listening.get_music_list('spotify_tracks_log.txt', gpx_file_path)

        # 데이터 수집 및 DataFrame 생성
        data = []
        for entry in music_list:
            artist = entry['artist']
            title = entry['title']
            details = fetch_song_details(artist, title)
            if details:
                data.append(details)

        df = pd.DataFrame(data)

        # 속성값들의 평균 계산
        mean_values = calculate_mean_values(df)

        # 모든 데이터 리스트에 추가
        all_data.append(mean_values)

# 모든 파일의 평균 값을 계산
if all_data:
    all_mean_values_df = pd.DataFrame(all_data)
    overall_mean_values = all_mean_values_df.mean()

    # 결과 출력
    print("================================")
    print("Average Values of Track Features for All Files")
    print("================================")
    print(overall_mean_values)
else:
    print("No data available for analysis.")
