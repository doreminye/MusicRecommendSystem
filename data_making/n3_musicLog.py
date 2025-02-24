import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Spotify API 인증 설정
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='bbb60dd02ca3480f8da2af6e4d0c4b14',
                                                           client_secret='8bed2c31363740e7b1d4fecee42c14c4'))

# 파일 경로 설정
spotify_log_file = 'spotify_tracks_log.txt'
gpx_file = '../interpol_gpx/sc_hom.gpx'

# 함수 호출
music_list = listening.get_music_list(spotify_log_file, gpx_file)

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
def calculate_mean_values(df):
    # 필요없는 열 제외
    features_df = df[['tempo', 'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'loudness']]
    return features_df.mean()

mean_values = calculate_mean_values(df)

# DataFrame 출력
print("Track Data:")
print(df.to_string(index=False))

print("================================")
print("Average Values of Track Features")
print("================================")
print(mean_values)
