import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import datetime

# Spotify API 설정
SPOTIPY_CLIENT_ID = 'bbb60dd02ca3480f8da2af6e4d0c4b14'
SPOTIPY_CLIENT_SECRET = '8bed2c31363740e7b1d4fecee42c14c4'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

scope = "user-read-playback-state user-read-currently-playing"

# Spotipy 클라이언트 생성
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope=scope))


def get_current_track():
    """현재 재생 중인 트랙 정보"""
    try:
        current_track = sp.current_playback()
        if current_track and current_track['is_playing']:
            track = current_track['item']
            artist_name = track['artists'][0]['name']
            track_name = track['name']
            return {
                'artist_name': artist_name,
                'track_name': track_name,
                'progress_ms': current_track['progress_ms'],
                'timestamp': current_track['timestamp']
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching current track: {e}")
        return None


def log_track(start_time, end_time, track_info, play_duration):
    """트랙의 재생 정보를 로그 파일에 기록합니다."""
    start_time_str = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    with open("spotify_tracks_log.txt", "a", encoding="utf-8") as file:
        file.write(
            f"{start_time_str} - {end_time_str} - {track_info['artist_name']} - {track_info['track_name']} - {play_duration}\n")
    print(
        f"Logged: {start_time_str} - {end_time_str} - {track_info['artist_name']} - {track_info['track_name']} - {play_duration}")


def format_time(seconds):
    """초를 입력받아 '시간 분 초' 형식으로 변환합니다."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}시간 {int(minutes)}분 {int(seconds)}초"


if __name__ == "__main__":
    last_track_id = None
    start_time = None

    while True:
        current_track_info = get_current_track()

        if current_track_info:
            current_track_id = f"{current_track_info['artist_name']} - {current_track_info['track_name']}"

            # 트랙이 바뀌었는지 확인
            if current_track_id != last_track_id:
                # 이전 트랙의 재생 시간 계산 및 기록
                if last_track_id and start_time:
                    end_time = time.time()
                    play_duration = end_time - start_time
                    formatted_play_time = format_time(play_duration)
                    log_track(start_time, end_time, previous_track_info, formatted_play_time)

                # 새로운 트랙 시작 시간 기록
                start_time = time.time()
                previous_track_info = current_track_info
                last_track_id = current_track_id

        else:
            # 노래가 재생 중이지 않을 때 이전 트랙의 재생 시간 계산 및 기록
            if last_track_id and start_time:
                end_time = time.time()
                play_duration = end_time - start_time
                formatted_play_time = format_time(play_duration)
                log_track(start_time, end_time, previous_track_info, formatted_play_time)

                # 로그 기록 후 초기화
                last_track_id = None
                start_time = None

        time.sleep(5)  # 5초마다 상태 확인
