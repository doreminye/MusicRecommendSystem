import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np

# Spotify API 인증
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(client_id='bbb60dd02ca3480f8da2af6e4d0c4b14', client_secret='8bed2c31363740e7b1d4fecee42c14c4'))

# 트랙의 오디오 분석 가져오기
track_id = '곡 ID'  # 예: '3n3Ppam7vgaVa1iaRUc9Lp'
audio_analysis = sp.audio_analysis(track_id)

# 세그먼트 정보 추출
segments = audio_analysis['segments']


# 각 세그먼트에서 분위기 변화 감지
def calculate_mood_change_per_second(segments):
    mood_changes = []

    # 이전 세그먼트의 에너지와 기분 값 초기화
    prev_valence = segments[0]['confidence']
    prev_energy = segments[0]['loudness_max']

    for segment in segments:
        # 현재 세그먼트의 에너지와 기분 값
        valence = segment['confidence']
        energy = segment['loudness_max']

        # 분위기 변화율 계산 (이전 값과의 차이 절대값)
        valence_change = abs(valence - prev_valence)
        energy_change = abs(energy - prev_energy)

        # 변화 정도에 따른 점수 계산 (변화가 클수록 점수가 높아짐)
        mood_change_score = (valence_change + energy_change) * 100  # 점수화

        # 결과 저장
        mood_changes.append(mood_change_score)

        # 현재 값을 이전 값으로 업데이트
        prev_valence = valence
        prev_energy = energy

    return mood_changes


# 각 초별로 분위기 변화 점수 계산
mood_change_scores = calculate_mood_change_per_second(segments)

# 결과 출력
for i, score in enumerate(mood_change_scores):
    print(f"{i + 1}초 분위기 변화 점수: {score:.2f}")

#git test
#git test2
print("Test2")
print("Test3")