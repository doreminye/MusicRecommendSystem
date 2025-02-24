import spotipy
import lyricsgenius
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import csv

# Genius API에 접속하기 위한 토큰
genius_token = 'Q1hN9L_fsy8OERUNCwrFnaluP9pXmuUL__zE4qPY4bUceFk0qIBuBiAlg0u_pyRI'
# Genius 객체 생성
genius = lyricsgenius.Genius(genius_token)

# CSV 파일 읽기
csv1 = pd.read_csv('kpop2.csv')
#csv2 = pd.read_csv('kpop2.csv')
#csv3 = pd.read_csv('rap2.csv')
#csv4 = pd.read_csv('ballad2.csv')
#csv5 = pd.read_csv('classic2.csv')
csvList = [csv1]

# CSV 파일 생성 및 헤더 작성
with open('kpop2_lyrics2.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 헤더 정의
    header = ["class", "track_id", "artist", "title", "lyrics"]
    writer.writerow(header)

    # 트랙 목록 순회
    for genre in csvList:
        for _, song in genre.iterrows():  # 각 행을 순회
            # 트랙 정보
            genre_num = song['prefer']
            track_id = song['track_id']
            artist_name = song['artist']
            track_name = song['title']

            print(track_id, artist_name, track_name)

            # 노래 검색
            song = genius.search_song(artist_name, track_name)

            # 가사가 없는 경우 스킵
            if song is None or song.lyrics is None:
                continue

            # 가사 텍스트 처리 및 불필요한 부분 제거
            song_lyrics = song.lyrics
            print(song_lyrics)
            # 예를 들어, '7 Contributors'와 같은 메타데이터를 제거하는 간단한 방법
            song_lyrics = song_lyrics.split('\n')[1:]  # 첫 번째 줄(메타데이터)을 제외한 나머지 줄을 가져옴
            song_lyrics = '\n'.join(song_lyrics)

            # 한 행에 대한 데이터 준비
            oneRow = [genre_num, track_id, artist_name, track_name, song_lyrics]

            # csv로 한 줄 저장
            writer.writerow(oneRow)
