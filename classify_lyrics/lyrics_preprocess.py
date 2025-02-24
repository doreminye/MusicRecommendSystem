import gensim.models
from gensim.models import Word2Vec
import csv
import numpy as np
import re

# 전처리 함수
def preprocess(text):
    text = re.sub(r'\[.*?\]', '', text)  # 대괄호와 내용 제거
    text = text.replace('"', '').replace("'", '')  # 따옴표 제거
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
    text = re.sub(r'\s+', ' ', text).strip()  # 연속된 공백 제거 및 앞뒤 공백 제거
    return text

# 전처리 -> 띄어쓰기 기준으로 분리
def make_token(file_path):
    token_lyrics = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 건너뛰기
        for row in reader:
            lyrics = row[4]  # 가사 컬럼
            pre_lyrics = preprocess(lyrics)
            tokens = pre_lyrics.split()
            token_lyrics.append(tokens)
    return token_lyrics

# Word2Vec 모델 학습
def train_model(token_lyrics):
    model = Word2Vec(sentences=token_lyrics, vector_size=100, window=5, min_count=1, sg=0)
    return model

# 유사 벡터 계산
def lyrics_vector(lyrics, model_path):
    genre_word = []
    saved_model = gensim.models.Word2Vec.load(model_path)  # Word2Vec 모델 로드

    # 단어 벡터 추출
    for word in lyrics:
        if word in saved_model.wv:  # 모델에 단어가 있는지 확인
            vector = saved_model.wv[word]
            genre_word.append(vector)

    # 평균 벡터 계산
    if genre_word:  # 단어 벡터가 존재하면 평균 계산
        genre_avg = np.mean(genre_word, axis=0)
    else:  # 단어 벡터가 없으면 None 반환
        genre_avg = None

    return genre_avg

# 실행 코드
file_path = 'tmp_lyrics.csv'
token_lyrics = make_token(file_path)  # 가사 토큰화
test_model = train_model(token_lyrics)  # 모델 학습
test_model.save("word2vec_model.bin")  # 모델 저장


# 하나의 가사를 벡터로 변환
flattened_lyrics = [word for song in token_lyrics for word in song]  # 리스트 평탄화
genre_avg = lyrics_vector(flattened_lyrics, 'word2vec_model.bin')  # 평균 벡터 계산

print("장르 벡터 평균:", genre_avg)
