# 다른 파이썬 파일에서 word_to_id.py 사용하기
from lyrics_preprocess import create_word_to_id

file_path = 'tmp_lyrics.csv'

# 단어 사전 생성
word_to_id = create_word_to_id(file_path)

# 단어와 ID 출력
print("\n단어: ID 번호")
for word, word_id in word_to_id.items():
    print(f"{word}: {word_id}")
