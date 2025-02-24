import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa.display

# 폴더 내 모든 MP3 파일을 찾기 위한 경로
folder_path = '../youtube_m/downloads_dislike'
save_folder_path = 'spectrogram_dislike'
# 폴더 내 파일 목록 탐색
for file_name in os.listdir(folder_path):
    # MP3 파일만 처리
    if file_name.endswith('.mp3'):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_path}")

        # MP3 파일 로드 (1분까지만 로드)
        y, sr = librosa.load(file_path, sr=None, offset=30, duration=60)  # 60초로 제한

        # 전체 곡에 대해 스펙트로그램 생성
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        # 스펙트로그램 표시
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {file_name} (First 1 minute)')

        # 이미지로 저장 (파일명에 따라 저장)
        output_image_path = os.path.join(save_folder_path, f'{os.path.splitext(file_name)[0]}_spectrogram.png')
        plt.savefig(output_image_path)
        plt.close()  # 현재 플롯 닫기
        print(f"Saved spectrogram for {file_name} to {output_image_path}")

print("모든 파일에 대한 1분 스펙트로그램 생성 완료!")
