import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import time


def search_youtube(query):
    """YouTube에서 검색하여 첫 번째 동영상 URL을 반환"""
    search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})

    if response.status_code != 200:
        return None  # 요청 실패 시 None 반환

    soup = BeautifulSoup(response.text, "html.parser")
    video = soup.find("a", href=True, attrs={"href": lambda x: x and "/watch" in x})
    if video:
        return f"https://www.youtube.com{video['href']}"
    return None  # 동영상 링크가 없을 경우 None 반환


def process_csv(file_path):
    """CSV 파일에서 artist-title을 검색하고 URL만 결과에 저장"""
    results = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            artist = row["artist"]
            title = row["title"]
            query = f"{artist} {title}"
            print(f"Searching for: {query}")
            video_url = search_youtube(query)
            results.append({"youtube_url": video_url})
            time.sleep(2)  # 2초 대기 (YouTube 요청 제한을 피하기 위해)

    # 결과를 새로운 CSV로 저장
    output_file = "youtube_urls.csv"
    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["youtube_url"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_file}")


# CSV 파일 경로 지정
csv_file_path = "../genre_playlist/ballad.csv"
process_csv(csv_file_path)
