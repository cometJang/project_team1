import requests
from bs4 import BeautifulSoup
import pandas as pd

# ============================================
# 1. URL 설정
# ============================================
urls_wiki = "https://namu.wiki/w/%ED%99%98%EC%8A%B9%EC%97%B0%EC%95%A02"

headers = {
    "User-Agent": "Mozilla/5.0"
}

# ============================================
# 2. 요청 및 HTML 파싱
# ============================================
response = requests.get(urls_wiki, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

# ============================================
# 3. 제목 추출
# ============================================
title = soup.title.text.strip() if soup.title else "제목 없음"
print(f"[제목] {title}")

# ============================================
# 4. 본문 텍스트 추출
# (나무위키 본문 영역)
# ============================================
texts = []

# 나무위키는 보통 article 태그 하위에 본문이 있음
content_root = soup.select_one("article")

if content_root:
    for elem in content_root.find_all(["p", "li"]):
        text = elem.get_text(separator=" ", strip=True)
        if text:
            texts.append(text)

full_text = "\n".join(texts)

print(f"[본문 길이] {len(full_text)} characters")

# ============================================
# 5. 일차 정보 생성
# (URL 기반, 필요 시 수정 가능)
# ============================================
day = urls_wiki.split("/")[-1]   # 환승연애2

# ============================================
# 6. 데이터 적재
# ============================================
data = []

data.append({
    "일차": day,
    "제목": title,
    "본문": full_text,
    # 아래 컬럼은 기존 스키마 호환용 (없으므로 None)
    "속마음문자_id": None,
    "속마음문자": None,
    "문자내용": full_text,
    "데이트상대_id": None,
    "데이트상대": None
})

# ============================================
# 7. DataFrame 변환
# ============================================
df = pd.DataFrame(data)

print("\n[미리보기]")
print(df.head().to_string(index=False))

# ============================================
# 8. CSV 저장
# ============================================
output_path = "table/wiki_crawling_result.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n✅ CSV 저장 완료: {output_path}")
