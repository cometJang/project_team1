# 데이터 전처리
import pandas as pd
from bs4 import BeautifulSoup

# CSV 읽기 (큰따옴표 안의 , 무시)
df = pd.read_csv(r"C:\Users\최미숙\Downloads\naver_article_250822.csv",
                sep=",", encoding="utf-8", quotechar='"', header=None)
# 컬럼명 지정
df.columns = ['naver_article_id', 'writer_nickname', 'released_at', 'view_count',
            'title', 'like_count', 'comment_count', 'comments', 'content']
# content에서 텍스트만 추출
def extract_text(html):
    if pd.isna(html):
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for vote in soup.select("div.CafeCustomVote"):
        vote.decompose()
    for img in soup.select("div.se-module-image"):
        img.decompose()
    texts = []
    for c in soup.select("div.se-component-content"):
        t = c.get_text(separator="\n", strip=True)
        if t:
            texts.append(t)
    return "\n".join(texts)
df['text'] = df['content'].apply(extract_text)
# 원래 content 컬럼 삭제
df.drop(columns=['content'], inplace=True)
# 확인
print(df.head())
# CSV 저장
df.to_csv("cafe_posts_clean.csv", index=False, encoding="utf-8-sig")
    
# 데이터 불러오기


import pandas as pd
from google.colab import drive

drive.mount('/content/drive')

path = "/content/drive/MyDrive/스파르타 파이썬/7조/cafe_posts_clean.csv"
df = pd.read_csv(path)

print(df.head())

    
# API 키


import os

os.environ["GEMINI_API_KEY"] = "Apikey***"
    
# LLM


# ====================================
# 1. 라이브러리 import
# ====================================
import os, sys, re, unicodedata, enum
import pandas as pd
from typing import Optional, List
from pydantic import BaseModel

from tqdm import tqdm
from IPython.display import display, clear_output

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ====================================
# 2. Google Drive 마운트
# ====================================
from google.colab import drive
drive.mount('/content/drive')

# ====================================
# 3. API 키
# ====================================
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# ====================================
# 4. 출력 스키마 정의
# ====================================
class SentimentEnum(enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"

class TextTypeEnum(enum.Enum):
    WORRY_RECOMMEND = "고민/추천형"
    INFORMATION     = "정보탐색형"
    TRANSACTION     = "거래형"
    REVIEW          = "후기형"
    EMOTION         = "감성표현형"
    OTHER           = "기타"

class ItemFull(BaseModel):
    category: str
    sentiment: SentimentEnum
    brand: Optional[str] = None
    line: Optional[str] = None
    detected_item: Optional[List[str]] = None
    text_type: TextTypeEnum

# ====================================
# 5. category 보정
# ====================================
ALLOWED_PREFIXES = ["디자인", "가격", "만족도", "추천", "CS", "기타"]

def enforce_category(category: str, sentiment: SentimentEnum) -> str:
    suffix_map = {
        SentimentEnum.POSITIVE: "_긍정",
        SentimentEnum.NEGATIVE: "_부정",
        SentimentEnum.NEUTRAL: "_중립",
    }
    suffix = suffix_map.get(sentiment, "_중립")

    if not category:
        return "기타" + suffix

    for prefix in ALLOWED_PREFIXES:
        if prefix in category:
            return prefix + suffix

    return "기타" + suffix

# ====================================
# 6. brand/line 매핑 함수
# ====================================
sys.path.append("/content/drive/MyDrive/스파르타 파이썬/7조")
from data3 import data3   # 브랜드/라인 사전 불러오기

def _normalize(s: str) -> str:
    """
    브랜드/라인명 매칭을 위한 텍스트 정규화 함수.
    - 유니코드 NFKC 정규화 (한글 자모 결합 등 처리)
    - 소문자 변환
    - 한글, 영문, 숫자 제외 모든 특수문자 및 공백 제거
    """
    if not s:
        return ""
    # 유니코드 정규화 및 소문자 변환
    s = unicodedata.normalize("NFKC", s).lower()
    # 한글(가-힣), 영문(a-z), 숫자(0-9)를 제외한 모든 문자(공백, 특수문자 등) 제거
    s = re.sub(r"[^a-z0-9가-힣]", "", s)
    return s

def normalize_brand_line(text: str, data3: list):
    text_norm = _normalize(text)

    # 1) 라인 매칭
    for entry in data3:
        brand = entry["브랜드"]
        line = entry.get("라인명")
        line_syns = entry.get("동의어", [])

        for syn in line_syns:
            if _normalize(syn) in text_norm:
                cleaned = re.sub(r"[\r\n]", "", line).strip() if line else None
                return brand, cleaned if cleaned else line

    # 2) 브랜드 매칭
    for entry in data3:
        brand = entry["브랜드"]
        brand_syns = entry.get("브랜드 동의어", [])
        for syn in [brand] + brand_syns:
            if _normalize(syn) in text_norm:
                return brand, None

    return None, None

# ====================================
# 7. 아이템 키워드 매핑
# ====================================
ITEM_KEYWORDS = {
    "반지": ["반지", "링"],
    "목걸이": ["목걸이", "네클리스"],
    "브레이슬릿": ["팔찌", "브레이슬릿", "브레이브슬릿"],
    "시계": ["시계", "워치", "손목시계"],
    "귀걸이": ["귀걸이", "귀고리", "이어링"]
}

def extract_items(text: str) -> List[str]:
    found = []
    for key, synonyms in ITEM_KEYWORDS.items():
        for syn in synonyms:
            if syn in text:
                found.append(key)
                break
    return found if found else None

# ====================================
# 8. Classifier 빌드
# ====================================
def build_classifier():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system",
        "너는 감성 및 카테고리 분류기야. "
        "category는 반드시 '디자인, 가격, 만족도, 추천, CS, 기타' 중 하나에 "
        "'_긍정', '_부정', '_중립' 접미사가 붙은 형태여야 한다. "
        "sentiment는 positive/negative/neutral 중 하나. "
        "text_type은 고민/추천형, 정보탐색형, 거래형, 후기형, 감성표현형, 기타 중 하나여야 한다."),
    ("user", "{text}")
])

    structured_llm = llm.with_structured_output(schema=ItemFull)

    def classify(text: str) -> ItemFull:
        result = structured_llm.invoke(prompt.invoke({"text": text}))
        result.category = enforce_category(result.category, result.sentiment)
        return result

    return classify

# ====================================
# 9. Splitter 정의
# ====================================
class OpinionSegment(BaseModel):
    sub_text: str

class Segments(BaseModel):
    segments: List[OpinionSegment]

def build_splitter():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "문장을 브랜드/라인별 의견 단위로 나누어라. "
        "출력은 {{ \"segments\": [...] }} 형태의 JSON 객체여야 한다."),
    ("user", "{text}")
    ])
    return prompt | llm.with_structured_output(schema=Segments)

# ====================================
# 10. 실행
# ====================================
classifier = build_classifier()
splitter = build_splitter()

csv_input_path = "/content/drive/MyDrive/스파르타 파이썬/7조/cafe_posts_clean.csv"
df = pd.read_csv(csv_input_path)

# A 담당 범위
target_df = df.iloc[2000:5323].copy()
chunksize = 200
output_path = "/content/drive/MyDrive/스파르타 파이썬/7조/final_output_A.csv"

# 이미 저장된 행 확인
processed_ids = set()
if os.path.exists(output_path):
    saved_df = pd.read_csv(output_path, usecols=["naver_article_id"])
    processed_ids = set(saved_df["naver_article_id"].dropna().unique())
    print(f"이미 저장된 행 개수: {len(processed_ids)}")

for start in tqdm(range(0, len(target_df), chunksize), desc="샘플 처리 중"):
    batch = target_df.iloc[start:start+chunksize].copy()
    batch = batch[~batch["naver_article_id"].isin(processed_ids)]
    if batch.empty:
        continue

    results = []
    for _, row in batch.iterrows():
        text = row["text"]
        article_id = row.get("naver_article_id", None)
        writer_nickname = row.get("writer_nickname", None)
        view_count = row.get("view_count", None)
        released_at = row.get("released_at", None)

        # 1) 전체 문장 기준 text_type
        try:
            global_result = classifier(text)
            text_type_value = global_result.text_type.value
        except Exception as e:
            print(f"[분류기 에러: {e}] {article_id}")
            continue

        # 2) splitter 호출
        try:
            segs = splitter.invoke({"text": text})
        except Exception as e:
            print(f"[splitter 호출 실패: {e}] {article_id}")
            continue

        # splitter 결과 없을 때 건너뛰기
        if not segs or not hasattr(segs, "segments"):
            print(f"[splitter 결과 없음] {article_id}")
            continue

        for seg in segs.segments:
            seg_text = seg.sub_text.strip()
            brand, line = normalize_brand_line(seg_text, data3)
            if not brand and not line:
                continue

            try:
                local_result = classifier(seg_text).model_dump(mode="json")
            except Exception as e:
                print(f"[local classifier 실패: {e}] {article_id}")
                continue

            detected_items = extract_items(seg_text)

            results.append({
                "naver_article_id": article_id,
                "writer_nickname": writer_nickname,
                "view_count": view_count,
                "released_at": released_at,
                "brand": brand,
                "line": line,
                "detected_item": detected_items,
                "category": local_result["category"],
                "sentiment": local_result["sentiment"],
                "text_type": text_type_value
            })

    if not results:
        continue

    final_df = pd.DataFrame(results, columns=[
        "naver_article_id","writer_nickname","view_count","released_at",
        "brand","line","detected_item","category","sentiment","text_type"
    ])
    final_df = final_df.explode("detected_item").reset_index(drop=True)

    # 숫자/날짜형 자동 변환
    final_df["view_count"] = pd.to_numeric(final_df["view_count"], errors="coerce").astype("Int64")
    final_df["released_at"] = pd.to_datetime(final_df["released_at"], errors="coerce")

    # append 저장
    final_df.to_csv(output_path, mode="a", header=not os.path.exists(output_path),
                    index=False, encoding="utf-8-sig")

    processed_ids.update(final_df["naver_article_id"].dropna().unique())

    # 중간 진행상황 표시
    clear_output(wait=True)
    print(f"{start} ~ {start+chunksize-1} 행 처리 & 저장 완료 ✅ (누적 {len(processed_ids)})")
    print("중간 결과 미리보기:")
    display(final_df.head(3))

print("처리 완료")