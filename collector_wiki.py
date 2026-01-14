from bs4 import BeautifulSoup
import pandas as pd
import re

# ----------------------------------------
# 1. 참가자 프로필 로드
# ----------------------------------------
try:
    df_profile = pd.read_csv("table/character_profile.csv")
except FileNotFoundError:
    df_profile = pd.read_csv("character_profile.csv")

name_to_id = dict(zip(df_profile["name"], df_profile["performer_id"]))

# ----------------------------------------
# 2. HTML 로드
# ----------------------------------------
with open("source.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

# ----------------------------------------
# 3. 결과 저장용 리스트
# ----------------------------------------
rows = []

# ----------------------------------------
# 4. 일차별 데이터 파싱 (HTML 구조에 맞게 수정됨)
# ----------------------------------------

# 각주 정보 수집 (collector.py 로직 참조)
footnotes = {}
footnote_elements = soup.find_all('span', class_='_7L+-Tu3K')
for fn in footnote_elements:
    id_span = fn.find('span', id=lambda x: x and x.startswith('fn-'))
    if id_span:
        fn_id = id_span.get('id')
        full_text = fn.get_text(strip=True)
        content = re.sub(r'^\[\d+\]\s*', '', full_text).strip()
        footnotes[fn_id] = content

# 테이블 순회
tables = soup.find_all('table', class_='XrDIkehY')

for table in tables:
    # 첫 행은 헤더일 수 있으므로 확인 (collector.py는 [1:] 사용)
    trs = table.find_all('tr', class_='llbpDXNr')
    data_rows = trs[1:] if len(trs) > 1 else trs

    for tr in data_rows:
        tds = tr.find_all('td')
        if len(tds) < 3:
            continue

        # 1) 일차 파싱
        day_text = tds[0].get_text(strip=True)

        # '일차' 텍스트 포함 여부 확인 (잘못된 테이블 필터링)
        if "일차" not in day_text:
            continue

        # "1일차" -> 1 추출
        day_match = re.search(r'\d+', day_text)
        if not day_match:
            continue
        day = int(day_match.group())

        # 2) 발신자 (Sender) - 속마음문자
        sender_links = tds[1].find_all('a', class_='wGjUIbJ4')
        sender = sender_links[0].get_text(strip=True) if sender_links else None

        # 3) 문자 내용 (각주 참조)
        sender_footnote_links = tds[1].find_all('a', class_='KHXRqSq-')
        msg_parts = []
        for fn_link in sender_footnote_links:
            fn_href = fn_link.get('href', '').replace('#', '')
            fn_id = fn_href.replace('rfn-', 'fn-')
            if fn_id in footnotes:
                msg_parts.append(footnotes[fn_id])
        message_txt = ' | '.join(msg_parts) if msg_parts else None

        # 4) 수신자 (Receiver) - 데이트상대
        receiver_links = tds[2].find_all('a', class_='wGjUIbJ4')
        date_target = receiver_links[0].get_text(strip=True) if receiver_links else None

        # performer_id 변환: 값이 있으면 정수 변환, 없으면 None (CSV 저장 시 공란)
        p_id = name_to_id.get(sender)
        performer_id_val = int(p_id) if pd.notna(p_id) else None

        rows.append({
            "performer_id": performer_id_val,
            "name": sender,
            "day": day,
            "message": message_txt,
            "message_txt": message_txt,
            "date": date_target
        })

# ----------------------------------------
# 5. DataFrame 생성 및 저장
# ----------------------------------------
if rows:
    df = pd.DataFrame(rows)
    # Ensure correct column order: performer_id,name,day,message,message_txt,date
    df = df[['performer_id', 'name', 'day', 'message', 'message_txt', 'date']]
    # ID를 정수형(Int64)으로 변환하여 .0 제거
    if 'performer_id' in df.columns:
        df['performer_id'] = df['performer_id'].astype('Int64')

    df.to_csv("table/message_log_new.csv", index=False, encoding="utf-8-sig")
    print(f"Succeess: {len(df)} rows saved to table/message_log_new.csv with new schema.")
else:
    print("No rows extracted. Check parsing logic.")
    # Create empty with headers
    pd.DataFrame(columns=['performer_id', 'name', 'day', 'message', 'message_txt', 'date']).to_csv("table/message_log.csv", index=False, encoding="utf-8-sig")
