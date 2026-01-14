import pandas as pd
import re
import os
from bs4 import BeautifulSoup

# ----------------------------------------
# 1. 설정 및 로드
# ----------------------------------------
profile_path = "table/character_profile.csv" if os.path.exists("table/character_profile.csv") else "character_profile.csv"
df_profile = pd.read_csv(profile_path)
name_to_id = dict(zip(df_profile["name"], df_profile["performer_id"]))

# ----------------------------------------
# 2. HTML 로드 및 각주 추출 (BeautifulSoup 방식: 안정적)
# ----------------------------------------
print("=" * 50)  # 구분선 출력
print("HTML 파일을 읽고 각주를 추출하는 중 (BS4 모드)...")  # 진행 로그

with open("source.html", "r", encoding="utf-8") as f:  # HTML 파일 오픈
    html_content = f.read()  # HTML 전체 문자열 읽기

soup = BeautifulSoup(html_content, "html.parser")  # 파서로 soup 생성

footnotes = {}  # fn-id -> 각주 텍스트 딕셔너리

# 클래스가 "_7L+-Tu3K" 인 span을 모두 찾기 (각주 컨테이너)
# (CSS 선택자에서는 + - 같은 문자가 이스케이프가 까다로워서 find_all을 추천)
fn_spans = soup.find_all("span", class_="_7L+-Tu3K")  # 각주 span 목록 수집

for sp in fn_spans:  # 각 각주 span 순회
    inner = sp.find("span", id=re.compile(r"^fn-\d+$"))  # 내부 span id="fn-숫자" 찾기
    if not inner:  # id span이 없으면 스킵
        continue  # 다음으로

    fn_id = inner.get("id")  # 예: "fn-22" 추출

    # 원문 예시: [22] "나도 응원할게"
    # span 전체 텍스트를 가져오면 [22]가 같이 섞이므로 제거 처리
    text = sp.get_text(" ", strip=True)  # span 내부 텍스트를 공백으로 합쳐서 추출

    text = re.sub(r"^\[\d+\]\s*", "", text)  # 맨 앞의 [22] 같은 번호 제거
    text = text.replace('"', "").replace("“", "").replace("”", "")  # 따옴표류 제거
    text = text.strip()  # 양끝 공백 제거

    if text:  # 비어있지 않으면 저장
        footnotes[fn_id] = text  # fn_id -> 텍스트 매핑 저장

print(f"✅ 총 {len(footnotes)}개의 각주를 추출했습니다.")  # 결과 로그
if footnotes:  # 샘플 출력
    sample_key = list(footnotes.keys())[0]  # 첫 키
    print(f"   샘플: {sample_key} -> {footnotes[sample_key]}")  # 샘플 로그
print("=" * 50)  # 구분선 출력

# ----------------------------------------
# 3. 테이블 데이터 파싱
# ----------------------------------------
soup = BeautifulSoup(html_content, "html.parser")
rows = []
tables = soup.find_all('table')

for table in tables:
    table_text = table.get_text()
    if "일차" not in table_text or "문자" not in table_text:
        continue

    # 발신자 이름 찾기
    sender_name = None
    sender_id = None
    for prev in table.find_all_previous(['h2', 'h3', 'h4', 'strong'], limit=15):
        ptext = prev.get_text(strip=True)
        for name in name_to_id.keys():
            if name in ptext:
                sender_name = name
                sender_id = name_to_id[name]
                break
        if sender_name: break
    
    if not sender_name: continue
    
    # 데이터 행 처리
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) < 2: continue

        day_txt = tds[0].get_text(strip=True)
        day_match = re.search(r'(\d+)일차', day_txt)
        if not day_match: continue
        day = int(day_match.group(1))

        # 수신자
        receiver_name = None
        r_link = tds[1].find('a', class_='wGjUIbJ4') or tds[1].find('a')
        if r_link: receiver_name = r_link.get_text(strip=True)

        # 메시지 내용 매칭 (모든 링크 검사)
        msg_parts = []
        for a_tag in tds[1].find_all('a'):
            href = a_tag.get('href', '')
            # href에서 fn-숫자 형태 추출
            fn_id_match = re.search(r'fn-(\d+)', href)
            if fn_id_match:
                fid = f"fn-{fn_id_match.group(1)}"
                if fid in footnotes:
                    msg_parts.append(footnotes[fid])
        
        message_txt = ' | '.join(msg_parts) if msg_parts else None
        
        # 데이트 상대
        date_target = None
        if len(tds) > 2:
            d_link = tds[2].find('a', class_='wGjUIbJ4')
            if d_link: date_target = d_link.get_text(strip=True)

        rows.append({
            "performer_id": int(sender_id),
            "name": sender_name,
            "day": day,
            "message": receiver_name,
            "message_txt": message_txt,
            "date": date_target
        })

# ----------------------------------------
# 4. 저장 및 결과 확인
# ----------------------------------------
if rows:
    df = pd.DataFrame(rows)
    df['performer_id'] = df['performer_id'].astype('Int64')
    df.to_csv("table/message_log.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ {len(df)}행 추출 완료. (message_txt가 있는 행: {df['message_txt'].notna().sum()}개)")
else:
    print("\n❌ 데이터를 찾지 못했습니다.")