from bs4 import BeautifulSoup
import pandas as pd
import re

# ============================================
# 1. Performer 데이터 로드
# ============================================
print("=" * 50)
print("[Performer 정보 로드 중...]")

df_performer = pd.read_csv('table/character_profile.csv', encoding='utf-8')
name_to_id = dict(zip(df_performer['name'], df_performer['performer_id']))

print(f"  총 {len(df_performer)}명 출연자 로드 완료")
print(f"  매핑: {name_to_id}")
print("=" * 50)

# ============================================
# 2. HTML 파일 읽기
# ============================================
print("\n[HTML 파일 읽기 중...]")

with open('source.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

print("  HTML 파일 로드 완료")

# ============================================
# 3. HTML 파싱 및 각주 수집
# ============================================
soup = BeautifulSoup(html_content, 'html.parser')

print("\n[각주 정보 수집 중...]")
footnotes = {}

footnote_elements = soup.find_all('span', class_='_7L+-Tu3K')

for fn in footnote_elements:
    id_span = fn.find('span', id=lambda x: x and x.startswith('fn-'))
    if id_span:
        fn_id = id_span.get('id')
        full_text = fn.get_text(strip=True)
        content = re.sub(r'^\[\d+\]\s*', '', full_text).strip()
        footnotes[fn_id] = content

print(f"  총 {len(footnotes)}개 각주 발견")
print("=" * 50)

# ============================================
# 4. 메시지 로그 추출
# ============================================
all_tables = soup.find_all('table', class_='XrDIkehY')

print(f"\n[디버깅] 총 테이블 개수: {len(all_tables)}개")
print("=" * 50)

all_data = []

for table_idx, table in enumerate(all_tables, 1):
    all_rows = table.find_all('tr', class_='llbpDXNr')
    data_rows = all_rows[1:] if len(all_rows) > 1 else all_rows
    
    print(f"\n[테이블 {table_idx}] - {len(data_rows)}개 행 처리 중...")
    
    for row_idx, row in enumerate(data_rows, 1):
        cells = row.find_all('td')
        
        if len(cells) < 3:
            continue
        
        try:
            # 일차 추출 (숫자만)
            day_text = cells[0].get_text(strip=True)
            day = int(re.sub(r'\D', '', day_text))  # "1일차" → 1
            
            # 속마음문자 발신자 이름
            sender_links = cells[1].find_all('a', class_='wGjUIbJ4')
            sender_name = sender_links[0].get_text(strip=True) if sender_links else None
            
            # sender_id 매칭 (FK)
            sender_id = name_to_id.get(sender_name) if sender_name else None
            
            # 속마음 문자 내용 추출
            sender_footnote_links = cells[1].find_all('a', class_='KHXRqSq-')
            sender_message_contents = []
            
            for fn_link in sender_footnote_links:
                fn_href = fn_link.get('href', '').replace('#', '')
                fn_id = fn_href.replace('rfn-', 'fn-')
                if fn_id in footnotes:
                    sender_message_contents.append(footnotes[fn_id])
            
            message_content = ' | '.join(sender_message_contents) if sender_message_contents else None
            
            # 데이트상대 이름
            receiver_links = cells[2].find_all('a', class_='wGjUIbJ4')
            receiver_name = receiver_links[0].get_text(strip=True) if receiver_links else None
            
            # receiver_id 매칭 (FK)
            receiver_id = name_to_id.get(receiver_name) if receiver_name else None
            
            # 디버깅: 처음 3개만 출력
            if row_idx <= 3:
                print(f"  행 {row_idx}: day={day}, sender={sender_name}(id={sender_id}), receiver={receiver_name}(id={receiver_id})")
                print(f"    문자내용: {message_content or 'NULL'}")
            
            all_data.append({
                '일차': day,
                '속마음문자_id': sender_id,
                '속마음문자': sender_name,
                '문자내용': message_content,
                '데이트상대_id': receiver_id,
                '데이트상대': receiver_name
            })
            
        except Exception as e:
            print(f"  ❌ 행 {row_idx} 처리 중 에러: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"  ✅ 완료")

# ============================================
# 5. DataFrame 변환 및 저장
# ============================================
df = pd.DataFrame(all_data)

print(f"\n" + "=" * 50)
print(f"[최종 결과]")
print(f"  - 총 {len(df)}행 추출 완료")
print(f"  - 속마음문자_id NULL: {df['속마음문자_id'].isna().sum()}개")
print(f"  - 데이트상대_id NULL: {df['데이트상대_id'].isna().sum()}개")
print(f"  - 문자내용 있음: {df['문자내용'].notna().sum()}개")
print("=" * 50)

# CSV 저장 (디버깅용 - 이름 포함)
try:
    df.to_csv('table/message_log_with_names.csv', index=False, encoding='utf-8-sig')
    print("\n✅ table/message_log_with_names.csv 저장 완료 (디버깅용)")
except Exception as e:
    print(f"\n❌ CSV 저장 실패: {e}")

# DB 삽입용 (이름 제외, ID만)
df_db = df[['일차', '속마음문자_id', '문자내용', '데이트상대_id']].copy()

try:
    df_db.to_csv('table/message_log.csv', index=False, encoding='utf-8-sig')
    print("✅ table/message_log.csv 저장 완료 (DB용)")
except Exception as e:
    print(f"❌ CSV 저장 실패: {e}")

# ============================================
# 6. 미리보기
# ============================================
print("\n[디버깅용 데이터 미리보기 - 이름 포함]")
print(df.head(20).to_string(index=False))

print("\n[DB 삽입용 데이터 미리보기 - ID만]")
print(df_db.head(20).to_string(index=False))

print("\n[통계]")
print(f"  - 총 행 수: {len(df)}")
print(f"  - 문자내용 있는 행: {df['문자내용'].notna().sum()}개")
print(f"  - 속마음문자 NULL: {df['속마음문자_id'].isna().sum()}개")
print(f"  - 데이트상대 NULL: {df['데이트상대_id'].isna().sum()}개")

print("\n[출연자별 속마음문자 횟수]")
sender_stats = df.groupby(['속마음문자_id', '속마음문자']).size().sort_values(ascending=False)
print(sender_stats)