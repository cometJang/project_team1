import pandas as pd

# 1) 참가자 프로필
profiles = pd.DataFrame([
    [1,"성해은","F",1994,"INFP","F","P","정규민",76,15,5],
    [2,"정규민","M",1994,"ENTJ","T","J","성해은",76,15,1],
    [3,"이나연","F",1996,"ESFP","F","P","남희두",31,5,1],
    [4,"남희두","M",1997,"INTP","T","P","이나연",31,5,6],
    [5,"박원빈","M",1997,"ENFJ","F","J","김지수",16,40,1],
    [6,"김지수","F",1997,"ENFP","F","P","박원빈",16,40,1],
    [7,"김태이","M",1996,"INTJ","T","J","이지연",6,18,1],
    [8,"이지연","F",2001,"ESTP","T","P","김태이",6,18,1],
    [9,"박나언","F",1998,"INTP","T","P","정현규",14,30,13],
    [10,"정현규","M",1998,"ESTJ","T","J","박나언",14,30,15],
], columns=["performer_id","name","gender","birth_year","mbti",
            "mbti_t_f","mbti_j_p","x_partner","relationship_months",
            "breakup_months","entry_episode"])

# 2) 나무위키에서 정리한 일차/문자상대/데이트상대 데이터 (message_log_new.csv 사용)
try:
    daily = pd.read_csv("table/message_log_new.csv")
    # 기존 performer_id 제거 (프로필 조인으로 새로 부여하기 위함)
    if 'performer_id' in daily.columns:
        daily = daily.drop(columns=['performer_id'])
except FileNotFoundError:
    # 파일이 없을 경우 예외 처리 또는 기존 이름 시도
    daily = pd.read_csv("hwanseung_daily_raw.csv")

# 3) 이름으로 프로필 조인해서 performer_id 붙이기
merged = daily.merge(profiles[["performer_id","name"]], on="name", how="left")

# 4) 컬럼 순서 정리
final = merged[["performer_id","name","day","message","message_txt","date"]]

# 5) 최종 CSV 저장
final.to_csv("table/hwanseung2_messages.csv", index=False, encoding="utf-8-sig")
