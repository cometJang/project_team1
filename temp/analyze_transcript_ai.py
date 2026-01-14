import os
import re
import json
import pandas as pd
import sys
from pathlib import Path
# 스크립트가 포함된 폴더를 path에 추가
sys.path.append(str(Path(__file__).parent))

from typing import Optional, List, Dict
from pydantic import BaseModel
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ====================================
# 1. 환경 설정 및 API 키 (사용자 환경에 맞춰 설정 필요)
# ====================================
# 실제 실행 시에는 환경 변수나 .env 파일을 통해 설정하세요.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", " ")

# ====================================
# 2. 데이터 구조 정의 (Pydantic)
# ====================================
class SpeakerMapping(BaseModel):
    speaker_id: str
    real_name: str
    reason: str

class SpeakerMappingList(BaseModel):
    mappings: List[SpeakerMapping]

class DialogueAnalysis(BaseModel):
    target_person: Optional[str] = "None"
    sentiment: str # positive, negative, neutral
    category: str # emotion, attention, initiative, etc.
    summary: str

# ====================================
# 3. 인물 정보 및 설정 로드
# ====================================
def load_project_data():
    # 출연진 프로필 로드
    profile_df = pd.read_csv(r"c:\Users\Comet\Documents\GitHub\project_team1\table\character_profile.csv")
    profile_text = profile_df.to_string(index=False)
    
    # 분석 기준 로드
    with open(r"c:\Users\Comet\Documents\GitHub\project_team1\table\all_text_anlst.json", "r", encoding="utf-8") as f:
        analysis_config = json.load(f)
    
    return profile_text, analysis_config, profile_df["name"].tolist()

# ====================================
# 4. 화자 식별 (Speaker Identification)
# ====================================
def identify_speakers(dialogues, profile_text):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GEMINI_API_KEY)
    
    # 화자별로 특징적인 대화 샘플 추출 (최대 20개)
    speaker_samples = {}
    for d in dialogues:
        sid = d['speaker']
        if sid not in speaker_samples:
            speaker_samples[sid] = []
        if len(speaker_samples[sid]) < 5:
            speaker_samples[sid].append(d['text'])
            
    samples_str = ""
    for sid, texts in speaker_samples.items():
        samples_str += f"[{sid} 샘플 대화]:\n" + "\n".join(texts[:15]) + "\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"너는 연애 리얼리티 프로그램 분석가야. 다음 출연진 정보를 바탕으로 대화 샘플 속의 SPEAKER_XX가 실제 누구인지 매칭해줘.\n\n[출연진 정보]\n{profile_text}"),
        ("user", f"다음 대화 내용을 분석해서 각 SPEAKER ID가 누구인지 추론해줘. 반드시 제공된 출연진 이름 중에서만 골라야 해.\n\n{samples_str}")
    ])
    
    structured_llm = llm.with_structured_output(schema=SpeakerMappingList)
    result = structured_llm.invoke(prompt.invoke({}))
    
    mapping_dict = {m.speaker_id: m.real_name for m in result.mappings}
    return mapping_dict

# ====================================
# 5. 대화 분석 (Analysis)
# ====================================
def analyze_dialogue_with_ai(dialogue_text, current_speaker, mapping_dict, config_text, performer_names):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GEMINI_API_KEY)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"너는 대화 분석기야. 화자({current_speaker})의 말을 분석해서 누구에 대한 내용인지, 감정은 어떤지, 어떤 유형의 발화인지 분류해.\n"
                   f"분석 기준: {config_text}\n"
                   f"대상 인물 후보: {performer_names}"),
        ("user", "{text}")
    ])
    
    structured_llm = llm.with_structured_output(schema=DialogueAnalysis)
    try:
        result = structured_llm.invoke(prompt.invoke({"text": dialogue_text}))
        return result
    except:
        return None

# ====================================
# 6. 메인 실행 로직
# ====================================
def main():
    # 1. 데이터 로드
    print("데이터 로딩 중...", flush=True)
    profile_text, analysis_config, performer_names = load_project_data()
    
    # 2. 트랜스크립트 파싱 (기존 로직 활용)
    from analyze_transcript import parse_transcript
    transcript_path = r"c:\Users\Comet\Documents\GitHub\project_team1\transcript\환연2_1화_20화_요약.txt"
    dialogues = parse_transcript(transcript_path)
    print(f"총 {len(dialogues)}개의 대화를 불러왔습니다.", flush=True)
    
    # 3. 화자 매핑 (최초 1회 실행)
    print("AI가 화자를 식별 중입니다 (이 작업은 잠시 시간이 걸립니다)...", flush=True)
    mapping_dict = identify_speakers(dialogues, profile_text)
    print(f"식별 완료: {mapping_dict}", flush=True)
    
    # 4. 대화 분석
    output_path = r"c:\Users\Comet\Documents\GitHub\project_team1\table\ai_analysis_result.csv"
    
    # 이미 처리된 데이터가 있는지 확인 (재개 기능)
    results = []
    processed_count = 0
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            processed_count = len(existing_df)
            results = existing_df.to_dict('records')
            print(f"기존 분석 결과({processed_count}건)를 불러왔습니다. 이어서 분석을 시작합니다.", flush=True)
        except Exception as e:
            print(f"기존 파일을 읽는 중 오류 발생, 새로 시작합니다: {e}", flush=True)

    config_summary = json.dumps(analysis_config, ensure_ascii=False)
    
    print("대화 분석을 시작합니다 (AI 기반, 테스트용 상위 100개만)...", flush=True)
    for i, d in enumerate(tqdm(dialogues[:100])):
        # 이미 처리된 건너뛰기
        if i < processed_count:
            continue
            
        speaker_name = mapping_dict.get(d['speaker'], "Unknown")
        analysis = analyze_dialogue_with_ai(d['text'], speaker_name, mapping_dict, config_summary, performer_names)
        
        if analysis:
            results.append({
                "speaker": speaker_name,
                "text": d['text'],
                "target": analysis.target_person,
                "sentiment": analysis.sentiment,
                "category": analysis.category,
                "summary": analysis.summary
            })
            
        # 10개마다 중간 저장
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
            
    # 최종 저장
    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"분석 전과정 완료! 결과가 {output_path}에 저장되었습니다.", flush=True)

if __name__ == "__main__":
    main()
