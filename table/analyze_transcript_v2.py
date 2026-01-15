
import os
import re
import json
import pandas as pd
import sys
from pathlib import Path

# ====================================
# 1. Imports and Setup
# ====================================
try:
    from typing import Optional, List, Dict
    from pydantic import BaseModel, ValidationError
    from tqdm import tqdm
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please run: pip install langchain-google-genai langchain langchain-core pydantic tqdm pandas")
    print("Also ensure you handle api keys correctly.")
    sys.exit(1)

# Helper function imports
try:
    from dotenv import load_dotenv
    load_dotenv()
    # Check for GEMINI_API_KEY
    if "GEMINI_API_KEY" not in os.environ:
         # Fallback to hardcoded key if provided in original snippet, though env var is safer
         pass 
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDKBnPv37oJ2F3_79eSmLhRcvrAWksbe1k")

# ====================================
# 2. Data Structures (Pydantic)
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
# 3. Helper: Parse Transcript (Copied)
# ====================================
def parse_transcript(file_path):
    """
    Parses the transcript file.
    Assumes format:
    SPEAKER_XX HH:MM:SS
    Text
    """
    dialogues = []
    current_speaker = "UNKNOWN"
    current_timestamp = "00:00:00"
    current_text = []

    # Regex for Speaker + Time: SPEAKER_19 00:00:00
    speaker_time_pattern = re.compile(r'^(SPEAKER_\d+)\s+(\d{2}:\d{2}:\d{2})$')
    # Regex for Time only
    time_only_pattern = re.compile(r'^(\d{2}:\d{2}:\d{2})$')

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check patterns
        speaker_match = speaker_time_pattern.match(line)
        time_match = time_only_pattern.match(line)
        
        if speaker_match:
            if current_text:
                dialogues.append({
                    'speaker': current_speaker,
                    'timestamp': current_timestamp,
                    'text': " ".join(current_text)
                })
                current_text = []
            current_speaker = speaker_match.group(1)
            current_timestamp = speaker_match.group(2)
        elif time_match:
             if current_text:
                dialogues.append({
                    'speaker': current_speaker,
                    'timestamp': current_timestamp,
                    'text': " ".join(current_text)
                })
                current_text = []
             current_timestamp = time_match.group(1)
        else:
             if line.startswith('[') and ('m4a' in line or '환승연애' in line):
                continue
             current_text.append(line)
             
    if current_text:
        dialogues.append({
             'speaker': current_speaker,
             'timestamp': current_timestamp,
             'text': " ".join(current_text)
        })
    return dialogues

# ====================================
# 4. Load Resources
# ====================================
def load_project_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Assuming script is in table/, go up one level to root if needed, or check absolute paths
    # User env: /Users/t2024-m0246/Documents/GitHub/project_team1
    # Script is in: /Users/t2024-m0246/Documents/GitHub/project_team1/table/analyze_transcript_v2.py
    
    # Adjust paths relative to script location
    project_root = os.path.abspath(os.path.join(base_path, "..")) 
    
    profile_path = os.path.join(project_root, "table/character_profile.csv")
    config_path = os.path.join(project_root, "table/all_text_anlst.json")
    
    if os.path.exists(profile_path):
        profile_df = pd.read_csv(profile_path)
    else:
        # Fallback to hardcoded path if relative fails
        profile_df = pd.read_csv("/Users/t2024-m0246/Documents/GitHub/project_team1/table/character_profile.csv")
        
    profile_text = profile_df.to_string(index=False)
    
    if os.path.exists(config_path):
         with open(config_path, "r", encoding="utf-8") as f:
            analysis_config = json.load(f)
    else:
         with open("/Users/t2024-m0246/Documents/GitHub/project_team1/table/all_text_anlst.json", "r", encoding="utf-8") as f:
            analysis_config = json.load(f)

    return profile_text, analysis_config, profile_df["name"].tolist()

# ====================================
# 5. Speaker Identification (Improved)
# ====================================
def identify_speakers(dialogues, profile_text, file_name):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GEMINI_API_KEY)

    # Strategy: Collect more extensive samples per speaker
    # Instead of random, collect the longest texts? Or just the first 15 occurences.
    speaker_samples = {}
    for d in dialogues:
        sid = d['speaker']
        if sid not in speaker_samples:
            speaker_samples[sid] = []
        # Increase sample count to capture more context
        if len(speaker_samples[sid]) < 20: 
            speaker_samples[sid].append(d['text'])

    samples_str = ""
    for sid, texts in speaker_samples.items():
        # Join lines with newlines
        content = "\n".join(texts)
        samples_str += f"[{sid}]:\n{content}\n----------------\n"

    # Enhanced Propmt with specific instructions for context
    system_prompt = (
        f"너는 연애 리얼리티 프로그램(환승연애2) 분석가야. 다음 출연진 정보를 바탕으로 대화 샘플 속의 SPEAKER_XX가 실제 누구인지 매칭해줘.\n"
        f"파일 이름: '{file_name}' (파일 이름에 있는 인물이 주요 화자일 가능성이 매우 높음. 예를 들어 '희두나연'이면 남희두, 이나연이 주 화자일 것임).\n"
        f"문맥 단서: 아이스하키(희두), 아나운서(나연), 승무원(해은), 편집샵(규민) 등의 직업 키워드나, 과거 서사, 말투를 주의 깊게 봐줘.\n"
        f"반드시 모든 SPEAKER_ID를 매핑해야 해.\n\n"
        f"[출연진 정보]\n{profile_text}"
    )
    
    user_prompt = f"다음 대화 내용을 분석해서 각 SPEAKER ID가 누구인지 추론해줘. 반드시 제공된 출연진 이름 중에서만 골라야 해.\n\n{samples_str}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    structured_llm = llm.with_structured_output(schema=SpeakerMappingList)
    
    print("Sending request to Gemini for speaker identification...", flush=True)
    try:
        result = structured_llm.invoke(prompt.invoke({}))
        mapping_dict = {m.speaker_id: m.real_name for m in result.mappings}
        return mapping_dict
    except Exception as e:
        print(f"Error in speaker identification logic: {e}")
        # Identify fail => Empty mapping? Or fallback?
        return {}


# ====================================
# 6. Dialogue Analysis
# ====================================
def analyze_dialogue_with_ai(dialogue_text, current_speaker, mapping_dict, config_text, performer_names):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GEMINI_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 대화 분석기야. 화자({current_speaker})의 말을 분석해서 누구에 대한 내용인지, 감정은 어떤지, 어떤 유형의 발화인지 분류해.\n"
                   "분석 기준: {analysis_config_text}\n"
                   "대상 인물 후보: {performer_names}"),
        ("user", "{dialogue_text}")
    ])

    structured_llm = llm.with_structured_output(schema=DialogueAnalysis)
    try:
        result = structured_llm.invoke(prompt.invoke({
            "current_speaker": current_speaker,
            "analysis_config_text": config_text,
            "performer_names": performer_names,
            "dialogue_text": dialogue_text
        }))
        return result
    except Exception as e:
        # print(f"Error for speaker {current_speaker}: {e}")
        return None

# ====================================
# 7. Main
# ====================================
def main():
    # Setup Paths
    # Current script is at: /Users/t2024-m0246/Documents/GitHub/project_team1/table/analyze_transcript_v2.py
    # or root? The user created it in table/.
    
    # Let's verify absolute path of transcript
    transcript_path = "/Users/t2024-m0246/Documents/GitHub/project_team1/transcript/환연2_희두나연.txt"

    print("====================================")
    print("   Transcript Analysis V2 (Gemini)  ")
    print("====================================")
    
    # 1. Load Data
    print("1. Loading project data...", flush=True)
    try:
        profile_text, analysis_config, performer_names = load_project_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 2. Parse Transcript
    print(f"2. Parsing transcript: {transcript_path}", flush=True)
    dialogues = parse_transcript(transcript_path)
    if not dialogues:
        print("No dialogues parsed. Exiting.")
        return
    print(f"   -> {len(dialogues)} segments found.", flush=True)

    # 3. Speaker Identification
    file_name = os.path.basename(transcript_path)
    print(f"3. Identifying speakers (Hint: File name is '{file_name}')...", flush=True)
    
    # We will pass a subset of dialogues to identification if they are too many?
    # Pass all? Passing 600 segments might exceed token limit or confuse the model.
    # Let's pass the distinct speakers' samples (done in identify_speakers)
    mapping_dict = identify_speakers(dialogues, profile_text, file_name)
    print(f"   -> Result: {mapping_dict}", flush=True)

    # 4. Analysis
    output_path = "/Users/t2024-m0246/Documents/GitHub/project_team1/table/ai_analysis_result_v2.csv"
    print(f"4. Analyzing content -> {output_path}", flush=True)

    results = []
    config_summary = json.dumps(analysis_config, ensure_ascii=False)
    
    # Process all or subset? User didn't specify limit, so all.
    # But using tqdm
    
    count = 0
    for d in tqdm(dialogues, desc="Analyzing"):
        speaker_name = mapping_dict.get(d['speaker'], d['speaker']) # Fallback to ID if not mapped
        
        # Skip unknown if desired? No, analyze anyway.
        
        analysis = analyze_dialogue_with_ai(d['text'], speaker_name, mapping_dict, config_summary, performer_names)
        
        if analysis:
            results.append({
                "speaker_id": d['speaker'],
                "speaker_name": speaker_name,
                "text": d['text'],
                "target": analysis.target_person,
                "sentiment": analysis.sentiment,
                "category": analysis.category,
                "summary": analysis.summary
            })
            count += 1
            
        # Optional: save every 50 items
        if count % 50 == 0 and count > 0:
             pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")

    # Final Save
    if results:
        pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
        print("Done.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()