
import os
import re
import json
import shutil
import pandas as pd
from collections import defaultdict
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate

# 0) ENV
load_dotenv(".env")
SOLAR_API_KEY = os.environ.get("SOLAR_API_KEY")

if not SOLAR_API_KEY:
    print("Warning: SOLAR_API_KEY not found in .env. Analysis might fail.")

# 1) Parse
def parse_transcript(transcript_path: str) -> List[Dict[str, str]]:
    dialogues = []
    current_speaker = "UNKNOWN"
    current_text = []
    speaker_time_pattern = re.compile(r'^(?:\[(.+?)\]|(SPEAKER_\d+))\s+(\d{2}:\d{2}:\d{2})')
    time_only_pattern = re.compile(r'^\s*(\d{2}:\d{2}:\d{2})\s*$')

    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        speaker_match = speaker_time_pattern.match(line)
        time_match = time_only_pattern.match(line)
        
        if speaker_match:
            if current_text:
                dialogues.append({"speaker": current_speaker, "text": " ".join(current_text)})
                current_text = []
            current_speaker = speaker_match.group(1) or speaker_match.group(2)
            current_speaker = current_speaker.strip()
        elif time_match:
            if current_text:
                dialogues.append({"speaker": current_speaker, "text": " ".join(current_text)})
                current_text = []
        else:
            current_text.append(line)
            
    if current_text:
        dialogues.append({"speaker": current_speaker, "text": " ".join(current_text)})
    return dialogues

# 2) Config & Regex
def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_text_regex(dialogues: List[Dict[str, str]], config: Dict) -> Dict:
    keyword_map = {}
    if "emotion_polarity" in config:
        if "positive" in config["emotion_polarity"]:
            keyword_map["positive"] = config["emotion_polarity"]["positive"]["keywords"]
        if "negative" in config["emotion_polarity"]:
            keyword_map["negative"] = config["emotion_polarity"]["negative"]["keywords"]
    if "attention" in config:
        if "mention_x" in config["attention"]:
            keyword_map["mention_x"] = config["attention"]["mention_x"]["keywords"]
        if "mention_other" in config["attention"]:
             keyword_map["mention_other"] = ["해은", "규민", "나연", "희두", "원빈", "지수", "태희", "지연", "나언", "현규", "지현"]
    if "initiative" in config:
        keyword_map["initiative"] = config["initiative"]["keywords"]
    
    total_counts = defaultdict(int)
    speaker_stats = defaultdict(lambda: defaultdict(int))
    
    for entry in dialogues:
        speaker = entry["speaker"]
        text = entry["text"]
        for category, keywords in keyword_map.items():
            for kw in keywords:
                if kw in text:
                    c = text.count(kw)
                    total_counts[category] += c
                    speaker_stats[speaker][category] += c
    return {"summary": dict(total_counts), "by_speaker": {k: dict(v) for k, v in speaker_stats.items()}}

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# 3) Schemas
class SpeakerMapping(BaseModel):
    speaker_id: str
    real_name: str
    reason: str
    confidence: float

class SpeakerMappingList(BaseModel):
    mappings: List[SpeakerMapping]

class DialogueAnalysis(BaseModel):
    target_person: Optional[str] = "None"
    sentiment: str
    category: str
    summary: str

# 4) Cache
def load_mapping_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_mapping_cache(cache_path: str, mapping_obj: Dict[str, Any]) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(mapping_obj, f, ensure_ascii=False, indent=2)

# 5) Samples
def build_speaker_samples(dialogues, max_samples=20, min_len=20):
    speaker_samples = {}
    for d in dialogues:
        sid = d["speaker"]
        text = d["text"].strip()
        if len(text) < min_len: continue
        score = 0
        if re.search(r"(오빠|언니|누나|형)", text): score += 2
        if re.search(r"(집|회사|학교|공항|한국|연대|북문)", text): score += 1
        if re.search(r"(헤어|만나|연락|데이트|사귀|좋아|싫어)", text): score += 1
        if score < 1: continue
        
        speaker_samples.setdefault(sid, [])
        if len(speaker_samples[sid]) >= max_samples: continue
        speaker_samples[sid].append(text)
    return speaker_samples

# 6) Identify
def identify_speakers_force_choice(dialogues, profile_text, performer_names, llm, cache_path):
    cached = load_mapping_cache(cache_path)
    if cached:
        print("Using cached mapping.")
        return cached

    print("Identifying speakers...")
    speaker_samples = build_speaker_samples(dialogues)
    samples_str = ""
    for sid, texts in speaker_samples.items():
        samples_str += f"[{sid} 샘플]\n" + "\n".join(texts) + "\n\n"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 연애 리얼리티 분석가야. SPEAKER_XX를 {performer_names} 중 하나로 매핑해. Unknown 금지.\n[출연진 정보]\n{profile_text}"),
        ("user", "{samples_str}")
    ])
    try:
        structured = llm.with_structured_output(schema=SpeakerMappingList)
        result = structured.invoke(prompt.invoke({
            "samples_str": samples_str,
            "performer_names": str(performer_names),
            "profile_text": profile_text
        }))
        mapping_obj = {}
        for m in result.mappings:
            mapping_obj[m.speaker_id] = {"name": m.real_name, "confidence": m.confidence, "reason": m.reason}
        save_mapping_cache(cache_path, mapping_obj)
        return mapping_obj
    except Exception as e:
        print(f"ID Error: {e}")
        return {}

# 7) Analyze
def analyze_dialogue_with_ai(dialogue_text, current_speaker, config_text, performer_names, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 대화 분석기야. 화자({current_speaker})의 말을 분석해. target_person은 {performer_names} 중 하나 또는 None. 기준: {config_text}"),
        ("user", "{dialogue_text}")
    ])
    try:
        structured = llm.with_structured_output(schema=DialogueAnalysis)
        return structured.invoke(prompt.invoke({
            "current_speaker": current_speaker,
            "performer_names": str(performer_names),
            "config_text": config_text, 
            "dialogue_text": dialogue_text
        }))
    except Exception as e:
        print(f"Analyze Error: {e}")
        return None

# 8) Run Case
def run_case_local(case_name, transcript_path, performer_names, config_path, profile_path, out_dir, mode="pair"):
    print(f"Run Case: {case_name}")
    dialogues = parse_transcript(transcript_path)
    print(f"Segments: {len(dialogues)}")
    if not dialogues: return

    config = load_config(config_path)
    # 1) Regex
    regex_result = analyze_text_regex(dialogues, config)
    save_json(regex_result, os.path.join(out_dir, f"{case_name}_{mode}_regex.json"))
    
    # 2) Mapping
    try:
        profile_df = pd.read_csv(profile_path)
        profile_text = profile_df.to_string(index=False)
    except:
        profile_text = str(performer_names)
    
    # Init Solar
    llm = ChatUpstage(api_key=SOLAR_API_KEY, model="solar-1-mini-chat", temperature=0)
    
    mapping_path = os.path.join(out_dir, f"{case_name}_{mode}_speaker_mapping.json")
    mapping_obj = identify_speakers_force_choice(dialogues, profile_text, performer_names, llm, mapping_path)
    
    # 3) Analyze Loop
    csv_out = os.path.join(out_dir, f"{case_name}_{mode}_ai_solar.csv") # renamed output
    results = []
    
    analysis_config_text = json.dumps(config, ensure_ascii=False)
    target_set = set(performer_names)
    
    start_idx = 0
    if os.path.exists(csv_out):
        try:
            results = pd.read_csv(csv_out).to_dict("records")
            start_idx = len(results)
            print(f"Resuming CSV from {start_idx}")
        except: pass
    
    # Use smaller range for demo if needed, but here we run full
    for i in tqdm(range(start_idx, len(dialogues))):
        d = dialogues[i]
        sid = d["speaker"]
        speaker_name = mapping_obj.get(sid, {}).get("name")
        speaker_conf = mapping_obj.get(sid, {}).get("confidence")
        
        if not speaker_name:
            if d["speaker"] in performer_names:
                speaker_name = d["speaker"]
            else:
                speaker_name = d["speaker"]
        
        analysis = analyze_dialogue_with_ai(d["text"], speaker_name, analysis_config_text, performer_names, llm)
        if analysis:
            results.append({
                "speaker": speaker_name,
                "confidence": speaker_conf,
                "text": d["text"],
                "target": analysis.target_person,
                "sentiment": analysis.sentiment,
                "category": analysis.category,
                "summary": analysis.summary
            })
            
        if (len(results)) % 10 == 0:
            pd.DataFrame(results).to_csv(csv_out, index=False, encoding="utf-8-sig")
            
    pd.DataFrame(results).to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"Completed. Saved to {csv_out}")

def main():
    if not SOLAR_API_KEY:
        print("No Solar API Key")
        # return # Try run anyway to see error? No, return.
        return
        
    out_dir = "result2_solar"
    os.makedirs(out_dir, exist_ok=True)
    
    # Copy old mapping if exists to avoid re-identifying cost
    old_mapping = "result2/환연2_해은규민_pair_speaker_mapping.json"
    new_mapping = f"{out_dir}/환연2_해은규민_pair_speaker_mapping.json"
    if os.path.exists(old_mapping) and not os.path.exists(new_mapping):
        print(f"Copying {old_mapping} -> {new_mapping}")
        try:
            shutil.copy(old_mapping, new_mapping)
        except: pass

    run_case_local(
        case_name="환연2_해은규민",
        transcript_path="transcript/환연2_해은규민_수정중.txt",
        performer_names=["성해은", "정규민"],
        config_path="table/all_text_anlst.json",
        profile_path="table/character_profile.csv",
        out_dir=out_dir,
        mode="pair"
    )

if __name__ == "__main__":
    main()
