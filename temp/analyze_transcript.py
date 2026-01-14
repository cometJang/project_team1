import json
import re
from collections import defaultdict

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_transcript(transcript_path):
    """
    Parses the transcript file.
    Assumes format:
    SPEAKER_XX HH:MM:SS
    Text
    
    or
    HH:MM:SS
    Text
    
    Returns a list of dicts: {'speaker': '...', 'text': '...'}
    """
    dialogues = []
    current_speaker = "UNKNOWN"
    current_text = []
    
    # Regex for Speaker + Time: SPEAKER_19 00:00:00
    speaker_time_pattern = re.compile(r'^(SPEAKER_\d+)\s+(\d{2}:\d{2}:\d{2})$')
    # Regex for Time only: 00:00:08 (implies continuation of previous speaker or just a timestamp)
    time_only_pattern = re.compile(r'^(\d{2}:\d{2}:\d{2})$')
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check patterns
        speaker_match = speaker_time_pattern.match(line)
        time_match = time_only_pattern.match(line)
        
        if speaker_match:
            # Save previous dialogue if exists
            if current_text:
                dialogues.append({
                    'speaker': current_speaker,
                    'text': " ".join(current_text)
                })
                current_text = []
            
            # Start new dialogue
            current_speaker = speaker_match.group(1)
            # Timestamp is speaker_match.group(2) - ignore for now
            
        elif time_match:
            # If time only, it might be a new segment but same speaker, 
            # or just a timestamp separator. 
            # We'll treat it as a break. Save previous text.
            if current_text:
                dialogues.append({
                    'speaker': current_speaker,
                    'text': " ".join(current_text)
                })
                current_text = []
            # Keep current speaker
            
        else:
            # Text content
            # Sometimes parsing leaves line numbers or noise, assuming clean txt from conversion
            current_text.append(line)
            
    # Add last entry
    if current_text:
        dialogues.append({
            'speaker': current_speaker,
            'text': " ".join(current_text)
        })
        
    return dialogues

def analyze_text(dialogues, config):
    """
    Analyzes dialogues based on config.
    """
    
    # 1. Flatten config keywords
    # categories: 'positive', 'negative', 'mention_x', 'initiative'
    keyword_map = {}
    
    # Emotion Polarity
    if 'emotion_polarity' in config:
        if 'positive' in config['emotion_polarity']:
            keyword_map['positive'] = config['emotion_polarity']['positive']['keywords']
        if 'negative' in config['emotion_polarity']:
            keyword_map['negative'] = config['emotion_polarity']['negative']['keywords']
            
    # Attention
    if 'attention' in config:
        if 'mention_x' in config['attention']:
            keyword_map['mention_x'] = config['attention']['mention_x']['keywords']
        # mention_other handling (Names)
        # We start with a predefined list of names common in the show if config enables reference type
        # Ideally we'd extract this dynamically, but hardcoding provided context is safer for now.
        if 'mention_other' in config['attention']:
             keyword_map['mention_other'] = ["해은", "규민", "나연", "희두", "원빈", "지수", "태희", "지연", "나언", "현규", "지현"]

    # Initiative
    if 'initiative' in config:
        keyword_map['initiative'] = config['initiative']['keywords']
        
    # Results container
    # total_counts = {'positive': 0, ...}
    # speaker_stats = {'SPEAKER_X': {'positive': 0, ...}}
    
    total_counts = defaultdict(int)
    speaker_stats = defaultdict(lambda: defaultdict(int))
    
    for entry in dialogues:
        speaker = entry['speaker']
        text = entry['text']
        
        for category, keywords in keyword_map.items():
            for kw in keywords:
                if kw in text:
                    count = text.count(kw) # Count occurrences? or just 1 per text block? 
                    # "Quantitative analysis" usually counts occurrences or presence. 
                    # Let's count occurrences to be precise.
                    total_counts[category] += count
                    speaker_stats[speaker][category] += count
                    
    return total_counts, speaker_stats

def main():
    config_path = 'table/all_text_anlst.json'
    transcript_path = 'transcript/환연2_1화_20화_요약.txt'
    output_path = 'table/analysis_result.json'
    
    print(f"Loading config from {config_path}...")
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    print(f"Parsing transcript from {transcript_path}...")
    try:
        dialogues = parse_transcript(transcript_path)
        print(f"Parsed {len(dialogues)} dialogue segments.")
    except Exception as e:
        print(f"Error parsing transcript: {e}")
        return

    print("Analyzing text...")
    total_counts, speaker_stats = analyze_text(dialogues, config)
    
    # Print Summary
    print("\n=== Analysis Summary ===")
    for cat, count in total_counts.items():
        print(f"  {cat}: {count}")
        
    # Output to JSON
    result = {
        "summary": total_counts,
        "by_speaker": speaker_stats
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()
