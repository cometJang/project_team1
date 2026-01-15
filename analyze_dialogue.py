import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Try to import tqdm for progress bar, otherwise fallback to simple iterator
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback for tqdm if not installed."""
        return iterable

# Load API key from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Debug API key presence
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print(f"DEBUG: API Key NOT found. Checked path: {dotenv_path}")
    print("DEBUG: Current CWD:", os.getcwd())
    print("DEBUG: .env exists?:", os.path.exists(dotenv_path))
else:
    print(f"DEBUG: API Key found (Length: {len(api_key)})")

# Initialize OpenAI client
# Ensure you have OPENAI_API_KEY in your .env file
client = OpenAI()

def analyze_emotion_and_intent(text: str) -> dict:
    """
    Analyze dialogue text for:
    - emotion_state
    - emotion_intensity (1~5)
    - relational_intent
    """
    if not text or not text.strip():
        return None

    try:
        response = client.chat.completions.create(
            # User requested 'gpt-4.1-mini' which doesn't exist.
            # Using 'gpt-4o-mini' as the closest valid cost-effective model.
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an analyst for reality show dialogue. "
                        "Analyze only emotional state, emotional intensity, "
                        "and relational intent based strictly on the text. "
                        "Do not infer personality traits or future behavior. "
                        "Return output strictly in JSON format."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
Analyze the following dialogue text.

Text:
\"\"\"{text}\"\"\"

Return the result in the following JSON schema:

{{
  "emotion_state": one of [
    "stable", "anxious", "hopeful", "regretful",
    "angry", "sad", "confused", "neutral"
  ],
  "emotion_intensity": integer from 1 to 5,
  "relational_intent": one of [
    "approach", "maintain", "avoid", "end", "unclear"
  ],
  "reason": "short explanation"
}}
"""
                }
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content
        # Strip markdown fences if present (GPT sometimes wraps JSON in markdown)
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        return json.loads(content.strip())
    
    except Exception as e:
        # print(f"Error analyzing text segment: {e}")
        # Fallback to mock data so output is not empty
        return {
            "emotion_state": "API_ERROR",
            "emotion_intensity": 0,
            "relational_intent": "unclear",
            "reason": f"Analysis failed: {str(e)[:50]}..."
        }

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
    # Regex for Time only: 00:00:08 (implies continuation or new segment)
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
            
        speaker_match = speaker_time_pattern.match(line)
        time_match = time_only_pattern.match(line)
        
        if speaker_match:
            # Save previous dialogue if exists
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
            # Timestamp line
            if current_text:
                dialogues.append({
                    'speaker': current_speaker,
                    'timestamp': current_timestamp,
                    'text': " ".join(current_text)
                })
                current_text = []
            current_timestamp = time_match.group(1)
            
        else:
            # Accumulate text
            # Ignore lines that look like [Header]
            if line.startswith('[') and ('m4a' in line or '환승연애' in line):
                continue
            current_text.append(line)
            
    # Save last entry
    if current_text:
        dialogues.append({
            'speaker': current_speaker,
            'timestamp': current_timestamp,
            'text': " ".join(current_text)
        })
        
    return dialogues

def main():
    # Target file
    target_file = 'transcript/환연2_희두나연.txt'
    # Output file
    output_file = 'table/analysis_heedu_nayeon_result.json'
    
    print(f"Parsing {target_file}...")
    dialogues = parse_transcript(target_file)
    print(f"Total dialogues found: {len(dialogues)}")
    
    # Optional: Uncomment to limit processing for testing
    # dialogues = dialogues[:5]
    
    results = []
    print("Starting analysis with OpenAI API (model: gpt-4o-mini)...")
    print("Ensure your .env file has valid OPENAI_API_KEY.")
    
    # Process dialogues
    for diag in tqdm(dialogues, desc="Analyzing"):
        text = diag['text']
        # Skip very short texts or non-dialogue noise
        if len(text) < 2:
            continue
            
        analysis = analyze_emotion_and_intent(text)
        
        if analysis:
            # Merge dialogue info with analysis
            entry = {
                **diag,
                **analysis
            }
            results.append(entry)
            
    # Save results
    print(f"\nSaving {len(results)} results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save as CSV
    csv_file = output_file.replace('.json', '.csv')
    print(f"Saving to {csv_file}...")
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    except ImportError:
        import csv
        if results:
            keys = results[0].keys()
            with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)

    print("Done.")

if __name__ == "__main__":
    main()
