import re
import os

def fix_transcript(file_path):
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    new_lines = []
    changes = 0
    
    # Suffix to Name mapping
    # Order matters: check longer suffixes first (e.g. -패널들 before -패널)
    suffix_map = {
        '-희두': '남희두',
        '-나연': '이나연',
        '-태이': '김태이',
        '-지연': '이지연',
        '-해은': '성해은',
        '-패널들': '패널',
        '-패널': '패널',
        '-출연자들': '출연자',
        '-출연자': '출연자'
    }

    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line looks like a speaker line (has timestamp and brackets)
        # We specifically look for [SPEAKER_1] to replace it
        if i + 1 < len(lines):
            next_line = lines[i+1].rstrip() # strip newline/spaces from right
            
            matched_suffix = None
            matched_name = None
            
            # Check suffixes
            for suffix, name in suffix_map.items():
                if next_line.strip().endswith(suffix):
                    matched_suffix = suffix
                    matched_name = name
                    break
            
            if matched_suffix:
                # 1. Replace [SPEAKER_1] -> [Name] in current line if present
                if '[SPEAKER_1]' in line:
                    line = line.replace('[SPEAKER_1]', f'[{matched_name}]')
                    # We count this as a speaker change
                
                # 2. Remove suffix from next line
                # Be careful to remove only the suffix at the end
                # We stripped next_line for checking, but we want to modify the original string in the list
                # which might have \n. 
                original_next = lines[i+1]
                # Find the suffix index from the right
                idx = original_next.rfind(matched_suffix)
                if idx != -1:
                    # Remove it and strip trailing spaces before it if natural? 
                    # User asked to remove the suffix characters.
                    # " -희두" might be better to remove than just "-희두" if there's a space?
                    # But user said "글자는 삭제해주고". I will strictly remove the suffix char sequence.
                    # But usually, there is a space before hyphen. "잘지냈어? -희두"
                    # I'll remove the suffix.
                    
                    # Construct new next line
                    # We keep everything up to the suffix, and keep the trailing newline if it was after
                    pre_suffix = original_next[:idx]
                    post_suffix = original_next[idx+len(matched_suffix):]
                    
                    # Clean up trailing whitespace in pre_suffix if it was just before the tag
                    pre_suffix = pre_suffix.rstrip()
                    
                    lines[i+1] = pre_suffix + post_suffix 
                    changes += 1
        
        new_lines.append(line)
        i += 1
        
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
        
    print(f"Complete. {changes} replacements made.")

if __name__ == "__main__":
    target_file = 'transcript/환승연애_희두나연.txt'
    fix_transcript(target_file)
