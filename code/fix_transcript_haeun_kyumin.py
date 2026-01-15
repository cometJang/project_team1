import re
import os

target_file = 'transcript/환연2_해은규민_수정중.txt'

def get_fullname(name_suffix):
    mapping = {
        '해은': '성해은',
        '규민': '정규민',
        '나연': '이나연',
        '희두': '남희두',
        '태이': '김태이',
        '지연': '이지연',
        '나언': '박나언',
        '현규': '정현규',
        '지수': '김지수',
        '원빈': '박원빈',
        '패널': '패널'
    }
    return mapping.get(name_suffix, name_suffix)

def process_transcript(target_file):
    with open(target_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    # Header format: SPEAKER_X HH:MM:SS or HH:MM:SS
    # Capturing regex
    header_pattern = re.compile(r'^(?:(SPEAKER_\d+)\s+)?(\d{2}:\d{2}:\d{2})')
    
    # Mode boundaries
    boundary_1 = "00:32:10"
    boundary_2_start = "00:32:25"
    boundary_2_end = "01:01:34"
    boundary_3_start = "01:01:36"

    current_time_str = "00:00:00"
    last_header_index = -1
    
    # Store lines in a mutable list to update headers retroactively
    lines_buffer = lines[:] # Copy

    for i, line in enumerate(lines_buffer):
        line = line.rstrip() # Keep indentation if any, remove eol
        if not line:
            continue
            
        match = header_pattern.match(line)
        if match:
            # It is a header
            speaker_part = match.group(1) # e.g. SPEAKER_1 or None
            time_part = match.group(2)    # e.g. 00:32:10
            current_time_str = time_part
            last_header_index = i
            
            # Mode 1: <= 00:32:10
            if current_time_str <= boundary_1:
                # Replace specific IDs
                if speaker_part == 'SPEAKER_1':
                    new_header = f"[성해은] {time_part}"
                    lines_buffer[i] = new_header + '\n'
                elif speaker_part == 'SPEAKER_3':
                    new_header = f"[정규민] {time_part}"
                    lines_buffer[i] = new_header + '\n'
                elif speaker_part == 'SPEAKER_4':
                    new_header = f"[이나연] {time_part}"
                    lines_buffer[i] = new_header + '\n'
                # Else keep as is (e.g. SPEAKER_2, or just usage of time)
            
            # Mode 2: 00:32:25 <= time <= 01:01:34 -> Do nothing
            
            # Mode 3: time >= 01:01:36 -> Prepare for suffix extraction (handled in dialogue lines)
            continue
        
        # If not header, it is dialogue (or empty text)
        # Check Mode 3 suffix logic
        if current_time_str >= boundary_3_start:
            # Check for suffix like -해은 or -규민 at the end
            # Regex: look for -Name at end of string
            suffix_match = re.search(r'-(해은|규민|나연|희두|태이|지연|나언|현규|지수|원빈|패널)\s*$', line)
            if suffix_match:
                name_suffix = suffix_match.group(1)
                full_name = get_fullname(name_suffix)
                
                # 1. Update the last header
                if last_header_index != -1:
                    # Retrieve the original time from the header line
                    # We need to preserve the time.
                    # The header line in buffer might have been modified or original.
                    # But in Mode 3 we haven't modified it yet.
                    
                    # Read current header line
                    header_line = lines_buffer[last_header_index]
                    # Extract time again to be safe
                    h_match = header_pattern.match(header_line.strip())
                    if h_match:
                        h_time = h_match.group(2)
                        # Replace header with [Name] Time
                        lines_buffer[last_header_index] = f"[{full_name}] {h_time}\n"
                
                # 2. Remove suffix from dialogue
                # Remove the match part
                new_dialogue = line[:suffix_match.start()].rstrip()
                lines_buffer[i] = new_dialogue + '\n'

    # Save
    with open(target_file, 'w', encoding='utf-8') as f:
        f.writelines(lines_buffer)
    print(f"Processed {target_file}")

if __name__ == "__main__":
    process_transcript(target_file)
