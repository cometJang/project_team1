
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 1. Load ENV
load_dotenv(".env")
DB_URL = os.getenv("SUPABASE_DB_URL")

if not DB_URL:
    print("Error: SUPABASE_DB_URL not found in .env")
    exit(1)

# 2. Setup SQLAlchemy Engine
try:
    engine = create_engine(DB_URL)
    print("✅ Connected to Supabase Database")
except Exception as e:
    print(f"❌ Connection Failed: {e}")
    exit(1)

# 3. Define Files to Upload
# Format: (File Path, Table Name)
files_to_upload = [
    # Table folder
    ("table/character_profile.csv", "character_profile"),
    ("table/message_log.csv", "message_log"),
    ("table/episode_event_log.csv", "episode_event_log"),
    
    # Result folder (Solar excluded as requested)
    ("result/환연2_해은규민_pair_ai.csv", "haeun_kyumin_pair_ai"),
    ("result/환연2_희두나연_pair_ai.csv", "heedu_nayeon_pair_ai"),
    ("result/youtube_pairs_detail.csv", "youtube_pairs_detail")
]

# 4. Upload Loop
for file_path, table_name in files_to_upload:
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path} (Skipping)")
        continue
        
    try:
        print(f"📤 Uploading {file_path} -> Table [{table_name}]...")
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Upload to DB (replace if exists)
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        
        print(f"   ✅ Success! ({len(df)} rows)")
        
    except Exception as e:
        print(f"   ❌ Error uploading {file_path}: {e}")

print("\n🎉 All tasks completed.")
