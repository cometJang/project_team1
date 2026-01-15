import os
import pandas as pd
import json
import time
from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Setup
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Error: GEMINI_API_KEY not found.")
    exit(1)

youtube = build('youtube', 'v3', developerKey=API_KEY)

def analyze_sentiment_simple(comments):
    """
    Analyzes sentiment for a list of comments using Gemini.
    Returns counts: {'positive': x, 'negative': y, 'neutral': z}
    """
    if not comments:
        return {'positive': 0, 'negative': 0, 'neutral': 0}
        
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0, 
            google_api_key=API_KEY
        )
        
        comments_text = "\n".join([f"- {c}" for c in comments])
        prompt_str = (
            "Analyze the sentiment of these YouTube comments. "
            "Return ONLY a JSON with counts of positive, negative, neutral. "
            "Example: {\"positive\": 1, \"negative\": 0, \"neutral\": 0}\n\n"
            f"{comments_text}"
        )
        
        response = llm.invoke(prompt_str)
        content = response.content.strip()
        
        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as e:
        # print(f"  AI Error: {e}")
        return {'positive': 0, 'negative': 0, 'neutral': len(comments)}

def process_group_detail(group_name, keyword1, keyword2, target_count=100):
    print(f"\nProcessing Group: {group_name} ('{keyword1}' AND '{keyword2}')")
    
    # 1. Search Videos
    video_items = []
    next_page_token = None
    query = f"{keyword1} {keyword2}"
    
    print(f"  Searching...")
    while len(video_items) < target_count:
        try:
            search_response = youtube.search().list(
                q=query, part='id,snippet', type='video',
                maxResults=50, order='viewCount', pageToken=next_page_token
            ).execute()
            
            for item in search_response.get('items', []):
                title = item['snippet']['title']
                vid = item['id']['videoId']
                # Strict Filtering
                if keyword1 in title and keyword2 in title:
                    video_items.append({
                        'video_id': vid,
                        'title': title,
                        'published_at': item['snippet']['publishedAt'],
                        'group_name': group_name
                    })
            
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            print(f"  Search Error: {e}")
            break
            
    # Remove duplicates
    unique_items = {v['video_id']: v for v in video_items}.values()
    video_items = list(unique_items)
    print(f"  Found {len(video_items)} unique videos.")
    
    if not video_items:
        return []

    # 2. Get Stats (Chunked)
    video_ids = [v['video_id'] for v in video_items]
    stats_map = {}
    
    chunk_size = 50
    for i in range(0, len(video_ids), chunk_size):
        chunk = video_ids[i:i+chunk_size]
        try:
            resp = youtube.videos().list(part='statistics', id=','.join(chunk)).execute()
            for item in resp.get('items', []):
                st = item['statistics']
                stats_map[item['id']] = {
                    'view_count': int(st.get('viewCount', 0)),
                    'like_count': int(st.get('likeCount', 0)),
                    'comment_count': int(st.get('commentCount', 0))
                }
        except Exception as e:
            print(f"  Stats Error: {e}")

    # 3. Process Each Video (Comments + Sentiment)
    final_rows = []
    print(f"  Analyzing individual videos (fetching comments & sentiment)...")
    
    for i, v in enumerate(video_items):
        vid = v['video_id']
        stats = stats_map.get(vid, {'view_count':0, 'like_count':0, 'comment_count':0})
        
        # Fetch Comments (Sample 3)
        comments_sample = []
        try:
            c_resp = youtube.commentThreads().list(
                part='snippet', videoId=vid, maxResults=3, textFormat='plainText'
            ).execute()
            for c in c_resp.get('items', []):
                text = c['snippet']['topLevelComment']['snippet']['textDisplay']
                comments_sample.append(text)
        except:
            pass # Comments disabled or error

        # Sentiment Analysis
        pos_r, neg_r, neu_r = 0.0, 0.0, 0.0
        if comments_sample:
            counts = analyze_sentiment_simple(comments_sample)
            total = sum(counts.values()) or 1
            pos_r = round(counts.get('positive', 0)/total, 2)
            neg_r = round(counts.get('negative', 0)/total, 2)
            neu_r = round(counts.get('neutral', 0)/total, 2)
        
        row = {
            'group_name': group_name,
            'video_id': vid,
            'title': v['title'],
            'view_count': stats['view_count'],
            'like_count': stats['like_count'],
            'comment_count': stats['comment_count'],
            'positive_ratio': pos_r,
            'negative_ratio': neg_r,
            'neutral_ratio': neu_r,
            'published_at': v['published_at']
        }
        final_rows.append(row)
        
        if (i+1) % 10 == 0:
            print(f"    Processed {i+1}/{len(video_items)} videos...")
            
    return final_rows

def main():
    all_rows = []
    
    # Group 1
    rows1 = process_group_detail("희두_나연", "희두", "나연", target_count=70) # Limit slightly for speed
    all_rows.extend(rows1)
    
    # Group 2
    rows2 = process_group_detail("해은_규민", "해은", "규민", target_count=70)
    all_rows.extend(rows2)
    
    # Save
    df = pd.DataFrame(all_rows)
    cols = ['group_name', 'video_id', 'title', 'view_count', 'like_count', 'comment_count', 
            'positive_ratio', 'negative_ratio', 'neutral_ratio', 'published_at']
            
    output_path = "table/youtube_pairs_detail.csv"
    os.makedirs("table", exist_ok=True)
    
    # Sort by views desc
    if not df.empty:
        df = df.sort_values(by='view_count', ascending=False)
        
    df.to_csv(output_path, columns=cols, index=False, encoding='utf-8-sig')
    print(f"\nSaved detailed analysis to {output_path}")
    print(f"Total Rows: {len(df)}")
    if not df.empty:
        print(df[['group_name', 'title', 'view_count', 'positive_ratio']].head(5))

if __name__ == "__main__":
    main()
