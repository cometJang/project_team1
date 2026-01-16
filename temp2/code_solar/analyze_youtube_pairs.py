
import os
import pandas as pd
import json
import time
from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain_upstage import ChatUpstage

# 1. Setup
load_dotenv()
YOUTUBE_API_KEY = os.getenv("GEMINI_API_KEY") # Use Gemini Key for YouTube API
SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")    # Use Solar Key for LLM

if not YOUTUBE_API_KEY:
    print("Error: GEMINI_API_KEY (for YouTube) not found.")
    exit(1)

if not SOLAR_API_KEY:
    print("Error: SOLAR_API_KEY not found in .env")
    # We won't exit, but sentiment analysis will fail if strict.
    # But usually we should exit.
    # exit(1) 

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def analyze_sentiment_simple(comments):
    if not comments:
        return {'positive': 0, 'negative': 0, 'neutral': 0}
        
    try:
        # Solar Model
        llm = ChatUpstage(
            model="solar-1-mini-chat", 
            api_key=SOLAR_API_KEY
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
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as e:
        print(f"Sentiment Analysis Error: {e}")
        return {'positive': 0, 'negative': 0, 'neutral': len(comments)}

def process_group_detail(group_name, keyword1, keyword2, target_valid_count=100):
    print(f"\nProcessing Group: {group_name} ('{keyword1}' AND '{keyword2}')")
    print(f"  Target: {target_valid_count} valid videos with comments.")
    
    final_rows = []
    next_page_token = None
    query = f"{keyword1} {keyword2}"
    
    seen_ids = set()
    total_searched = 0
    max_search_limit = 500 
    
    while len(final_rows) < target_valid_count and total_searched < max_search_limit:
        try:
            search_response = youtube.search().list(
                q=query, part='id,snippet', type='video',
                maxResults=50, order='viewCount', pageToken=next_page_token
            ).execute()
        except Exception as e:
            print(f"  Search Error: {e}")
            break
            
        items = search_response.get('items', [])
        if not items:
            break
            
        batch_candidates = []
        for item in items:
            vid = item['id']['videoId']
            title = item['snippet']['title']
            if vid not in seen_ids and keyword1 in title and keyword2 in title:
                seen_ids.add(vid)
                batch_candidates.append({
                    'video_id': vid,
                    'title': title,
                    'published_at': item['snippet']['publishedAt']
                })
        
        total_searched += len(items)
        
        if not batch_candidates:
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token: break
            continue

        video_ids = [c['video_id'] for c in batch_candidates]
        stats_map = {}
        try:
            resp = youtube.videos().list(part='statistics', id=','.join(video_ids)).execute()
            for item in resp.get('items', []):
                st = item['statistics']
                stats_map[item['id']] = {
                    'view_count': int(st.get('viewCount', 0)),
                    'like_count': int(st.get('likeCount', 0)),
                    'comment_count': int(st.get('commentCount', 0))
                }
        except:
            pass

        for cand in batch_candidates:
            vid = cand['video_id']
            stats = stats_map.get(vid, {'view_count':0, 'like_count':0, 'comment_count':0})
            
            comments_sample = []
            try:
                c_resp = youtube.commentThreads().list(
                    part='snippet', videoId=vid, maxResults=5, textFormat='plainText'
                ).execute()
                c_items = c_resp.get('items', [])
                
                if len(c_items) == 0:
                    continue 
                
                if stats['comment_count'] == 0:
                    stats['comment_count'] = len(c_items)
                    
                for c in c_items:
                    text = c['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments_sample.append(text)
                    
            except:
                continue
            
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
                'title': cand['title'],
                'view_count': stats['view_count'],
                'like_count': stats['like_count'],
                'comment_count': stats['comment_count'],
                'positive_ratio': pos_r,
                'negative_ratio': neg_r,
                'neutral_ratio': neu_r,
                'published_at': cand['published_at']
            }
            final_rows.append(row)
            
            if len(final_rows) >= target_valid_count:
                break
        
        print(f"  Collected {len(final_rows)} valid videos so far...")
        
        if len(final_rows) >= target_valid_count:
            break
            
        next_page_token = search_response.get('nextPageToken')
        if not next_page_token:
            break
            
    print(f"  Finished Group. Total Valid: {len(final_rows)}\n")
    return final_rows

def main():
    all_rows = []
    # Group 1
    rows1 = process_group_detail("희두_나연", "희두", "나연", target_valid_count=100)
    all_rows.extend(rows1)
    
    # Group 2
    rows2 = process_group_detail("해은_규민", "해은", "규민", target_valid_count=100)
    all_rows.extend(rows2)
    
    # Save
    df = pd.DataFrame(all_rows)
    cols = ['group_name', 'video_id', 'title', 'view_count', 'like_count', 'comment_count', 
            'positive_ratio', 'negative_ratio', 'neutral_ratio', 'published_at']
            
    output_path = "result/youtube_pairs_detail_solar.csv"
    os.makedirs("result", exist_ok=True)
    
    if not df.empty:
        df = df.sort_values(by='view_count', ascending=False)
        
    df.to_csv(output_path, columns=cols, index=False, encoding='utf-8-sig')
    print(f"\nSaved detailed analysis to {output_path}")

if __name__ == "__main__":
    main()
