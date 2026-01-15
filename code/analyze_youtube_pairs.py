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
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as e:
        return {'positive': 0, 'negative': 0, 'neutral': len(comments)}

def process_group_detail(group_name, keyword1, keyword2, target_valid_count=100):
    print(f"\nProcessing Group: {group_name} ('{keyword1}' AND '{keyword2}')")
    print(f"  Target: {target_valid_count} valid videos with comments.")
    
    final_rows = []
    next_page_token = None
    query = f"{keyword1} {keyword2}"
    
    # We maintain a list of seen video IDs to avoid dupes
    seen_ids = set()
    total_searched = 0
    max_search_limit = 500 # Don't search forever
    
    while len(final_rows) < target_valid_count and total_searched < max_search_limit:
        # A. Search Batch
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
        
        total_searched += len(items) # Approximation
        
        if not batch_candidates:
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token: break
            continue

        # B. Get Stats for Batch
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

        # C. Check Validity & Analyze
        for cand in batch_candidates:
            vid = cand['video_id']
            stats = stats_map.get(vid, {'view_count':0, 'like_count':0, 'comment_count':0})
            
            # Initial Check: if stats says >0, it's promising. If 0, try fetching.
            # We want ONLY rows with actual comments > 0
            
            comments_sample = []
            try:
                c_resp = youtube.commentThreads().list(
                    part='snippet', videoId=vid, maxResults=5, textFormat='plainText'
                ).execute()
                c_items = c_resp.get('items', [])
                
                if len(c_items) == 0:
                    # No comments found (or disabled)
                    continue 
                
                # Update stats if needed (correction)
                if stats['comment_count'] == 0:
                    stats['comment_count'] = len(c_items)
                    
                for c in c_items:
                    text = c['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments_sample.append(text)
                    
            except:
                # Disabled comments or error -> Skip this video
                continue
            
            # If we reached here, video is valid
            # Analyze Sentiment
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
    # Group 1: 100 valid
    rows1 = process_group_detail("희두_나연", "희두", "나연", target_valid_count=100)
    all_rows.extend(rows1)
    
    # Group 2: 100 valid
    rows2 = process_group_detail("해은_규민", "해은", "규민", target_valid_count=100)
    all_rows.extend(rows2)
    
    # Save
    df = pd.DataFrame(all_rows)
    cols = ['group_name', 'video_id', 'title', 'view_count', 'like_count', 'comment_count', 
            'positive_ratio', 'negative_ratio', 'neutral_ratio', 'published_at']
            
    output_path = "result/youtube_pairs_detail.csv"
    os.makedirs("result", exist_ok=True)
    
    if not df.empty:
        df = df.sort_values(by='view_count', ascending=False)
        
    df.to_csv(output_path, columns=cols, index=False, encoding='utf-8-sig')
    print(f"\nSaved detailed analysis to {output_path}")
    print(f"Total Rows: {len(df)}")
    if not df.empty:
        print(df[['group_name', 'title', 'comment_count']].head(5))

if __name__ == "__main__":
    main()
