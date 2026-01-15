import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional

# 1. Environment Setup
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") 

# Check if API Key exists
if not API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

# 2. Database Setup
Base = declarative_base()

class YoutubeVideoMetrics(Base):
    __tablename__ = 'youtube_video_metrics'
    
    video_id = Column(String, primary_key=True)
    view_count = Column(Integer)
    like_count = Column(Integer)
    comment_count = Column(Integer)
    positive_ratio = Column(Float, nullable=True) # To be updated after analysis
    negative_ratio = Column(Float, nullable=True)
    neutral_ratio = Column(Float, nullable=True)

class YoutubeCommentsAnalysis(Base):
    __tablename__ = 'youtube_comments_analysis'
    
    comment_id = Column(String, primary_key=True)
    video_id = Column(String)
    comment_text = Column(Text)
    sentiment_label = Column(String) # positive, negative, neutral
    sentiment_score = Column(Float)  # 0.0 to 1.0 (or -1 to 1)
    dominant_emotion = Column(String)
    summary_reason = Column(Text)
    type = Column(String) # e.g., 'top_level' or 'reply'
    like_count = Column(Integer)
    publishedAt = Column(DateTime)

# Create DB Engine (SQLite)
# Ensure the directory exists
os.makedirs("table", exist_ok=True)
DB_PATH = "sqlite:///table/youtube_analysis.db"
engine = create_engine(DB_PATH)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# 3. YouTube API Client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# 4. Analysis with Gemini (LangChain)
class CommentSentiment(BaseModel):
    sentiment_label: str = Field(description="positive, negative, or neutral")
    sentiment_score: float = Field(description="Score between 0.0 (very negative) and 1.0 (very positive)")
    dominant_emotion: str = Field(description="Main emotion keywords like 'joy', 'anger', 'sadness', 'admiration'")
    summary_reason: str = Field(description="Brief reason for the sentiment")

def analyze_comment_with_ai(comment_text):
    """
    Analyzes a single comment using Gemini API.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=API_KEY)
        
        parser = PydanticOutputParser(pydantic_object=CommentSentiment)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a specialized sentiment analyzer for YouTube comments. Analyze the following comment and provide structured output.\n{format_instructions}"),
            ("user", "{comment_text}")
        ])
        
        chain = prompt | llm | parser
        result = chain.invoke({
            "comment_text": comment_text,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        # Return fallback/neutral result on error
        return CommentSentiment(
            sentiment_label="neutral",
            sentiment_score=0.5,
            dominant_emotion="unknown",
            summary_reason="Analysis failed"
        )

# 5. Data Collection Functions
def collect_video_data(video_id, max_comments=300):
    print(f"Collecting data for video: {video_id}")

    # 1. Video Metrics
    try:
        response = youtube.videos().list(
            part='statistics',
            id=video_id
        ).execute()

        if not response['items']:
            print(f"Video {video_id} not found.")
            return

        stats = response['items'][0]['statistics']
        session.merge(
            YoutubeVideoMetrics(
                video_id=video_id,
                view_count=int(stats.get('viewCount', 0)),
                like_count=int(stats.get('likeCount', 0)),
                comment_count=int(stats.get('commentCount', 0))
            )
        )
        session.commit()
    except Exception as e:
        print(f"Error metrics: {e}")
        return

    # 2. Comments (pagination ON)
    comments_data = []
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText'
        )

        while request and len(comments_data) < max_comments:
            response = request.execute()

            for item in response['items']:
                snippet = item['snippet']['topLevelComment']['snippet']
                comments_data.append({
                    'id': item['id'],
                    'text': snippet['textDisplay'],
                    'like_count': snippet['likeCount'],
                    'published_at': snippet['publishedAt'],
                    'type': 'top_level'
                })

            request = youtube.commentThreads().list_next(request, response)
    except Exception as e:
        print(f"Pagination/Fetch Error (might have reached end): {e}")

    print(f"Fetched {len(comments_data)} comments")

    # 3. Sentiment Analysis
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

    for c in comments_data:
        # Check if analyzed
        if session.query(YoutubeCommentsAnalysis).filter_by(comment_id=c['id']).first():
            continue

        analysis = analyze_comment_with_ai(c['text'])
        dt = datetime.strptime(c['published_at'], "%Y-%m-%dT%H:%M:%SZ")

        session.add(
            YoutubeCommentsAnalysis(
                comment_id=c['id'],
                video_id=video_id,
                comment_text=c['text'],
                sentiment_label=analysis.sentiment_label,
                sentiment_score=analysis.sentiment_score,
                dominant_emotion=analysis.dominant_emotion,
                summary_reason=analysis.summary_reason,
                type=c['type'],
                like_count=c['like_count'],
                publishedAt=dt
            )
        )

        if analysis.sentiment_label in sentiment_counts:
            sentiment_counts[analysis.sentiment_label] += 1
            
        # Optional: Commit periodically or check limit
    
    session.commit()

    # 4. Ratio Update
    total = sum(sentiment_counts.values())
    if total > 0:
        video = session.query(YoutubeVideoMetrics).filter_by(video_id=video_id).first()
        # Update logic: we should probably count ALL comments in DB for this video, not just this batch
        # But for now sticking to batch logic or simplified
        video.positive_ratio = round(sentiment_counts['positive'] / total, 3)
        video.negative_ratio = round(sentiment_counts['negative'] / total, 3)
        video.neutral_ratio = round(sentiment_counts['neutral'] / total, 3)
        session.commit()
    print(f"Updated ratios based on {total} new analyses.")

if __name__ == "__main__":
    # Example Usage: Replace with the actual Video ID you want to analyze
    # Example ID: 'jNQXAC9IVRw' (Me at the zoo)
    target_video_id = "jNQXAC9IVRw" # Hardcoded for test
    print(f"Analyzing Video ID: {target_video_id}")
    if target_video_id:
        collect_video_data(target_video_id, max_comments=10) # Testing with small number
        print("Done. Check table/youtube_analysis.db")
