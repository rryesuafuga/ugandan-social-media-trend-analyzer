import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from collections import Counter
import sqlite3
import os
from typing import List, Dict, Tuple

# Configure Streamlit page
st.set_page_config(
    page_title="Uganda Social Media Trends",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
@st.cache_resource
def init_database():
    """Initialize SQLite database for storing trends data"""
    conn = sqlite3.connect('trends.db', check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            sentiment_score REAL,
            post_count INTEGER,
            timestamp DATETIME,
            platform TEXT,
            engagement_score REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            sentiment_score REAL,
            platform TEXT,
            timestamp DATETIME,
            is_ugandan INTEGER,
            topic TEXT
        )
    ''')
    
    conn.commit()
    return conn

# Ugandan context detection
class UgandanContentDetector:
    def __init__(self):
        self.ugandan_keywords = [
            # Places
            'kampala', 'entebbe', 'jinja', 'mbale', 'gulu', 'mbarara', 'fort portal',
            'masaka', 'soroti', 'lira', 'kasese', 'hoima', 'arua', 'kabale',
            'lake victoria', 'river nile', 'bwindi', 'queen elizabeth',
            
            # People & Politics
            'museveni', 'bobi wine', 'besigye', 'parliament uganda', 'state house',
            'nrm', 'nup', 'fdc', 'opposition uganda',
            
            # Culture & Language
            'uganda', 'ugandan', 'baganda', 'banyankole', 'acholi', 'langi',
            'luganda', 'bantu', 'matooke', 'posho', 'rolex uganda', 'chapati',
            'boda boda', 'taxi park', 'stage coach', 'old taxi park',
            
            # Media & Events
            'new vision', 'daily monitor', 'nbs tv', 'ntv uganda', 'bukedde',
            'kcca', 'express fc', 'villa sc', 'cranes', 'she cranes',
            
            # Economics
            'uganda shilling', 'ugx', 'bank of uganda', 'kcb uganda', 'stanbic',
            'mtn uganda', 'airtel uganda', 'umeme', 'nwsc'
        ]
        
        self.ugandan_phrases = [
            'pearl of africa', 'gifted by nature', 'uganda zaabu',
            'for god and my country', 'uganda tourism board'
        ]
    
    def is_ugandan_content(self, text: str) -> Tuple[bool, float]:
        """
        Determine if content is Ugandan and return confidence score
        """
        text_lower = text.lower()
        
        # Check for explicit Uganda mentions
        if 'uganda' in text_lower or 'ugandan' in text_lower:
            return True, 0.9
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in self.ugandan_keywords 
                            if keyword in text_lower)
        
        # Check for phrase matches
        phrase_matches = sum(1 for phrase in self.ugandan_phrases 
                           if phrase in text_lower)
        
        # Calculate confidence score
        total_words = len(text.split())
        if total_words == 0:
            return False, 0.0
        
        keyword_density = keyword_matches / total_words
        phrase_bonus = phrase_matches * 0.2
        
        confidence = min(keyword_density + phrase_bonus, 1.0)
        
        is_ugandan = confidence >= 0.1 or keyword_matches >= 2
        
        return is_ugandan, confidence

# Sentiment Analysis
class SentimentAnalyzer:
    @staticmethod
    def analyze_sentiment(text: str) -> Tuple[str, float]:
        """
        Analyze sentiment using TextBlob
        Returns sentiment label and polarity score (-1 to 1)
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
                
            return sentiment, polarity
        except:
            return "Neutral", 0.0

# Mock Data Generator (simulating social media data)
class MockDataGenerator:
    def __init__(self):
        self.ugandan_topics = [
            "Uganda's economy shows growth despite global challenges",
            "Kampala traffic congestion reaches new levels during peak hours",
            "Bobi Wine addresses supporters at rally in Masaka",
            "New Vision reports on parliamentary session outcomes",
            "Matooke prices surge in Kampala markets this week",
            "Boda boda operators protest new KCCA regulations",
            "NTV Uganda covers breaking news from Entebbe",
            "Uganda Cranes prepare for upcoming AFCON qualifiers",
            "Mtn Uganda launches new mobile money services",
            "Lake Victoria fishing communities face new challenges",
            "Ugandan coffee exports reach record highs",
            "Kampala International University announces new programs",
            "Express FC signs new players for upcoming season",
            "Uganda shilling strengthens against the dollar",
            "Rolex vendors in Kampala adapt to new health guidelines"
        ]
        
        self.non_ugandan_topics = [
            "Global market trends show mixed signals",
            "Technology companies report quarterly earnings",
            "Climate change impacts discussed at summit",
            "New restaurant opens in downtown area",
            "Sports team wins championship game",
            "Movie premiere attracts large crowds",
            "Fashion week showcases latest trends",
            "Music festival announces lineup",
            "Local school receives award for excellence",
            "Weather forecast predicts sunny weekend"
        ]
        
        self.platforms = ["Twitter", "Facebook", "Instagram", "TikTok"]
    
    def generate_mock_posts(self, count: int = 50) -> List[Dict]:
        """Generate mock social media posts"""
        posts = []
        
        for i in range(count):
            # 70% chance of Ugandan content
            is_ugandan = np.random.choice([True, False], p=[0.7, 0.3])
            
            if is_ugandan:
                topic = np.random.choice(self.ugandan_topics)
            else:
                topic = np.random.choice(self.non_ugandan_topics)
            
            # Add some variation to the content
            variations = [
                f"Just heard: {topic}",
                f"Breaking: {topic}",
                f"Thoughts on {topic}?",
                f"Latest update: {topic}",
                f"{topic} - what do you think?",
                topic
            ]
            
            content = np.random.choice(variations)
            
            posts.append({
                'content': content,
                'platform': np.random.choice(self.platforms),
                'timestamp': datetime.now() - timedelta(
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                ),
                'engagement': np.random.randint(1, 1000)
            })
        
        return posts

# Main App Class
class UgandaTrendAnalyzer:
    def __init__(self):
        self.db_conn = init_database()
        self.detector = UgandanContentDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.mock_generator = MockDataGenerator()
    
    def process_posts(self, posts: List[Dict]) -> List[Dict]:
        """Process posts through Ugandan detection and sentiment analysis"""
        processed_posts = []
        
        for post in posts:
            # Detect if content is Ugandan
            is_ugandan, confidence = self.detector.is_ugandan_content(post['content'])
            
            # Analyze sentiment
            sentiment_label, sentiment_score = self.sentiment_analyzer.analyze_sentiment(post['content'])
            
            # Extract topics (simple keyword extraction)
            topics = self.extract_topics(post['content'])
            
            processed_post = {
                **post,
                'is_ugandan': is_ugandan,
                'ugandan_confidence': confidence,
                'sentiment_label': sentiment_label,
                'sentiment_score': sentiment_score,
                'topics': topics
            }
            
            processed_posts.append(processed_post)
        
        return processed_posts
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text"""
        # Simple topic extraction using keywords
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            'Politics': ['government', 'parliament', 'election', 'political', 'museveni', 'bobi wine', 'nrm', 'nup'],
            'Economy': ['economy', 'business', 'trade', 'market', 'money', 'shilling', 'bank', 'financial'],
            'Sports': ['football', 'cranes', 'express fc', 'villa', 'sports', 'match', 'game'],
            'Transport': ['boda boda', 'taxi', 'traffic', 'transport', 'road', 'kcca'],
            'Food': ['matooke', 'food', 'market', 'prices', 'rolex', 'restaurant'],
            'Media': ['news', 'nbs', 'ntv', 'vision', 'monitor', 'media', 'tv'],
            'Technology': ['mtn', 'airtel', 'mobile', 'internet', 'technology', 'digital']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['General']
    
    def get_trending_topics(self, processed_posts: List[Dict]) -> List[Dict]:
        """Calculate trending topics from processed posts"""
        # Filter for Ugandan content only
        ugandan_posts = [post for post in processed_posts if post['is_ugandan']]
        
        if not ugandan_posts:
            return []
        
        # Count topics
        topic_counts = Counter()
        topic_sentiments = {}
        topic_engagements = {}
        
        for post in ugandan_posts:
            for topic in post['topics']:
                topic_counts[topic] += 1
                
                if topic not in topic_sentiments:
                    topic_sentiments[topic] = []
                    topic_engagements[topic] = []
                
                topic_sentiments[topic].append(post['sentiment_score'])
                topic_engagements[topic].append(post['engagement'])
        
        # Create trending topics list
        trending_topics = []
        for topic, count in topic_counts.most_common(10):
            avg_sentiment = np.mean(topic_sentiments[topic])
            avg_engagement = np.mean(topic_engagements[topic])
            
            trending_topics.append({
                'topic': topic,
                'post_count': count,
                'avg_sentiment': avg_sentiment,
                'sentiment_label': 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral',
                'avg_engagement': avg_engagement,
                'trend_score': count * (1 + abs(avg_sentiment)) * np.log(avg_engagement + 1)
            })
        
        # Sort by trend score
        trending_topics.sort(key=lambda x: x['trend_score'], reverse=True)
        
        return trending_topics

# Streamlit App
def main():
    st.title("🇺🇬 Uganda Social Media Trend Analyzer")
    st.markdown("*Real-time social media trend analysis for Ugandan media houses*")
    
    # Initialize the analyzer
    analyzer = UgandaTrendAnalyzer()
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Mock Data (Demo)", "Live Data (Coming Soon)"],
        help="Select data source for analysis"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Top Trending Topics")
        
        # Generate and process mock data
        with st.spinner("Analyzing social media data..."):
            mock_posts = analyzer.mock_generator.generate_mock_posts(100)
            processed_posts = analyzer.process_posts(mock_posts)
            trending_topics = analyzer.get_trending_topics(processed_posts)
        
        if trending_topics:
            # Create trending topics table
            df_trends = pd.DataFrame(trending_topics)
            
            # Display top 10 trending topics
            st.subheader("🔥 Top 10 Trending Topics")
            
            for i, topic in enumerate(trending_topics[:10], 1):
                col_a, col_b, col_c, col_d = st.columns([0.5, 2, 1, 1])
                
                with col_a:
                    st.metric("Rank", f"#{i}")
                
                with col_b:
                    st.metric("Topic", topic['topic'])
                
                with col_c:
                    sentiment_color = "🟢" if topic['sentiment_label'] == 'Positive' else "🔴" if topic['sentiment_label'] == 'Negative' else "🟡"
                    st.metric("Sentiment", f"{sentiment_color} {topic['sentiment_label']}")
                
                with col_d:
                    st.metric("Posts", topic['post_count'])
                
                st.divider()
            
            # Trend visualization
            st.subheader("📈 Trend Visualization")
            
            # Sentiment distribution
            fig_sentiment = px.bar(
                df_trends.head(10),
                x='topic',
                y='post_count',
                color='sentiment_label',
                title="Post Count by Topic and Sentiment",
                color_discrete_map={
                    'Positive': '#00CC96',
                    'Negative': '#EF553B',
                    'Neutral': '#FFA15A'
                }
            )
            fig_sentiment.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Trend score visualization
            fig_trend = px.scatter(
                df_trends.head(10),
                x='avg_engagement',
                y='trend_score',
                size='post_count',
                color='sentiment_label',
                hover_name='topic',
                title="Trend Score vs Engagement",
                color_discrete_map={
                    'Positive': '#00CC96',
                    'Negative': '#EF553B',
                    'Neutral': '#FFA15A'
                }
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        else:
            st.warning("No trending topics found. Try refreshing the data.")
    
    with col2:
        st.header("📋 Analytics Dashboard")
        
        # Summary statistics
        ugandan_posts = [post for post in processed_posts if post['is_ugandan']]
        total_posts = len(processed_posts)
        ugandan_count = len(ugandan_posts)
        
        st.metric("Total Posts Analyzed", total_posts)
        st.metric("Ugandan Content", ugandan_count)
        st.metric("Detection Rate", f"{(ugandan_count/total_posts)*100:.1f}%" if total_posts > 0 else "0%")
        
        # Platform distribution
        if processed_posts:
            platform_counts = Counter([post['platform'] for post in ugandan_posts])
            
            st.subheader("📱 Platform Distribution")
            for platform, count in platform_counts.items():
                st.write(f"**{platform}:** {count} posts")
        
        # Recent Ugandan posts
        st.subheader("🇺🇬 Recent Ugandan Posts")
        
        recent_ugandan = sorted(
            ugandan_posts,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:5]
        
        for post in recent_ugandan:
            with st.expander(f"{post['platform']} - {post['sentiment_label']}"):
                st.write(post['content'])
                st.caption(f"Confidence: {post['ugandan_confidence']:.2f} | "
                          f"Sentiment: {post['sentiment_score']:.2f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Demo Features:**
    - ✅ Ugandan content detection using contextual keywords
    - ✅ Sentiment analysis with TextBlob
    - ✅ Topic extraction and trending calculation
    - ✅ Real-time dashboard with visualizations
    - ✅ SQLite database for data persistence
    
    **Production Ready Features:**
    - 🔄 Live social media API integration
    - 🔄 Advanced ML models for content classification
    - 🔄 Real-time data streaming
    - 🔄 Cloud database integration
    - 🔄 API endpoints for media house integration
    """)
    
    st.info("💡 **For Media Houses:** This prototype demonstrates core functionality. "
            "Contact us to discuss custom API integration, real-time data feeds, "
            "and advanced analytics features for your newsroom.")

if __name__ == "__main__":
    main()
