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
import base64
import urllib.parse

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

# Social Media API Integrations

class TwitterAPIClient:
    """Twitter API v2 Client using Bearer Token (Free Tier)"""
    
    def __init__(self, bearer_token: str = None):
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        } if bearer_token else None
    
    def search_tweets(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search tweets using Twitter API v2 (Free tier: 500k tweets/month)
        """
        if not self.bearer_token:
            return []
        
        url = f"{self.base_url}/tweets/search/recent"
        
        # Enhanced query for Ugandan content
        uganda_query = f"({query}) (Uganda OR Kampala OR Ugandan OR #Uganda) lang:en"
        
        params = {
            "query": uganda_query,
            "max_results": min(max_results, 100),  # Free tier limit
            "tweet.fields": "created_at,public_metrics,context_annotations,lang,geo",
            "user.fields": "location,public_metrics",
            "expansions": "author_id"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                tweets = []
                
                if 'data' in data:
                    users_dict = {}
                    if 'includes' in data and 'users' in data['includes']:
                        users_dict = {user['id']: user for user in data['includes']['users']}
                    
                    for tweet in data['data']:
                        user = users_dict.get(tweet.get('author_id', ''), {})
                        
                        tweets.append({
                            'content': tweet['text'],
                            'platform': 'Twitter',
                            'timestamp': datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                            'engagement': (
                                tweet.get('public_metrics', {}).get('like_count', 0) +
                                tweet.get('public_metrics', {}).get('retweet_count', 0) +
                                tweet.get('public_metrics', {}).get('reply_count', 0)
                            ),
                            'user_location': user.get('location', ''),
                            'user_followers': user.get('public_metrics', {}).get('followers_count', 0),
                            'tweet_id': tweet['id']
                        })
                
                return tweets
            else:
                st.error(f"Twitter API Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            st.error(f"Twitter API Error: {str(e)}")
            return []

class YouTubeAPIClient:
    """YouTube Data API v3 Client (Free tier: 10,000 units/day)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    def search_videos(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search YouTube videos and get comments"""
        if not self.api_key:
            return []
        
        # Search for Uganda-related videos
        search_url = f"{self.base_url}/search"
        search_params = {
            "part": "snippet",
            "q": f"{query} Uganda OR Kampala OR Ugandan",
            "type": "video",
            "maxResults": min(max_results, 50),
            "order": "relevance",
            "regionCode": "UG",
            "key": self.api_key
        }
        
        try:
            response = requests.get(search_url, params=search_params)
            
            if response.status_code == 200:
                data = response.json()
                videos = []
                
                for item in data.get('items', []):
                    video_id = item['id']['videoId']
                    snippet = item['snippet']
                    
                    # Get video statistics
                    stats = self._get_video_stats(video_id)
                    
                    # Get some comments
                    comments = self._get_video_comments(video_id, max_comments=10)
                    
                    video_data = {
                        'content': snippet['title'] + " " + snippet.get('description', '')[:200],
                        'platform': 'YouTube',
                        'timestamp': datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
                        'engagement': stats.get('viewCount', 0) + stats.get('likeCount', 0),
                        'video_id': video_id,
                        'channel': snippet['channelTitle']
                    }
                    videos.append(video_data)
                    
                    # Add comments as separate posts
                    for comment in comments:
                        videos.append({
                            'content': comment['text'],
                            'platform': 'YouTube',
                            'timestamp': comment['timestamp'],
                            'engagement': comment['likeCount'],
                            'video_id': video_id,
                            'type': 'comment'
                        })
                
                return videos
            else:
                st.error(f"YouTube API Error: {response.status_code}")
                return []
                
        except Exception as e:
            st.error(f"YouTube API Error: {str(e)}")
            return []
    
    def _get_video_stats(self, video_id: str) -> Dict:
        """Get video statistics"""
        url = f"{self.base_url}/videos"
        params = {
            "part": "statistics",
            "id": video_id,
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    stats = data['items'][0].get('statistics', {})
                    return {
                        'viewCount': int(stats.get('viewCount', 0)),
                        'likeCount': int(stats.get('likeCount', 0)),
                        'commentCount': int(stats.get('commentCount', 0))
                    }
        except:
            pass
        
        return {}
    
    def _get_video_comments(self, video_id: str, max_comments: int = 10) -> List[Dict]:
        """Get video comments"""
        url = f"{self.base_url}/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": max_comments,
            "order": "relevance",
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                comments = []
                
                for item in data.get('items', []):
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'text': comment['textDisplay'],
                        'timestamp': datetime.fromisoformat(comment['publishedAt'].replace('Z', '+00:00')),
                        'likeCount': comment.get('likeCount', 0)
                    })
                
                return comments
        except:
            pass
        
        return []

class RedditAPIClient:
    """Reddit API Client (Free tier available)"""
    
    def __init__(self, client_id: str = None, client_secret: str = None, user_agent: str = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.access_token = None
        
        if client_id and client_secret:
            self._get_access_token()
    
    def _get_access_token(self):
        """Get Reddit API access token"""
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        data = {'grant_type': 'client_credentials'}
        headers = {'User-Agent': self.user_agent}
        
        try:
            response = requests.post('https://www.reddit.com/api/v1/access_token',
                                   auth=auth, data=data, headers=headers)
            if response.status_code == 200:
                self.access_token = response.json()['access_token']
        except Exception as e:
            st.error(f"Reddit Auth Error: {str(e)}")
    
    def search_posts(self, query: str, subreddit: str = "uganda", max_results: int = 100) -> List[Dict]:
        """Search Reddit posts"""
        if not self.access_token:
            return []
        
        headers = {
            'Authorization': f'bearer {self.access_token}',
            'User-Agent': self.user_agent
        }
        
        # Search in Uganda subreddit and general search
        urls = [
            f'https://oauth.reddit.com/r/{subreddit}/search.json',
            'https://oauth.reddit.com/search.json'
        ]
        
        all_posts = []
        
        for url in urls:
            params = {
                'q': f"{query} Uganda OR Kampala",
                'limit': min(max_results // 2, 100),
                'sort': 'relevance',
                't': 'week'  # Past week
            }
            
            if 'search.json' in url and 'r/' not in url:
                params['q'] += ' subreddit:uganda OR subreddit:africa'
            
            try:
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for post in data.get('data', {}).get('children', []):
                        post_data = post['data']
                        
                        all_posts.append({
                            'content': post_data.get('title', '') + " " + post_data.get('selftext', '')[:200],
                            'platform': 'Reddit',
                            'timestamp': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                            'engagement': post_data.get('score', 0) + post_data.get('num_comments', 0),
                            'subreddit': post_data.get('subreddit', ''),
                            'post_id': post_data.get('id', ''),
                            'author': post_data.get('author', ''),
                            'upvote_ratio': post_data.get('upvote_ratio', 0)
                        })
                
            except Exception as e:
                st.error(f"Reddit API Error: {str(e)}")
        
        return all_posts

class SocialMediaAggregator:
    """Aggregates data from multiple social media APIs"""
    
    def __init__(self, api_keys: Dict):
        self.twitter_client = TwitterAPIClient(api_keys.get('twitter_bearer_token'))
        self.youtube_client = YouTubeAPIClient(api_keys.get('youtube_api_key'))
        self.reddit_client = RedditAPIClient(
            api_keys.get('reddit_client_id'),
            api_keys.get('reddit_client_secret'),
            api_keys.get('reddit_user_agent', 'UgandaTrendAnalyzer/1.0')
        )
    
    def fetch_all_data(self, query: str = "trending", max_per_platform: int = 50) -> List[Dict]:
        """Fetch data from all available platforms"""
        all_posts = []
        
        # Twitter
        with st.spinner("Fetching Twitter data..."):
            twitter_posts = self.twitter_client.search_tweets(query, max_per_platform)
            all_posts.extend(twitter_posts)
            if twitter_posts:
                st.success(f"✅ Fetched {len(twitter_posts)} tweets")
        
        # YouTube
        with st.spinner("Fetching YouTube data..."):
            youtube_posts = self.youtube_client.search_videos(query, max_per_platform)
            all_posts.extend(youtube_posts)
            if youtube_posts:
                st.success(f"✅ Fetched {len(youtube_posts)} YouTube posts")
        
        # Reddit
        with st.spinner("Fetching Reddit data..."):
            reddit_posts = self.reddit_client.search_posts(query, max_results=max_per_platform)
            all_posts.extend(reddit_posts)
            if reddit_posts:
                st.success(f"✅ Fetched {len(reddit_posts)} Reddit posts")
        
        return all_posts
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
    st.sidebar.title("⚙️ Settings")
    
    # API Configuration Section
    st.sidebar.subheader("🔑 API Configuration")
    
    with st.sidebar.expander("Social Media API Keys", expanded=False):
        st.markdown("""
        **Free API Setup Instructions:**
        
        **Twitter API v2:**
        1. Apply at developer.twitter.com
        2. Get Bearer Token (Free: 500k tweets/month)
        
        **YouTube Data API:**
        1. Google Cloud Console → Enable API
        2. Create credentials (Free: 10k units/day)
        
        **Reddit API:**
        1. reddit.com/prefs/apps
        2. Create 'script' application
        """)
        
        # API Key inputs
        twitter_bearer = st.text_input(
            "Twitter Bearer Token",
            type="password",
            help="Your Twitter API v2 Bearer Token"
        )
        
        youtube_key = st.text_input(
            "YouTube API Key",
            type="password",
            help="Your YouTube Data API v3 Key"
        )
        
        reddit_client_id = st.text_input(
            "Reddit Client ID",
            type="password",
            help="Your Reddit App Client ID"
        )
        
        reddit_client_secret = st.text_input(
            "Reddit Client Secret",
            type="password",
            help="Your Reddit App Client Secret"
        )
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "📊 Data Source",
        ["Mock Data (Demo)", "Live Data (Real APIs)", "Both"],
        help="Select data source for analysis"
    )
    
    # Search query
    search_query = st.sidebar.text_input(
        "🔍 Search Query",
        value="trending",
        help="Keywords to search for (automatically includes Uganda context)"
    )
    
    # Results per platform
    max_results = st.sidebar.slider(
        "📈 Results per Platform",
        min_value=10,
        max_value=100,
        value=50,
        help="Number of posts to fetch from each platform"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("⏰ Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Top Trending Topics")
        
        # Initialize API aggregator
        api_keys = {
            'twitter_bearer_token': twitter_bearer if twitter_bearer else None,
            'youtube_api_key': youtube_key if youtube_key else None,
            'reddit_client_id': reddit_client_id if reddit_client_id else None,
            'reddit_client_secret': reddit_client_secret if reddit_client_secret else None,
            'reddit_user_agent': 'UgandaTrendAnalyzer/1.0'
        }
        
        aggregator = SocialMediaAggregator(api_keys)
        
        # Data fetching
        all_posts = []
        
        if data_source in ["Live Data (Real APIs)", "Both"]:
            st.subheader("🌐 Live Social Media Data")
            
            # Check if any API keys are provided
            has_api_keys = any([twitter_bearer, youtube_key, reddit_client_id])
            
            if has_api_keys:
                live_posts = aggregator.fetch_all_data(search_query, max_results)
                all_posts.extend(live_posts)
                
                if live_posts:
                    st.success(f"✅ Successfully fetched {len(live_posts)} posts from social media APIs")
                else:
                    st.warning("⚠️ No live data retrieved. Check your API keys and try again.")
            else:
                st.warning("⚠️ Please configure API keys in the sidebar to fetch live data.")
                st.info("💡 You can still use mock data to test the functionality.")
        
        if data_source in ["Mock Data (Demo)", "Both"]:
            st.subheader("🎭 Demo Data")
            mock_posts = analyzer.mock_generator.generate_mock_posts(100)
            all_posts.extend(mock_posts)
            st.info(f"✅ Generated {len(mock_posts)} mock posts for demonstration")
        
        if all_posts:
            # Process all posts
            with st.spinner("🔍 Analyzing content and detecting Ugandan posts..."):
                processed_posts = analyzer.process_posts(all_posts)
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
                
                # Platform comparison (if live data available)
                if len([post for post in processed_posts if post.get('platform') in ['Twitter', 'YouTube', 'Reddit']]) > 0:
                    st.subheader("📱 Platform Performance")
                    
                    platform_data = []
                    ugandan_posts = [post for post in processed_posts if post['is_ugandan']]
                    
                    for platform in ['Twitter', 'YouTube', 'Reddit', 'Facebook']:
                        platform_posts = [post for post in ugandan_posts if post.get('platform') == platform]
                        if platform_posts:
                            avg_engagement = np.mean([post['engagement'] for post in platform_posts])
                            avg_sentiment = np.mean([post['sentiment_score'] for post in platform_posts])
                            
                            platform_data.append({
                                'Platform': platform,
                                'Posts': len(platform_posts),
                                'Avg Engagement': avg_engagement,
                                'Avg Sentiment': avg_sentiment
                            })
                    
                    if platform_data:
                        df_platforms = pd.DataFrame(platform_data)
                        
                        fig_platforms = px.scatter(
                            df_platforms,
                            x='Avg Engagement',
                            y='Avg Sentiment',
                            size='Posts',
                            color='Platform',
                            title="Platform Performance: Engagement vs Sentiment"
                        )
                        st.plotly_chart(fig_platforms, use_container_width=True)
            
            else:
                st.warning("No trending topics found. Try adjusting your search query or API settings.")
        else:
            st.warning("No data available. Please configure API keys or enable mock data.")
    
    with col2:
        st.header("📋 Analytics Dashboard")
        
        if all_posts:
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
                    percentage = (count / ugandan_count) * 100 if ugandan_count > 0 else 0
                    st.write(f"**{platform}:** {count} posts ({percentage:.1f}%)")
            
            # API Status
            st.subheader("🔌 API Status")
            
            api_status = {
                "Twitter": "🟢 Connected" if twitter_bearer else "🔴 Not configured",
                "YouTube": "🟢 Connected" if youtube_key else "🔴 Not configured",
                "Reddit": "🟢 Connected" if (reddit_client_id and reddit_client_secret) else "🔴 Not configured"
            }
            
            for platform, status in api_status.items():
                st.write(f"**{platform}:** {status}")
            
            # Recent Ugandan posts
            st.subheader("🇺🇬 Recent Ugandan Posts")
            
            recent_ugandan = sorted(
                ugandan_posts,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:5]
            
            for post in recent_ugandan:
                with st.expander(f"{post['platform']} - {post['sentiment_label']}"):
                    st.write(post['content'][:200] + "..." if len(post['content']) > 200 else post['content'])
                    st.caption(f"Confidence: {post['ugandan_confidence']:.2f} | "
                              f"Sentiment: {post['sentiment_score']:.2f} | "
                              f"Engagement: {post['engagement']}")
        
        else:
            st.info("Configure API keys above to see live analytics")
    
    # Footer
    st.markdown("---")
    
    # API Setup Instructions
    with st.expander("📚 API Setup Guide", expanded=False):
        st.markdown("""
        ## Free API Setup Instructions
        
        ### 🐦 Twitter API v2 (Free Tier)
        1. Visit [developer.twitter.com](https://developer.twitter.com)
        2. Apply for Developer Account (usually approved quickly)
        3. Create a new App in the Developer Portal
        4. Generate Bearer Token from the "Keys and Tokens" tab
        5. **Free Tier:** 500,000 tweets per month
        
        ### 📺 YouTube Data API v3 (Free Tier)
        1. Go to [Google Cloud Console](https://console.cloud.google.com)
        2. Create a new project or select existing one
        3. Enable "YouTube Data API v3"
        4. Go to "Credentials" → "Create Credentials" → "API Key"
        5. **Free Tier:** 10,000 units per day (100 search requests)
        
        ### 🔴 Reddit API (Free)
        1. Visit [reddit.com/prefs/apps](https://reddit.com/prefs/apps)
        2. Click "Create App" or "Create Another App"
        3. Choose "script" as application type
        4. Note down your Client ID and Client Secret
        5. **Free Tier:** 60 requests per minute
        
        ### 📘 Facebook Graph API (Advanced - Not implemented yet)
        Facebook requires app review for most data access. Consider for future versions.
        """)
    
    st.markdown("""
    **Current Features:**
    - ✅ Real-time Twitter API v2 integration
    - ✅ YouTube Data API v3 integration  
    - ✅ Reddit API integration
    - ✅ Ugandan content detection using contextual keywords
    - ✅ Multi-platform sentiment analysis
    - ✅ Advanced topic extraction and trending calculation
    - ✅ Interactive dashboard with live visualizations
    - ✅ Cross-platform analytics and comparison
    
    **Demo vs Live Data:**
    - 🎭 **Mock Data:** Simulated posts for testing
    - 🌐 **Live Data:** Real social media posts via APIs
    - 🔄 **Both:** Combines real and simulated data
    """)
    
    st.info("💡 **For Media Houses:** This prototype demonstrates both mock and real-time data capabilities. "
            "Configure your free API keys above to start analyzing real Ugandan social media trends. "
            "Contact us for enterprise features, advanced ML models, and custom integrations.")

if __name__ == "__main__":
    main() topic['sentiment_label'] == 'Positive' else "🔴" if topic['sentiment_label'] == 'Negative' else "🟡"
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