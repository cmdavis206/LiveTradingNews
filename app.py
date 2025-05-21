# --- Imports ---
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import re
import urllib.parse
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
import openai
import feedparser
import firebase_admin
from firebase_admin import credentials, firestore
import json

# --- Streamlit page config ---
st.set_page_config(page_title="Real Time Trader News", layout="wide")

# --- Helper Functions ---
def contains_flagged_word(text, word_list):
    config = st.session_state['config']
    word_list = config.get('keywords', []) + config.get('tickers', [])
    for word in word_list:
        if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
            return True
    return False

def summarize_transcript(text, api_key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Summarize the following YouTube video transcript as a concise, actionable summary for traders. "
            "Focus on market news, trade ideas, or key opinions if mentioned:\n\n"
            f"{text[:6000]}"
        )
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Could not summarize with OpenAI: {e}")

# --- API Keys ---
try:
    YOUTUBE_API_KEY = st.secrets["default"]["YOUTUBE_API_KEY"]
    OPENAI_API_KEY = st.secrets["default"]["OPENAI_API_KEY"]
except KeyError as e:
    st.error(f"Missing API key in secrets: {e}")
    st.stop()

# --- Initialize Firebase Firestore ---
try:
    if not firebase_admin._apps:
        # Construct credentials dictionary from individual fields
        firebase_creds = {
            "type": st.secrets["firebase"]["type"],
            "project_id": st.secrets["firebase"]["project_id"],
            "private_key_id": st.secrets["firebase"]["private_key_id"],
            "private_key": st.secrets["firebase"]["private_key"],
            "client_email": st.secrets["firebase"]["client_email"],
            "client_id": st.secrets["firebase"]["client_id"],
            "auth_uri": st.secrets["firebase"]["auth_uri"],
            "token_uri": st.secrets["firebase"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
        }
        cred = credentials.Certificate(firebase_creds)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    db._client._timeout = 30  # Set timeout to 30 seconds
except Exception as e:
    st.error(f"Failed to initialize Firestore: {e}")
    st.stop()

# --- Load Config from Firestore ---
def load_config():
    try:
        doc_ref = db.collection('config').document('settings')
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            keywords = data.get('keywords', [])
            tickers = data.get('tickers', [])
            rss_feeds = data.get('rss_feeds', [])
            sp500 = data.get('sp500', [])
            return keywords, tickers, rss_feeds, sp500
        else:
            # Default values if Firestore document doesn't exist
            default_keywords = ["buy", "sell", "trade", "market", "stock"]
            default_tickers = ["AAPL", "MSFT", "NVDA", "SPY", "TSLA", "GOOG", "AMZN", "META", "NFLX", "AMD"]
            default_rss_feeds = [
                "https://finance.yahoo.com/news/rss",
                "https://www.cnbc.com/id/100003114/device/rss/rss.html"
            ]
            default_sp500 = ["AAPL", "MSFT", "GOOGL"]
            doc_ref.set({
                'keywords': default_keywords,
                'tickers': default_tickers,
                'rss_feeds': default_rss_feeds,
                'sp500': default_sp500
            })
            return default_keywords, default_tickers, default_rss_feeds, default_sp500
    except Exception as e:
        st.error(f"Error loading config from Firestore: {e}")
        # Fallback to default values
        default_keywords = ["buy", "sell", "trade", "market", "stock"]
        default_tickers = ["AAPL", "MSFT", "NVDA", "SPY", "TSLA", "GOOG", "AMZN", "META", "NFLX", "AMD"]
        default_rss_feeds = [
            "https://finance.yahoo.com/news/rss",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        ]
        default_sp500 = ["AAPL", "MSFT", "GOOGL"]
        return default_keywords, default_tickers, default_rss_feeds, default_sp500

# --- Save Config to Firestore ---
def save_config(keywords, tickers, rss_feeds, sp500):
    try:
        doc_ref = db.collection('config').document('settings')
        doc_ref.set({
            'keywords': keywords,
            'tickers': tickers,
            'rss_feeds': rss_feeds,
            'sp500': sp500
        })
        st.write("Config saved to Firestore successfully.")
    except Exception as e:
        st.error(f"Error saving config to Firestore: {e}")

# Load config with session state
if 'config' not in st.session_state:
    keywords, tickers, rss_feeds, sp500 = load_config()
    st.session_state['config'] = {
        'keywords': keywords,
        'tickers': tickers,
        'rss_feeds': rss_feeds,
        'sp500': sp500
    }
else:
    config = st.session_state['config']
    keywords = config['keywords']
    tickers = config['tickers']
    rss_feeds = config['rss_feeds']
    sp500 = config['sp500']

# Function to check if a video has already been summarized
def check_existing_summary(video_id):
    try:
        doc_ref = db.collection('summaries').document(video_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        st.error(f"Error checking existing summary: {e}")
        return None

# Function to log a new summary
def log_summary(video_id, title, channel_name, published, summary, summarized_by):
    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        doc_ref = db.collection('summaries').document(video_id)
        doc_ref.set({
            'video_id': video_id,
            'title': title,
            'channel_name': channel_name,
            'published': published,
            'summary': summary,
            'timestamp': timestamp,
            'summarized_by': summarized_by
        })
    except Exception as e:
        st.error(f"Error logging summary to Firestore: {e}")

# Function to get all summaries for the log
def get_summary_log():
    try:
        summaries_ref = db.collection('summaries').order_by('timestamp', direction=firestore.Query.DESCENDING)
        docs = summaries_ref.stream()
        summaries = []
        for doc in docs:
            summaries.append(doc.to_dict())
        return pd.DataFrame(summaries)
    except Exception as e:
        st.error(f"Error retrieving summary log: {e}")
        return pd.DataFrame()

# --- Main Tabs ---
tab_dashboard, tab_video_summary, tab_settings = st.tabs(
    ["Dashboard", "Video Summary", "Settings"]
)

# --- Dashboard Tab: News Feed & Chart ---
with tab_dashboard:
    st.title("Real Time Trader News Dashboard")
    st.caption("MVP – News Dashboard for Active Traders")

    # --- Auto-Refresh ---
    st_autorefresh(interval=10000, limit=None, key="news_autorefresh")

    # --- Top Movers ---
    st.subheader("Top Movers Today (S&P 500)")
    try:
        df = yf.download(sp500, period="1d", interval="1d", group_by="ticker", progress=False, threads=False)
    except Exception:
        df = pd.DataFrame()

    movers = []
    for t in sp500:
        try:
            d = df[t] if t in df else df
            if not d.empty:
                open_p = d["Open"].iloc[0]
                close_p = d["Close"].iloc[0]
                pct = ((close_p - open_p) / open_p) * 100
                movers.append((t, pct, close_p))
        except Exception:
            continue

    if movers:
        movers = sorted(movers, key=lambda x: x[1], reverse=True)
        col_gain, col_loss = st.columns(2)
        with col_gain:
            st.markdown("**Top Gainers:**")
            for t, pct, cp in movers[:5]:
                st.write(f"🟢 **{t}** {pct:+.2f}% (Last: {cp:.2f})")
        with col_loss:
            st.markdown("**Top Losers:**")
            for t, pct, cp in movers[-5:][::-1]:
                st.write(f"🔴 **{t}** {pct:+.2f}% (Last: {cp:.2f})")
    else:
        st.info("No data available for movers (maybe market is closed or API rate limit).")

    # --- News Feed & Chart ---
    col1, col2 = st.columns([2, 3], gap="large")
    with col1:
        st.subheader("News Headlines")
        all_news = []
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                if feed.bozo:
                    continue
                for entry in feed.entries[:10]:
                    try:
                        pub_dt = (
                            datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                            if hasattr(entry, "published_parsed") and entry.published_parsed
                            else datetime.now(timezone.utc)
                        )
                    except Exception:
                        pub_dt = datetime.now(timezone.utc)
                    all_news.append({
                        "title": entry.title,
                        "link": entry.link,
                        "published": pub_dt,
                        "source": feed_url
                    })
            except Exception:
                pass

        unique_news = { (item["title"], item["link"]): item for item in all_news }.values()
        sorted_news = sorted(unique_news, key=lambda x: x["published"], reverse=True)

        for item in sorted_news[:50]:
            title = item["title"]
            link = item["link"]
            published = item["published"].strftime("%Y-%m-%d %H:%M:%S")
            source = item["source"]
            flagged = contains_flagged_word(title, keywords) or contains_flagged_word(title, tickers)
            prefix = ":red_circle: **" if flagged else ""
            suffix = "**" if flagged else ""
            st.markdown(f"- {prefix}[{title}]({link}){suffix}  \n*{published}*  \n`{source}`")

    with col2:
        st.subheader("Live Chart Demo – Yahoo Finance")
        demo_tickers = tickers if tickers else ["AAPL", "MSFT", "NVDA", "SPY", "TSLA", "GOOG", "AMZN", "META", "NFLX", "AMD"]
        ticker_search = st.text_input("Search for ticker (e.g., AAPL)", value=demo_tickers[0])

        matching = [t for t in demo_tickers if ticker_search.upper() in t]
        selected = matching[0] if matching else ticker_search.upper()

        intervals = [("1m", "1d"), ("5m", "5d"), ("1d", "6mo")]
        df_chart = pd.DataFrame()
        label = ""

        for interval, period in intervals:
            try:
                df_chart = yf.download(selected, period=period, interval=interval, progress=False)
                if not df_chart.empty:
                    label = f"{interval}, {period}"
                    break
            except Exception:
                continue

        if isinstance(df_chart.columns, pd.MultiIndex):
            df_chart.columns = [col[0] for col in df_chart.columns]

        if not df_chart.empty:
            if interval in ["1m", "5m"]:
                fig = go.Figure(
                    data=[go.Candlestick(
                        x=df_chart.index,
                        open=df_chart["Open"],
                        high=df_chart["High"],
                        low=df_chart["Low"],
                        close=df_chart["Close"],
                        name=selected
                    )]
                )
            else:
                fig = go.Figure(
                    data=[go.Scatter(
                        x=df_chart.index,
                        y=df_chart["Close"],
                        mode="lines+markers",
                        name=selected
                    )]
                )
            fig.update_layout(
                title=f"{selected} – Chart ({label})",
                xaxis_title="Time",
                yaxis_title="Price",
                xaxis_rangeslider_visible=True,
                height=600,
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No chart data available for '{selected}' (market closed, ticker unavailable, or Yahoo blocked your IP).")

# --- Video Summary Tab ---
with tab_video_summary:
    st.header("YouTube Video Summarizer")

    # Predefined list of YouTube channel IDs (you can add more)
    CHANNEL_IDS = [
        "UCymzDnu-l3vZ1fxuqvRePOA",  # The Trading Fraternity
        "UCAuUUnT6oDeKwE6v1NGQxug",  # TED (for testing)
        "UCNjyEXSvYUUCzagFAKmaJ1Q",  # The Rebel Capitalist
        "UCigUBIf-zt_DA6xyOQtq2WA",  # ClearValue Tax
        "UCrXNkk4IESnqU-8GMad2vyA",  # Eurodollar University
        "UCT3EznhW_CNFcfOlyDNTLLw",  # Minority Mindset
        "UCafVgkN5OArXy1nsePKlSYw",  # TastyLive Trending
        "UCpvyOqtEc86X8w8_Se0t4-w",  # George Gammon
        "UCLgJ1HO-7mTOkHZsn0EoKfQ",  # B The Trader
        "UCHhC2FEUxv0OSDrxUybKhcg",  # Geopolitical Futures
    ]

    # Initialize session_state variables
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}
    if "videos" not in st.session_state:
        st.session_state.videos = []
    if "errors" not in st.session_state:
        st.session_state.errors = {}
    if "custom_url_summary" not in st.session_state:
        st.session_state.custom_url_summary = None
    if "custom_url_error" not in st.session_state:
        st.session_state.custom_url_error = None
    if "last_fetch_time" not in st.session_state:
        st.session_state.last_fetch_time = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = "Anonymous"

    # User input for their name
    st.session_state.user_name = st.text_input("Enter your name (for summary logging):", value=st.session_state.user_name)

    # Function to calculate "time ago" string
    def time_ago(timestamp):
        if not timestamp:
            return "Never"
        now = datetime.now(timezone.utc)
        delta = now - timestamp
        seconds = delta.total_seconds()
        if seconds < 60:
            return f"{int(seconds)} seconds ago"
        elif seconds < 3600:
            return f"{int(seconds // 60)} minutes ago"
        elif seconds < 86400:
            return f"{int(seconds // 3600)} hours ago"
        else:
            return f"{int(seconds // 86400)} days ago"

    # Function to save videos and timestamp to Firestore
    def save_videos_to_firestore(videos, fetch_time):
        try:
            doc_ref = db.collection('videos').document('recent_videos')
            doc_ref.set({
                'videos': videos,
                'last_fetch_time': fetch_time.isoformat() if fetch_time else None
            })
        except Exception as e:
            st.error(f"Error saving videos to Firestore: {e}")

    # Function to load videos and timestamp from Firestore
    def load_videos_from_firestore():
        try:
            doc_ref = db.collection('videos').document('recent_videos')
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                videos = data.get('videos', [])
                last_fetch_time_str = data.get('last_fetch_time')
                last_fetch_time = datetime.fromisoformat(last_fetch_time_str).replace(tzinfo=timezone.utc) if last_fetch_time_str else None
                return videos, last_fetch_time
            return [], None
        except Exception as e:
            st.error(f"Error loading videos from Firestore: {e}")
            return [], None

    # Load videos and timestamp from Firestore at startup
    if not st.session_state.videos:
        videos, last_fetch_time = load_videos_from_firestore()
        st.session_state.videos = videos
        st.session_state.last_fetch_time = last_fetch_time

    # Section 1: Fetch videos from predefined channels (Uploads + Live)
    st.subheader("Recent Videos (Last 48 Hours)")
    
    # Display last fetch time if videos exist
    if st.session_state.videos and st.session_state.last_fetch_time:
        fetch_time_str = st.session_state.last_fetch_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        st.write(f"Last fetched: {fetch_time_str} ({time_ago(st.session_state.last_fetch_time)})")
    
    if st.button("Fetch Recent Videos", key="fetch_recent_videos"):
        if not YOUTUBE_API_KEY or not OPENAI_API_KEY:
            st.error("API keys missing. Check your settings tab or Streamlit secrets.")
        else:
            with st.spinner("Fetching recent videos..."):
                st.session_state.videos = []
                st.session_state.last_fetch_time = datetime.now(timezone.utc)  # Store fetch timestamp
                youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=48)
                video_ids_seen = set()

                for channel_id in CHANNEL_IDS:
                    try:
                        channel_res = youtube.channels().list(part='snippet', id=channel_id).execute()
                        if 'items' not in channel_res or not channel_res['items']:
                            st.session_state.errors[channel_id] = f"No channel found for ID {channel_id}."
                            continue
                        channel_name = channel_res['items'][0]['snippet']['title']

                        # Part 1: Fetch videos from the uploads playlist
                        res = youtube.channels().list(part='contentDetails', id=channel_id).execute()
                        if 'items' not in res or not res['items']:
                            st.session_state.errors[channel_id] = f"No channel found for ID {channel_id}."
                            continue
                        uploads_playlist = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']

                        playlist = youtube.playlistItems().list(
                            part='snippet',
                            playlistId=uploads_playlist,
                            maxResults=5
                        ).execute()

                        for item in playlist['items']:
                            published_at = datetime.strptime(item['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                            if published_at >= cutoff_time:
                                video_id = item['snippet']['resourceId']['videoId']
                                if video_id not in video_ids_seen:
                                    video_ids_seen.add(video_id)
                                    video_data = {
                                        'video_id': video_id,
                                        'title': item['snippet']['title'],
                                        'published': item['snippet']['publishedAt'],
                                        'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                                        'channel_id': channel_id,
                                        'channel_name': channel_name,
                                        'source': 'Uploads'
                                    }
                                    st.session_state.videos.append(video_data)

                        # Part 2: Fetch past live streams (VODs) from the "Live" section
                        search_res = youtube.search().list(
                            part='id,snippet',
                            channelId=channel_id,
                            eventType='completed',
                            type='video',
                            order='date',
                            maxResults=5,
                            publishedAfter=cutoff_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        ).execute()

                        if search_res.get('items'):
                            for item in search_res['items']:
                                published_at = datetime.strptime(item['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                if published_at >= cutoff_time:
                                    video_id = item['id']['videoId']
                                    if video_id not in video_ids_seen:
                                        video_ids_seen.add(video_id)
                                        video_data = {
                                            'video_id': video_id,
                                            'title': item['snippet']['title'],
                                            'published': item['snippet']['publishedAt'],
                                            'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                                            'channel_id': channel_id,
                                            'channel_name': channel_name,
                                            'source': 'Live'
                                        }
                                        st.session_state.videos.append(video_data)
                        else:
                            st.session_state.errors[channel_id] = f"No recent live videos found for channel {channel_name} in the last 48 hours."

                    except Exception as e:
                        st.session_state.errors[channel_id] = f"Error fetching videos for channel {channel_id}: {e}"

                # Save videos and timestamp to Firestore
                save_videos_to_firestore(st.session_state.videos, st.session_state.last_fetch_time)

    # Display fetched videos with thumbnails and summarize buttons
    if st.session_state.videos:
        for video in st.session_state.videos:
            vid = video['video_id']
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(video['thumbnail'], use_container_width=True)
            with col2:
                st.subheader(video['title'])
                st.write(f"Channel: {video['channel_name']}")
                st.write(f"Published: {video['published']}")
                st.write(f"Source: {video['source']}")
                st.markdown(f"[Watch on YouTube](https://www.youtube.com/watch?v={vid})")

                # Check if the video has already been summarized
                existing_summary = check_existing_summary(vid)
                if existing_summary:
                    st.info(f"This video has already been summarized by {existing_summary['summarized_by']} at {existing_summary['timestamp']}.")
                    st.success("Existing Summary:")
                    st.write(existing_summary['summary'])
                else:
                    if st.button("Summarize", key=f"summarize_{vid}"):
                        with st.spinner(f"Summarizing {video['title']}..."):
                            try:
                                transcript_data = YouTubeTranscriptApi.get_transcript(
                                    vid,
                                    languages=['en', 'en-US']
                                )
                                text = " ".join(segment['text'] for segment in transcript_data if 'text' in segment)
                            except TranscriptsDisabled:
                                st.session_state.errors[vid] = "Transcripts are disabled for this video."
                                continue
                            except NoTranscriptFound:
                                try:
                                    transcript_list = YouTubeTranscriptApi.list_transcripts(vid)
                                    transcript = next(transcript_list.__iter__(), None)
                                    if not transcript:
                                        raise NoTranscriptFound("No transcripts available for this video.")
                                    transcript_data = transcript.fetch()
                                    text = " ".join(segment['text'] for segment in transcript_data if 'text' in segment)
                                except NoTranscriptFound:
                                    st.session_state.errors[vid] = "No transcripts available for this video."
                                    continue
                            except Exception as e:
                                st.session_state.errors[vid] = f"Could not get transcript: {e}"
                                continue

                            try:
                                summary = summarize_transcript(text, OPENAI_API_KEY)
                                st.session_state.summaries[vid] = summary
                                # Log the summary to Firestore
                                log_summary(vid, video['title'], video['channel_name'], video['published'], summary, st.session_state.user_name)
                                if vid in st.session_state.errors:
                                    del st.session_state.errors[vid]
                            except Exception as e:
                                st.session_state.errors[vid] = str(e)
                                st.session_state.summaries[vid] = "Summarization failed."

            if vid in st.session_state.errors:
                st.error(st.session_state.errors[vid])
            if summary := st.session_state.summaries.get(vid):
                st.success("Summary:")
                st.write(summary)
            st.markdown("---")
    else:
        st.info("Click 'Fetch Recent Videos' to load recent videos from selected channels.")

    # Section 2: Custom URL summarization
    st.subheader("Summarize a Custom YouTube Video")
    custom_url = st.text_input("Enter YouTube URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)", key="custom_url")
    if st.button("Go", key="summarize_custom_url"):
        if not custom_url:
            st.session_state.custom_url_error = "Please enter a YouTube URL."
        else:
            with st.spinner("Summarizing custom video..."):
                try:
                    # Parse the URL to extract the video_id
                    parsed_url = urllib.parse.urlparse(custom_url)
                    video_id = None

                    if parsed_url.netloc in ("www.youtube.com", "youtube.com"):
                        query_params = urllib.parse.parse_qs(parsed_url.query)
                        video_id = query_params.get("v", [None])[0]
                    elif parsed_url.netloc in ("youtu.be", "www.youtu.be"):
                        # Extract video_id from path (e.g., /VIDEO_ID)
                        path = parsed_url.path.strip("/")
                        video_id = path if path else None

                    # Validate the video_id format (11 characters, alphanumeric with - and _)
                    if not video_id or not re.match(r"^[A-Za-z0-9_-]{11}$", video_id):
                        raise ValueError("Invalid YouTube video ID. Please ensure the URL is correct and the video ID is 11 characters long (letters, numbers, underscores, or hyphens).")

                    # Check if the video has already been summarized
                    existing_summary = check_existing_summary(video_id)
                    if existing_summary:
                        st.session_state.custom_url_summary = existing_summary['summary']
                        st.session_state.custom_url_error = None
                        st.info(f"This video has already been summarized by {existing_summary['summarized_by']} at {existing_summary['timestamp']}.")
                    else:
                        transcript_data = None
                        try:
                            transcript_data = YouTubeTranscriptApi.get_transcript(
                                video_id,
                                languages=['en', 'en-US']
                            )
                        except NoTranscriptFound:
                            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                            transcript = next(transcript_list.__iter__(), None)
                            if not transcript:
                                raise NoTranscriptFound("No transcripts available for this video.")
                            transcript_data = transcript.fetch()

                        text = " ".join(segment['text'] for segment in transcript_data if 'text' in segment)
                        summary = summarize_transcript(text, OPENAI_API_KEY)
                        st.session_state.custom_url_summary = summary
                        st.session_state.custom_url_error = None

                        # Fetch metadata for logging
                        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
                        video_details = youtube.videos().list(part='snippet', id=video_id).execute()
                        if 'items' in video_details and video_details['items']:
                            title = video_details['items'][0]['snippet']['title']
                            channel_name = video_details['items'][0]['snippet']['channelTitle']
                            published = video_details['items'][0]['snippet']['publishedAt']
                            log_summary(video_id, title, channel_name, published, summary, st.session_state.user_name)
                        else:
                            st.warning("Video metadata could not be retrieved, but the summary was generated successfully.")

                except TranscriptsDisabled:
                    st.session_state.custom_url_error = "Transcripts are disabled for this video."
                    st.session_state.custom_url_summary = None
                except NoTranscriptFound:
                    st.session_state.custom_url_error = "No transcripts available for this video."
                    st.session_state.custom_url_summary = None
                except Exception as e:
                    st.session_state.custom_url_error = f"Error summarizing custom video: {e}"
                    st.session_state.custom_url_summary = None

    if st.session_state.custom_url_error:
        st.error(st.session_state.custom_url_error)
    if st.session_state.custom_url_summary:
        st.success("Custom Video Summary:")
        st.write(st.session_state.custom_url_summary)

    # Section 3: Summary Log
    st.subheader("Summary Log")
    summary_log = get_summary_log()
    if not summary_log.empty:
        st.dataframe(summary_log)
    else:
        st.info("No summaries have been logged yet.")

# --- Settings Tab ---
with tab_settings:
    st.header("Manage Your Settings")

    # --- Keywords ---
    st.subheader("Keywords")
    # Form for adding a keyword
    with st.form(key="add_keyword_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_keyword = st.text_input("Add a keyword", key="add_keyword_main")
        with col2:
            submit_add_keyword = st.form_submit_button("Add Keyword")
        if submit_add_keyword:
            if new_keyword and new_keyword not in keywords:
                keywords.append(new_keyword)
                save_config(keywords, tickers, rss_feeds, sp500)
                keywords, tickers, rss_feeds, sp500 = load_config()
                st.session_state['config'] = {
                    'keywords': keywords,
                    'tickers': tickers,
                    'rss_feeds': rss_feeds,
                    'sp500': sp500
                }
                st.success(f"Added keyword: {new_keyword}")
                st.rerun()

    # Display current keywords
    st.write("Current Keywords:")
    st.write(", ".join([f"`{kw}`" for kw in keywords]))

    # Form for removing keywords
    with st.form(key="remove_keyword_form"):
        remove_kw = st.multiselect("Remove keyword(s):", options=keywords, key="remove_kw_main")
        submit_remove_keyword = st.form_submit_button("Delete Selected Keywords")
        if submit_remove_keyword and remove_kw:
            keywords = [kw for kw in keywords if kw not in remove_kw]
            save_config(keywords, tickers, rss_feeds, sp500)
            keywords, tickers, rss_feeds, sp500 = load_config()
            st.session_state['config'] = {
                'keywords': keywords,
                'tickers': tickers,
                'rss_feeds': rss_feeds,
                'sp500': sp500
            }
            st.success(f"Removed keywords: {', '.join(remove_kw)}")
            st.rerun()

    st.markdown("---")

    # --- Tickers ---
    st.subheader("Tickers")
    # Form for adding a ticker
    with st.form(key="add_ticker_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_ticker = st.text_input("Add a ticker", key="add_ticker_main")
        with col2:
            submit_add_ticker = st.form_submit_button("Add Ticker")
        if submit_add_ticker:
            if new_ticker and new_ticker.upper() not in tickers:
                tickers.append(new_ticker.upper())
                save_config(keywords, tickers, rss_feeds, sp500)
                keywords, tickers, rss_feeds, sp500 = load_config()
                st.session_state['config'] = {
                    'keywords': keywords,
                    'tickers': tickers,
                    'rss_feeds': rss_feeds,
                    'sp500': sp500
                }
                st.success(f"Added ticker: {new_ticker.upper()}")
                st.rerun()

    # Display current tickers
    st.write("Current Tickers:")
    st.write(", ".join([f"`{t}`" for t in tickers]))

    # Form for removing tickers
    with st.form(key="remove_ticker_form"):
        remove_tkr = st.multiselect("Remove ticker(s):", options=tickers, key="remove_tkr_main")
        submit_remove_ticker = st.form_submit_button("Delete Selected Tickers")
        if submit_remove_ticker and remove_tkr:
            tickers = [t for t in tickers if t not in remove_tkr]
            save_config(keywords, tickers, rss_feeds, sp500)
            keywords, tickers, rss_feeds, sp500 = load_config()
            st.session_state['config'] = {
                'keywords': keywords,
                'tickers': tickers,
                'rss_feeds': rss_feeds,
                'sp500': sp500
            }
            st.success(f"Removed tickers: {', '.join(remove_tkr)}")
            st.rerun()

    st.markdown("---")

    # --- RSS Feeds ---
    st.subheader("RSS Feeds")
    # Form for adding an RSS feed
    with st.form(key="add_feed_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_feed = st.text_input("Add RSS Feed", key="add_feed_main")
        with col2:
            submit_add_feed = st.form_submit_button("Add RSS Feed")
        if submit_add_feed:
            if new_feed and new_feed not in rss_feeds:
                rss_feeds.append(new_feed)
                save_config(keywords, tickers, rss_feeds, sp500)
                keywords, tickers, rss_feeds, sp500 = load_config()
                st.session_state['config'] = {
                    'keywords': keywords,
                    'tickers': tickers,
                    'rss_feeds': rss_feeds,
                    'sp500': sp500
                }
                st.success(f"Added RSS feed: {new_feed}")
                st.rerun()

    # Display current feeds
    st.write("Current Feeds:")
    st.write(", ".join([f"`{f}`" for f in rss_feeds]))

    # Form for removing RSS feeds
    with st.form(key="remove_feed_form"):
        remove_feed = st.multiselect("Remove RSS Feed(s):", options=rss_feeds, key="remove_feed_main")
        submit_remove_feed = st.form_submit_button("Delete Selected Feeds")
        if submit_remove_feed and remove_feed:
            rss_feeds = [f for f in rss_feeds if f not in remove_feed]
            save_config(keywords, tickers, rss_feeds, sp500)
            keywords, tickers, rss_feeds, sp500 = load_config()
            st.session_state['config'] = {
                'keywords': keywords,
                'tickers': tickers,
                'rss_feeds': rss_feeds,
                'sp500': sp500
            }
            st.success(f"Removed RSS feeds: {', '.join(remove_feed)}")
            st.rerun()

    st.markdown("---")