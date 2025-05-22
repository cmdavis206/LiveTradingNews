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
import time
import pytz

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

def to_pdt(dt):
    """Convert UTC datetime to PDT."""
    if dt is None:
        return None
    pdt_tz = pytz.timezone("America/Los_Angeles")
    return dt.astimezone(pdt_tz)

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
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    db = firestore.client()
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
    try:
        keywords, tickers, rss_feeds, sp500 = load_config()
        st.session_state['config'] = {
            'keywords': keywords,
            'tickers': tickers,
            'rss_feeds': rss_feeds,
            'sp500': sp500
        }
        st.write("Config loaded from Firestore.")
    except Exception as e:
        st.error(f"Error loading config: {e}")
        # Fallback to defaults
        st.session_state['config'] = {
            'keywords': ["buy", "sell", "trade", "market", "stock"],
            'tickers': ["AAPL", "MSFT", "NVDA", "SPY", "TSLA", "GOOG", "AMZN", "META", "NFLX", "AMD"],
            'rss_feeds': ["https://finance.yahoo.com/news/rss", "https://www.cnbc.com/id/100003114/device/rss/rss.html"],
            'sp500': ["AAPL", "MSFT", "GOOGL"]
        }
        st.write("Using default config.")
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
        st.error(f"Error checking summary for video {video_id}: {e}")
        return None

# Function to log a new summary with retry
def log_summary(video_id, title, channel_name, published, summary, summarized_by):
    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        doc_ref = db.collection('summaries').document(video_id)
        for attempt in range(3):  # Retry up to 3 times
            try:
                doc_ref.set({
                    'video_id': video_id,
                    'title': title,
                    'channel_name': channel_name,
                    'published': published,
                    'summary': summary,
                    'timestamp': timestamp,
                    'summarized_by': summarized_by
                })
                st.write(f"Summary saved for video {video_id}.")
                return
            except Exception as e:
                st.warning(f"Retry {attempt + 1}/3: Error saving summary for video {video_id}: {e}")
                time.sleep(1)
        st.error(f"Failed to save summary for video {video_id} after 3 attempts.")
    except Exception as e:
        st.error(f"Error logging summary for video {video_id}: {e}")

# Function to preload summaries for videos
def preload_summaries(videos, max_videos=20):
    summaries = st.session_state.get('summaries', {})
    for video in videos[:max_videos]:  # Limit to max_videos to control reads
        vid = video['video_id']
        if vid not in summaries:
            existing_summary = check_existing_summary(vid)
            if existing_summary:
                summaries[vid] = existing_summary['summary']
    st.session_state.summaries = summaries

# --- Main Tabs ---
# Initialize tab index in session state
if 'tab_index' not in st.session_state:
    st.session_state.tab_index = 0

# Create tabs without key parameter
tab_dashboard, tab_video_summary, tab_settings = st.tabs(["Dashboard", "Video Summary", "Settings"])

# --- Dashboard Tab ---
with tab_dashboard:
    st.session_state.tab_index = 0
    st.title("Real Time Trader News Dashboard")
    st.caption("MVP – News Dashboard for Active Traders")

    # --- Auto-Refresh (scoped to Dashboard) ---
    st_autorefresh(interval=30000, limit=None, key=f"news_autorefresh_{int(time.time())}")  # 30 seconds, unique key

    # --- Top Movers ---
    st.subheader("Top Movers Today (S&P 500)")
    try:
        df = yf.download(sp500, period="1d", interval="1d", group_by="ticker", progress=False, threads=False, auto_adjust=False)
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
        # --- News Headlines with Pagination ---
        st.subheader("News Headlines")

        # Initialize session state for news and pagination
        if 'news_items' not in st.session_state:
            st.session_state.news_items = []
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        if 'last_news_reset' not in st.session_state:
            st.session_state.last_news_reset = datetime.now(timezone.utc)

        # Reset news_items every 10 minutes to ensure freshness
        now = datetime.now(timezone.utc)
        if (now - st.session_state.last_news_reset).total_seconds() >= 600:  # 10 minutes
            st.session_state.news_items = []
            st.session_state.last_news_reset = now

        # Fetch new articles
        new_articles = []
        existing_items = {(item["title"], item["link"]) for item in st.session_state.news_items}
        for feed_url in rss_feeds:
            try:
                # Force fresh fetch with no caching
                feed = feedparser.parse(feed_url, request_headers={'Cache-Control': 'no-cache'})
                if feed.bozo:
                    st.warning(f"Invalid RSS feed: {feed_url}")
                    continue
                for entry in feed.entries:  # Process all entries
                    try:
                        pub_dt = (
                            datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                            if hasattr(entry, "published_parsed") and entry.published_parsed
                            else now
                        )
                        item_key = (entry.title, entry.link)
                        if item_key not in existing_items:
                            new_articles.append({
                                "title": entry.title,
                                "link": entry.link,
                                "published": pub_dt,
                                "source": feed_url
                            })
                            existing_items.add(item_key)
                    except Exception:
                        continue
            except Exception as e:
                st.error(f"Error parsing RSS feed {feed_url}: {e}")
                continue

        # Update news_items if new articles found
        if new_articles:
            st.session_state.news_items.extend(new_articles)
            # Deduplicate by title and link
            unique_news = { (item["title"], item["link"]): item for item in st.session_state.news_items }.values()
            # Sort by published date (newest first) and limit to 100
            st.session_state.news_items = sorted(unique_news, key=lambda x: x["published"], reverse=True)[:100]
            st.write(f"Updated headlines at {to_pdt(now).strftime('%Y-%m-%d %H:%M:%S %Z')} ({len(new_articles)} new)")

        # Pagination settings
        items_per_page = 10
        total_items = len(st.session_state.news_items)
        total_pages = min((total_items + items_per_page - 1) // items_per_page, 10)  # Up to 10 pages
        current_page = st.session_state.current_page

        # Ensure current_page is valid
        if current_page < 1:
            current_page = 1
        elif current_page > total_pages:
            current_page = total_pages
        st.session_state.current_page = current_page

        # Calculate slice for current page
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        page_items = st.session_state.news_items[start_idx:end_idx]

        # Display page indicator
        st.write(f"Page {current_page} of {total_pages} ({total_items} headlines)")

        # Display headlines for current page
        if page_items:
            for item in page_items:
                title = item["title"]
                link = item["link"]
                published = to_pdt(item["published"]).strftime("%Y-%m-%d %H:%M:%S %Z")
                source = item["source"]
                flagged = contains_flagged_word(title, keywords) or contains_flagged_word(title, tickers)
                prefix = ":red_circle: **" if flagged else ""
                suffix = "**" if flagged else ""
                st.markdown(f"- {prefix}[{title}]({link}){suffix}  \n*{published}*  \n`{source}`")
        else:
            st.info("No news headlines available.")

        # Pagination controls
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.button("Previous", disabled=(current_page == 1), key="prev_page"):
                st.session_state.current_page = current_page - 1
                st.rerun()
        with col_next:
            if st.button("Next", disabled=(current_page == total_pages), key="next_page"):
                st.session_state.current_page = current_page + 1
                st.rerun()

        # Optional: Page number buttons
        st.write("Go to page:")
        page_cols = st.columns(total_pages)
        for i in range(total_pages):
            with page_cols[i]:
                if st.button(str(i + 1), key=f"page_{i+1}", disabled=(current_page == i + 1)):
                    st.session_state.current_page = i + 1
                    st.rerun()

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
                df_chart = yf.download(selected, period=period, interval=interval, progress=False, auto_adjust=False)
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
    st.session_state.tab_index = 1
    st.header("YouTube Video Summarizer")

    # Predefined list of YouTube channel IDs
    CHANNEL_IDS = [
        "UCymzDnu-l3vZ1fxuqvRePOA",  # The Trading Fraternity
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

    # Load videos only if not cached
    if 'videos' not in st.session_state or not st.session_state.videos:
        st.session_state.videos, st.session_state.last_fetch_time = load_videos_from_firestore()

    # Preload summaries for videos
    if st.session_state.videos and not st.session_state.summaries:
        preload_summaries(st.session_state.videos, max_videos=20)

    # Section 1: Fetch videos from predefined channels (Uploads + Live)
    st.subheader("Recent Videos (Last 48 Hours)")
    
    # Display last fetch time if videos exist
    if st.session_state.videos and st.session_state.last_fetch_time:
        fetch_time_pdt = to_pdt(st.session_state.last_fetch_time)
        fetch_time_str = fetch_time_pdt.strftime("%Y-%m-%d %H:%M:%S %Z")
        st.write(f"Last fetched: {fetch_time_str} ({time_ago(st.session_state.last_fetch_time)})")

    if st.button("Fetch Recent Videos", key="fetch_recent_videos"):
        if not YOUTUBE_API_KEY or not OPENAI_API_KEY:
            st.error("API keys missing. Check your settings tab or Streamlit secrets.")
        else:
            with st.spinner("Fetching recent videos..."):
                st.session_state.videos = []
                st.session_state.summaries = {}  # Clear summaries to sync with new videos
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
                # Preload summaries for new videos
                preload_summaries(st.session_state.videos, max_videos=20)

    # Display fetched videos with thumbnails and summarize buttons
    if st.session_state.videos:
        for video in st.session_state.videos:  # Show all videos
            vid = video['video_id']
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(video['thumbnail'], use_container_width=True)
                with col2:
                    st.subheader(video['title'])
                    st.write(f"Channel: {video['channel_name']}")
                    published_dt = datetime.strptime(video['published'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    st.write(f"Published: {to_pdt(published_dt).strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    st.write(f"Source: {video['source']}")
                    st.markdown(f"[Watch on YouTube](https://www.youtube.com/watch?v={vid})")

                    # Use form to isolate button actions
                    with st.form(key=f"video_form_{vid}"):
                        col_check, col_summarize = st.columns(2)
                        with col_check:
                            check_submitted = st.form_submit_button("Check Summary")
                        with col_summarize:
                            summarize_submitted = st.form_submit_button("Summarize")

                        if check_submitted:
                            st.write(f"Checking summary for video ID: {vid}")
                            existing_summary = check_existing_summary(vid)
                            if existing_summary:
                                st.session_state.summaries[vid] = existing_summary['summary']  # Update cache
                                summary_dt = datetime.strptime(existing_summary['timestamp'], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                                st.info(f"Summarized by {existing_summary['summarized_by']} at {to_pdt(summary_dt).strftime('%Y-%m-%d %H:%M:%S %Z')}.")
                            else:
                                st.info("No existing summary found.")

                        if summarize_submitted:
                            with st.spinner(f"Summarizing {video['title']}..."):
                                try:
                                    transcript_data = YouTubeTranscriptApi.get_transcript(vid, languages=['en', 'en-US'])
                                    text = " ".join(segment['text'] for segment in transcript_data if 'text' in segment)
                                    summary = summarize_transcript(text, OPENAI_API_KEY)
                                    st.session_state.summaries[vid] = summary  # Cache locally
                                    log_summary(vid, video['title'], video['channel_name'], video['published'], summary, st.session_state.user_name)
                                except Exception as e:
                                    st.error(f"Could not summarize video: {e}")

                    # Display summary if available
                    if vid in st.session_state.summaries:
                        st.success("Summary:")
                        st.write(st.session_state.summaries[vid])

                if vid in st.session_state.errors:
                    st.error(st.session_state.errors[vid])
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
                        summary_dt = datetime.strptime(existing_summary['timestamp'], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        st.info(f"This video has already been summarized by {existing_summary['summarized_by']} at {to_pdt(summary_dt).strftime('%Y-%m-%d %H:%M:%S %Z')}.")
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

# --- Settings Tab ---
with tab_settings:
    st.session_state.tab_index = 2
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