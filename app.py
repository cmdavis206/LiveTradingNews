import streamlit as st
import os
import json
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import re
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

# --- Config Loaders ---
def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def contains_flagged_word(text, word_list):
    for word in word_list:
        if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
            return True
    return False

# --- Setup & Config ---
st.set_page_config(page_title="Real Time Trader News", layout="wide")
st.title("Real Time Trader News")
st.caption("MVP – News Dashboard for Active Traders")

base = os.path.dirname(__file__)
keywords = load_json(os.path.join(base, "config", "keywords.json"))
tickers = load_json(os.path.join(base, "config", "tickers.json"))
rss_feeds = load_json(os.path.join(base, "config", "rss_feeds.json"))
sp500 = load_json(os.path.join(base, "config", "sp500.json"))


# --- Auto-Refresh ---
st_autorefresh(interval=5000, limit=None, key="news_autorefresh")




# --- Top Movers Section Side-by-Side ---
st.subheader("Top Movers Today (S&P 500)")

# Always use your full S&P 500 list from config/sp500.json for movers!
# (Already loaded as `sp500` above)

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



# --- Main Tabs ---
tab_dashboard, tab_settings = st.tabs(["Dashboard", "Settings"])

# --- Dashboard Tab: News Feed & Chart Side by Side ---
with tab_dashboard:
    st.header("Live Trading News – Real Time")

    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        st.subheader("News Headlines")
        import feedparser
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
        if matching:
            selected = matching[0]
        else:
            selected = ticker_search.upper()

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
                    data=[
                        go.Candlestick(
                            x=df_chart.index,
                            open=df_chart["Open"],
                            high=df_chart["High"],
                            low=df_chart["Low"],
                            close=df_chart["Close"],
                            name=selected
                        )
                    ]
                )
            else:
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=df_chart.index,
                            y=df_chart["Close"],
                            mode="lines+markers",
                            name=selected
                        )
                    ]
                )
            fig.update_layout(
                title=f"{selected} – Chart ({label})",
                xaxis_title="Time",
                yaxis_title="Price",
                xaxis_rangeslider_visible=True,
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=15, label="15m", step="minute", stepmode="backward"),
                            dict(count=1, label="1h", step="hour", stepmode="backward"),
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                ),
                height=600,
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No chart data available for '{selected}' (market closed, ticker unavailable, or Yahoo blocked your IP).")

# --- Settings Tab (unchanged, keep as before) ---
# [Keep your settings code for keywords, tickers, rss feeds]


# --- Settings Tab ---
with tab_settings:
    st.header("Manage Your Settings")

    # --- Keywords ---
    st.subheader("Keywords")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_keyword = st.text_input("Add a keyword", key="add_keyword_main")
    with col2:
        if st.button("Add Keyword", key="btn_add_keyword_main"):
            if new_keyword and new_keyword not in keywords:
                keywords.append(new_keyword)
                save_json(os.path.join(base, "config", "keywords.json"), keywords)
                st.experimental_rerun()
    st.write("Current Keywords:")
    st.write(", ".join([f"`{kw}`" for kw in keywords]))
    remove_kw = st.multiselect("Remove keyword(s):", options=keywords, key="remove_kw_main")
    if st.button("Delete Selected Keywords"):
        keywords = [kw for kw in keywords if kw not in remove_kw]
        save_json(os.path.join(base, "config", "keywords.json"), keywords)
        st.experimental_rerun()

    st.markdown("---")

    # --- Tickers ---
    st.subheader("Tickers")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input("Add a ticker", key="add_ticker_main")
    with col2:
        if st.button("Add Ticker", key="btn_add_ticker_main"):
            if new_ticker and new_ticker.upper() not in tickers:
                tickers.append(new_ticker.upper())
                save_json(os.path.join(base, "config", "tickers.json"), tickers)
                st.experimental_rerun()
    st.write("Current Tickers:")
    st.write(", ".join([f"`{t}`" for t in tickers]))
    remove_tkr = st.multiselect("Remove ticker(s):", options=tickers, key="remove_tkr_main")
    if st.button("Delete Selected Tickers"):
        tickers = [t for t in tickers if t not in remove_tkr]
        save_json(os.path.join(base, "config", "tickers.json"), tickers)
        st.experimental_rerun()

    st.markdown("---")

    # --- RSS Feeds ---
    st.subheader("RSS Feeds")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_feed = st.text_input("Add RSS Feed", key="add_feed_main")
    with col2:
        if st.button("Add RSS Feed", key="btn_add_feed_main"):
            if new_feed and new_feed not in rss_feeds:
                rss_feeds.append(new_feed)
                save_json(os.path.join(base, "config", "rss_feeds.json"), rss_feeds)
                st.experimental_rerun()
    st.write("Current Feeds:")
    st.write(", ".join([f"`{f}`" for f in rss_feeds]))
    remove_feed = st.multiselect("Remove RSS Feed(s):", options=rss_feeds, key="remove_feed_main")
    if st.button("Delete Selected Feeds"):
        rss_feeds = [f for f in rss_feeds if f not in remove_feed]
        save_json(os.path.join(base, "config", "rss_feeds.json"), rss_feeds)
        st.experimental_rerun()


