import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
import logging
import warnings
from urllib.parse import urlparse
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict"
PROJECT_ROOT = Path("L:/Important/MCA/Mini Project/fake_news_detection")

st.set_page_config(
    page_title="📰 Fake News Detector",
    page_icon="🕵️",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.title("About")
st.sidebar.info(
    "Detect fake vs. real news using fine-tuned BERT & RoBERTa on WELFake."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with 🤗 Transformers & FastAPI")

# --- Utility ---
def extract_text_from_url(url: str) -> dict:
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        p = urlparse(url)
        if not p.netloc:
            return {"success": False, "error": "Invalid URL"}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.extract()
        title = soup.title.get_text().strip() if soup.title else ""
        content = ""
        for sel in ("article", "main", ".entry-content", ".article-body"):
            node = soup.select_one(sel)
            if node:
                content = node.get_text()
                break
        if not content:
            paras = soup.find_all("p")
            content = " ".join(p.get_text() for p in paras)
        combined = f"{title} [SEP] {content}".strip()
        combined = re.sub(r"\s+", " ", combined)
        if len(combined) < 50:
            return {"success": False, "error": "Extracted text too short."}
        return {"success": True, "title": title, "combined": combined}
    except Exception as e:
        return {"success": False, "error": str(e)}

def call_api(text: str, title: str, model_type: str) -> dict:
    payload = {"text": text, "title": title, "model_type": model_type, "explain": True}
    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

# --- Main UI ---
st.title("📰 Fake News Detector")
mode = st.radio("Input mode", ["Text", "URL"], horizontal=True)
model_type = st.selectbox("Model type", ["ensemble", "bert-base-uncased", "roberta-base"])

input_text = ""
input_title = ""

if mode == "Text":
    input_text = st.text_area("Enter news text", height=200)
    if input_text.strip():
        with st.spinner("Analyzing text..."):
            result = call_api(input_text, "", model_type)
elif mode == "URL":
    url = st.text_input("Enter article URL")
    if url.strip():
        with st.spinner("Extracting content..."):
            ext = extract_text_from_url(url)
        if not ext["success"]:
            st.error(ext["error"])
        else:
            input_title = ext["title"]
            input_text = ext["combined"]
            with st.spinner("Analyzing extracted content..."):
                result = call_api(input_text, input_title, model_type)

# Display results if available
if input_text.strip():
    st.markdown("### 🎯 Prediction")
    pred = result["prediction"].upper()
    if pred == "REAL":
        st.success(f"**{pred}**")
    elif pred == "FAKE":
        st.error(f"**{pred}**")
    else:
        st.warning(f"**{pred}**")

    st.write(f"**Model used:** {result['model_used']}")
    st.write(f"**Processing time:** {result['processing_time']:.2f}s")
    st.write(f"**Timestamp:** {result['timestamp']}")

    st.markdown("### 📊 Probabilities")
    df_probs = pd.DataFrame.from_dict(result["probabilities"], orient="index", columns=["Probability"])
    st.bar_chart(df_probs)

    if result.get("explanation"):
        st.markdown("### 🔎 Explanation")
        st.json(result["explanation"])

    st.markdown("### 📋 Input Stats")
    stats = result.get("input_stats", {})
    st.metric("Text length", f"{stats.get('text_length',0)} chars")
    st.metric("Title length", f"{stats.get('title_length',0)} chars")
    st.metric("Word count", f"{stats.get('word_count',0)} words")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888;'>"
    "Fine-tuned on WELFake – BERT & RoBERTa Ensemble"
    "</div>",
    unsafe_allow_html=True
)