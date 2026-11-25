import os
import re
import requests
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from readability import Document
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """Search the web for recent information, articles, and URLs on a given query."""
    response = tavily.search(query=query, max_results=5)
    results = response.get("results", [])
    if not results:
        return "No search results found."

    out = []
    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("content", "")[:300]
        out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")

    return "\n----\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """Fetch a webpage and extract its main readable article text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        html = response.text

        # 1. Primary extractor: trafilatura
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
        )
        if extracted and len(extracted.strip()) > 200:
            cleaned = re.sub(r"\s+", " ", extracted).strip()
            return cleaned[:5000]

        # 2. Fallback extractor: readability-lxml + BeautifulSoup
        doc = Document(html)
        clean_html = doc.summary()

        soup = BeautifulSoup(clean_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        cleaned = re.sub(r"\s+", " ", text).strip()

        return cleaned[:5000] if cleaned else "No readable content found on this page."

    except Exception as exc:
        return f"Failed to scrape {url}: {exc}"