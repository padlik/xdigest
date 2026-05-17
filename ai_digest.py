# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31",
#   "openai>=1.30",
#   "python-dotenv>=1.0",
#   "sqlite-vec>=0.1.6",
# ]
# ///
"""
AI Tweet Digest — retrieval + analysis + indexing pipeline
===========================================================
Searches X (Twitter) for AI/LLM topics via Grok x_search, enriches each
tweet via FxTwitter (VxTwitter fallback), analyses results with any
OpenAI-compatible LLM, and optionally indexes everything into SQLite + vectors.

No intermediate files — all data flows in memory.

Run (no venv needed):
    uv run ai_digest.py                            # JSON → stdout
    uv run ai_digest.py --format md                # Markdown → stdout
    uv run ai_digest.py --format md -o out.md      # Markdown → file
    uv run ai_digest.py -o data.json               # JSON → file
    uv run ai_digest.py --format md --db           # index into ai_digest.db
    uv run ai_digest.py --format md --db my.db     # index into custom DB path

── Output formats ────────────────────────────────────────────────────────────
  json (default)   Structured JSON: { generated_at, topics, tweet_count,
                   tweets: [...], analysis: "...markdown text..." }
  md               Full Markdown digest with [N] citations as hyperlinks
                   and a Tweet References appendix.

── Indexing (--db) ───────────────────────────────────────────────────────────
  SQLite database with FTS5 full-text search + sqlite-vec vector similarity.
  Schema: digests, sections (with embeddings), tweet_refs.
  See index_reference.md for full schema and query examples.

── Embedding model selection ─────────────────────────────────────────────────
  EMBED_PROVIDER=openai   → OPENAI_API_KEY   model: text-embedding-3-small
  EMBED_PROVIDER=ollama   → (no key)         model: nomic-embed-text
  EMBED_PROVIDER=openai   → OPENAI_API_KEY   model: text-embedding-3-small
  Override: EMBED_MODEL=my-embed-model  EMBED_BASE_URL=https://...

  Falls back to PROVIDER/LLM_* settings when EMBED_* not set.
  Embeddings are skipped (NULL) if the embed call fails — digest still saved.

── LLM provider selection ────────────────────────────────────────────────────
  PROVIDER=anthropic  → ANTHROPIC_API_KEY   default model: claude-sonnet-4-6
  PROVIDER=openai     → OPENAI_API_KEY      default model: gpt-4o
  PROVIDER=xai        → XAI_API_KEY         default model: grok-4-1-fast
  PROVIDER=mistral    → MISTRAL_API_KEY     default model: mistral-large-latest
  PROVIDER=together   → TOGETHER_API_KEY    default model: meta-llama/Llama-3-70b-chat-hf
  PROVIDER=ollama     → (no key)            default model: llama3.2
  PROVIDER=lmstudio   → (no key)            default model: local-model

  Override: LLM_BASE_URL  LLM_API_KEY  LLM_MODEL

NOTE: All progress/log output goes to stderr; stdout is always clean output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import struct
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import requests
import sqlite_vec
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Logging — always stderr so stdout stays clean
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

# ── Retrieval ─────────────────────────────────────────────────────────────────
XAI_API_KEY   = os.getenv("XAI_API_KEY", "")
XAI_MODEL     = "grok-4-1-fast"
XAI_RESPONSES = "https://api.x.ai/v1/responses"

FXTWITTER_API = "https://api.fxtwitter.com"
VXTWITTER_API = "https://api.vxtwitter.com"

REQUEST_DELAY = 0.5
MAX_TWEETS    = 30

SEARCH_TOPICS = [
        # 110: LLM Latest -> Focuses on technical releases and evaluations with links
    "(LLM OR 'large language model') (weights OR benchmark OR 'context window' OR fine-tune) filter:links -is:reply min_faves:15",

    # 111: MCP -> Focuses on newly shipped open-source servers and tools
    "('Model Context Protocol' OR 'MCP server' OR 'MCP tool') (github OR open-source OR shipped) filter:links",

    # 112: AI Agent Framework -> Focuses on production code, architecture breakdowns, and GitHub repos
    "('AI agent' OR 'agentic') (langgraph OR crewai OR autogen OR production) filter:links min_faves:10 -is:reply",

    # 113: Claude/GPT/Gemini -> Filters out tech-bro hype; targets actual documentation and launch posts
    "(Claude OR 'GPT-5' OR Gemini OR DeepSeek or Kimi or Qwen) (release OR API OR documentation OR 'now available') filter:links min_faves:20 -is:reply",

    # 114: Research Papers -> Targets deep-dives and paper summaries with direct arXiv links
    "(arXiv OR 'research paper') (AI OR LLM OR 'deep learning') (breakdown OR thread OR summary) filter:links min_faves:25"
]

# ── LLM presets ───────────────────────────────────────────────────────────────
_LLM_PRESETS: dict[str, dict] = {
    "anthropic": {
        "base_url":    "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model":       "claude-sonnet-4-6",
    },
    "openai": {
        "base_url":    "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model":       "gpt-4o",
    },
    "xai": {
        "base_url":    "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "model":       "grok-4-1-fast",
    },
    "mistral": {
        "base_url":    "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "model":       "mistral-large-latest",
    },
    "together": {
        "base_url":    "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "model":       "meta-llama/Llama-3-70b-chat-hf",
    },
    "ollama": {
        "base_url":    "http://localhost:11434/v1",
        "api_key_env": None,
        "model":       "llama3.2",
    },
    "lmstudio": {
        "base_url":    "http://localhost:1234/v1",
        "api_key_env": None,
        "model":       "local-model",
    },
}

# ── Embedding presets ─────────────────────────────────────────────────────────
_EMBED_PRESETS: dict[str, dict] = {
    "openai": {
        "base_url":    "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model":       "text-embedding-3-small",
    },
    "ollama": {
        "base_url":    "http://localhost:11434/v1",
        "api_key_env": None,
        "model":       "nomic-embed-text",
    },
    "lmstudio": {
        "base_url":    "http://localhost:1234/v1",
        "api_key_env": None,
        "model":       "text-embedding-nomic-embed-text-v1.5",
    },
}


def _resolve_llm_config() -> tuple[str, str, str]:
    preset   = _LLM_PRESETS.get(os.getenv("PROVIDER", "anthropic").lower(),
                                  _LLM_PRESETS["anthropic"])
    base_url = os.getenv("LLM_BASE_URL") or preset["base_url"]
    key_env  = preset.get("api_key_env")
    api_key  = os.getenv("LLM_API_KEY") or (os.getenv(key_env) if key_env else None) or "ollama"
    model      = os.getenv("LLM_MODEL") or preset["model"]
    return base_url, api_key, model


def _resolve_embed_config() -> tuple[str, str, str]:
    """
    Return (base_url, api_key, model) for embeddings.
    EMBED_PROVIDER / EMBED_* env vars take precedence; falls back to LLM config.
    """
    embed_provider = os.getenv("EMBED_PROVIDER", os.getenv("PROVIDER", "openai")).lower()
    preset = _EMBED_PRESETS.get(embed_provider, _EMBED_PRESETS["openai"])

    base_url = (os.getenv("EMBED_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or preset["base_url"])
    key_env  = preset.get("api_key_env")
    api_key  = (os.getenv("EMBED_API_KEY")
                or os.getenv("LLM_API_KEY")
                or (os.getenv(key_env) if key_env else None)
                or "ollama")
    model = os.getenv("EMBED_MODEL") or preset["model"]
    return base_url, api_key, model


LLM_BASE_URL, LLM_API_KEY, LLM_MODEL = _resolve_llm_config()
EMBED_BASE_URL, EMBED_API_KEY, EMBED_MODEL            = _resolve_embed_config()

DEFAULT_DB = "ai_digest.db"


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — Tweet retrieval & enrichment
# ─────────────────────────────────────────────────────────────────────────────

_TWEET_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:twitter|x)\.com/(\w+)/status/(\d+)",
    re.IGNORECASE,
)


def _search_xai(query: str) -> str:
    payload = {
        "model": XAI_MODEL,
        "tools": [{"type": "x_search"}],
        "input": [{
            "role": "user",
            "content": (
                f"Search X (Twitter) for recent, high-quality posts about: {query}\n\n"
                "Return the full tweet URLs for each result. "
                "Focus on original insights, not retweets. "
                "Prefer posts from the last 7 days. "
                "Include at least 5 tweet URLs if possible."
            ),
        }],
    }
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    log.info("Searching X for: %s", query)
    resp = requests.post(XAI_RESPONSES, json=payload, headers=headers, timeout=60)
    if not resp.ok:
        log.error("xAI API error %s: %s", resp.status_code, resp.text[:400])
        resp.raise_for_status()
    data = resp.json()
    text_parts = [
        content.get("text", "")
        for block in data.get("output", [])
        if block.get("type") == "message"
        for content in block.get("content", [])
        if content.get("type") == "output_text"
    ]
    citation_urls = " ".join(data.get("citations", []))
    return "\n".join(text_parts) + "\n" + citation_urls


def _extract_tweet_ids(text: str) -> list[tuple[str, str]]:
    seen: set[str] = set()
    results: list[tuple[str, str]] = []
    for screen_name, tweet_id in _TWEET_URL_RE.findall(text):
        if tweet_id not in seen:
            seen.add(tweet_id)
            results.append((screen_name, tweet_id))
    return results


def _fetch_fxtwitter(screen_name: str, tweet_id: str) -> Optional[dict]:
    url = f"{FXTWITTER_API}/{screen_name}/status/{tweet_id}"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "ai-digest-bot/1.0"})
        if resp.status_code == 200:
            return resp.json().get("tweet")
        log.warning("FxTwitter %s: HTTP %s", tweet_id, resp.status_code)
    except requests.RequestException as exc:
        log.warning("FxTwitter %s failed: %s", tweet_id, exc)
    return None


def _fetch_vxtwitter(screen_name: str, tweet_id: str) -> Optional[dict]:
    url = f"{VXTWITTER_API}/{screen_name}/status/{tweet_id}"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "ai-digest-bot/1.0"})
        if resp.status_code == 200:
            raw = resp.json()
            return {
                "id":         raw.get("tweetID"),
                "url":        raw.get("tweetURL"),
                "text":       raw.get("text"),
                "created_at": raw.get("date"),
                "likes":      raw.get("likes"),
                "retweets":   raw.get("retweets"),
                "replies":    raw.get("replies"),
                "hashtags":   raw.get("hashtags", []),
                "media_urls": raw.get("mediaURLs", []),
                "author": {
                    "name":        raw.get("user_name"),
                    "screen_name": raw.get("user_screen_name"),
                },
                "_source": "vxtwitter",
            }
        log.warning("VxTwitter %s: HTTP %s", tweet_id, resp.status_code)
    except requests.RequestException as exc:
        log.warning("VxTwitter %s failed: %s", tweet_id, exc)
    return None


def _enrich_tweet(screen_name: str, tweet_id: str) -> Optional[dict]:
    tweet = _fetch_fxtwitter(screen_name, tweet_id)
    if tweet:
        tweet["_source"] = "fxtwitter"
        return tweet
    log.info("Falling back to VxTwitter for %s", tweet_id)
    return _fetch_vxtwitter(screen_name, tweet_id)


def _prepare_record(raw: dict, topic: str) -> dict:
    author = raw.get("author") or {}
    media  = raw.get("media") or {}
    return {
        "tweet_id":           raw.get("id") or raw.get("tweetID"),
        "url":                raw.get("url") or raw.get("tweetURL"),
        "topic":              topic,
        "text":               raw.get("text", ""),
        "created_at":         raw.get("created_at") or raw.get("date"),
        "lang":               raw.get("lang"),
        "author_name":        author.get("name") or raw.get("user_name"),
        "author_screen_name": author.get("screen_name") or raw.get("user_screen_name"),
        "author_followers":   author.get("followers"),
        "likes":              raw.get("likes"),
        "retweets":           raw.get("retweets"),
        "replies":            raw.get("replies"),
        "has_media":          bool(media.get("photos") or media.get("videos") or raw.get("mediaURLs")),
        "quote_text":         (raw.get("quote") or {}).get("text"),
        "hashtags":           raw.get("hashtags", []),
        "_enrichment_source": raw.get("_source", "unknown"),
        "_fetched_at":        datetime.now(timezone.utc).isoformat(),
    }


def retrieve_tweets() -> dict:
    all_ids: list[tuple[str, str, str]] = []
    for topic in SEARCH_TOPICS:
        try:
            pairs = _extract_tweet_ids(_search_xai(topic))
            log.info("Found %d tweet URLs for topic '%s'", len(pairs), topic)
            for sn, tid in pairs:
                all_ids.append((sn, tid, topic))
        except Exception as exc:
            log.error("Search failed for '%s': %s", topic, exc)

    seen: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for sn, tid, topic in all_ids:
        if tid not in seen:
            seen.add(tid)
            unique.append((sn, tid, topic))
    unique = unique[:MAX_TWEETS]
    log.info("Enriching %d unique tweets (cap %d)…", len(unique), MAX_TWEETS)

    enriched: list[dict] = []
    for i, (sn, tid, topic) in enumerate(unique, 1):
        log.info("[%d/%d] Enriching @%s / %s", i, len(unique), sn, tid)
        raw = _enrich_tweet(sn, tid)
        if raw:
            enriched.append(_prepare_record(raw, topic))
        else:
            log.warning("Could not enrich %s — skipping", tid)
        time.sleep(REQUEST_DELAY)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topics":       SEARCH_TOPICS,
        "tweet_count":  len(enriched),
        "tweets":       enriched,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — LLM analysis
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert AI research analyst specialising in large language models,
AI agents, and the broader machine-learning ecosystem.

Your task: read a batch of tweets collected from X (Twitter) and produce a
structured intelligence digest for a technical audience — AI engineers,
researchers, and product leaders.

When referencing a tweet, always use its number in square brackets, e.g. [3]
or [7] — these will be turned into hyperlinks automatically. Do NOT invent URLs.

Guiding principles:
- Prioritise signal over noise. Many tweets repeat the same news; surface
  the underlying development once, concisely.
- Distinguish between hype and substance. Flag when something is a PR
  announcement vs a genuine technical advancement.
- Note emerging patterns across multiple tweets, even if each seems minor.
- Be precise about model names, version numbers, and organisations.
- Keep the tone analytical and neutral.
- If a claim is unverified or speculative, say so explicitly.
"""


def _build_user_prompt(data: dict) -> str:
    tweets = data.get("tweets", [])
    blocks: list[str] = []
    for i, t in enumerate(tweets, 1):
        author        = t.get("author_screen_name") or t.get("author_name", "unknown")
        followers     = t.get("author_followers")
        followers_str = f"{followers:,}" if isinstance(followers, int) else "?"
        engagement    = (
            f"❤ {t.get('likes', 0)}  "
            f"🔁 {t.get('retweets', 0)}  "
            f"💬 {t.get('replies', 0)}"
        )
        block = (
            f"[{i}] @{author} ({followers_str} followers)\n"
            f"Topic tag: {t.get('topic', '')}\n"
            f"Posted: {t.get('created_at', 'unknown')}\n"
            f"Engagement: {engagement}\n"
            f"Text: {t.get('text', '').strip()}\n"
        )
        if t.get("quote_text"):
            block += f"Quoted tweet: {t['quote_text'].strip()}\n"
        if t.get("hashtags"):
            block += f"Hashtags: {' '.join('#' + h for h in t['hashtags'])}\n"
        blocks.append(block)

    tweets_section = "\n---\n".join(blocks)

    return f"""\
## Dataset metadata
- Collected at: {data.get("generated_at", "unknown")}
- Search topics: {", ".join(data.get("topics", []))}
- Total tweets: {len(tweets)}

## Raw tweet data

{tweets_section}

---

## Your task

Analyse the tweets above and produce a digest with the following sections.
Use Markdown. Cite tweets with [N] notation throughout.

### 1. Executive Summary (3-5 sentences)
### 2. Topic Clusters
### 3. Top 5 Most Significant Tweets
### 4. Emerging Themes & Weak Signals
### 5. Noise / Hype to Ignore
### 6. Recommended Follow-Up
"""


def analyse_tweets(data: dict) -> str:
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    log.info(
        "Analysing with %s  model=%s  base_url=%s",
        os.getenv("PROVIDER", "anthropic").upper(), LLM_MODEL, LLM_BASE_URL,
    )
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(data)},
            ],
        )
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        raise
    return response.choices[0].message.content or ""


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing — link injection & references appendix
# ─────────────────────────────────────────────────────────────────────────────

_REF_RE = re.compile(r"(?<!\[)\[(\d+)\](?!\()")


def _tweet_index(tweets: list[dict]) -> dict[int, dict]:
    index: dict[int, dict] = {}
    for i, t in enumerate(tweets, 1):
        text    = (t.get("text") or "").strip()
        snippet = text[:120] + ("…" if len(text) > 120 else "")
        index[i] = {
            "url":     t.get("url") or "",
            "author":  t.get("author_screen_name") or t.get("author_name") or "unknown",
            "snippet": snippet,
        }
    return index


def _inject_links(text: str, index: dict[int, dict]) -> str:
    def replace(m: re.Match) -> str:
        n = int(m.group(1))
        entry = index.get(n)
        return f"[[{n}]]({entry['url']})" if entry and entry["url"] else m.group(0)
    return _REF_RE.sub(replace, text)


def _cited_indices(text: str, index: dict[int, dict]) -> list[int]:
    return sorted({int(m.group(1)) for m in _REF_RE.finditer(text) if int(m.group(1)) in index})


def _references_section(raw_text: str, index: dict[int, dict]) -> str:
    indices = _cited_indices(raw_text, index)
    if not indices:
        return ""
    lines = ["---", "", "## Tweet References", ""]
    for n in indices:
        e = index[n]
        url = e["url"] if e["url"] else "(no url)"
        lines.append(f"[{n}] {url}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — SQLite + sqlite-vec indexing
# ─────────────────────────────────────────────────────────────────────────────

# Section headings produced by the LLM → canonical slug
_SECTION_SLUGS: dict[str, str] = {
    "executive summary":          "executive_summary",
    "topic clusters":             "topic_clusters",
    "top 5 most significant":     "top_tweets",
    "top 5":                      "top_tweets",
    "emerging themes":            "emerging_themes",
    "weak signals":               "emerging_themes",
    "noise":                      "noise",
    "hype":                       "noise",
    "recommended follow":         "follow_up",
    "follow-up":                  "follow_up",
}

_SECTION_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


def _slugify_heading(heading: str) -> str:
    h = heading.lower().strip("#").strip()
    for key, slug in _SECTION_SLUGS.items():
        if key in h:
            return slug
    # Fallback: snake_case the heading
    return re.sub(r"\W+", "_", h).strip("_")


def _parse_sections(raw_md: str) -> list[dict]:
    """
    Split the LLM Markdown output into {heading, slug, body} dicts.
    Each section runs from its heading to the next heading (or end of text).
    """
    matches = list(_SECTION_RE.finditer(raw_md))
    sections: list[dict] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start   = m.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(raw_md)
        body    = raw_md[start:end].strip()
        if body:
            sections.append({
                "heading": heading,
                "slug":    _slugify_heading(heading),
                "body":    body,
            })
    return sections


def _floats_to_blob(floats: list[float]) -> bytes:
    """Serialise a float32 vector to the bytes format sqlite-vec expects."""
    return struct.pack(f"{len(floats)}f", *floats)


def _embed_texts(texts: list[str]) -> list[Optional[bytes]]:
    """
    Embed a batch of texts. Returns a parallel list of blobs (or None on error).
    Silently degrades — digest is always saved even if embeddings fail.
    """
    if not texts:
        return []
    try:
        client   = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)
        response = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [_floats_to_blob(item.embedding) for item in response.data]
    except Exception as exc:
        log.warning("Embedding failed (sections will be stored without vectors): %s", exc)
        return [None] * len(texts)


def init_db(db_path: str) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database, load sqlite-vec, create schema.
    Idempotent — safe to call on every run.
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # Load sqlite-vec extension
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;

        -- ── digests ──────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS digests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT NOT NULL,          -- 'YYYY-MM-DD'
            run          INTEGER NOT NULL DEFAULT 1,  -- 1 or 2 per day
            generated_at TEXT,
            topics       TEXT,                   -- JSON array
            tweet_count  INTEGER,
            llm_model    TEXT,
            embed_model  TEXT,
            raw_md       TEXT,                   -- full Markdown for LLM synthesis
            raw_json     TEXT                    -- full JSON dataset
        );

        CREATE UNIQUE INDEX IF NOT EXISTS digests_date_run
            ON digests (date, run);

        -- ── sections ─────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS sections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            digest_id   INTEGER NOT NULL REFERENCES digests(id) ON DELETE CASCADE,
            date        TEXT NOT NULL,
            run         INTEGER NOT NULL DEFAULT 1,
            section     TEXT NOT NULL,           -- slug: executive_summary, etc.
            heading     TEXT,
            body        TEXT,
            embedding   BLOB                     -- float32 vector (sqlite-vec)
        );

        CREATE INDEX IF NOT EXISTS sections_date    ON sections (date);
        CREATE INDEX IF NOT EXISTS sections_section ON sections (section);

        -- FTS5 index over section content
        CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
            heading,
            body,
            content='sections',
            content_rowid='id'
        );

        -- ── tweet_refs ────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS tweet_refs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            digest_id   INTEGER NOT NULL REFERENCES digests(id) ON DELETE CASCADE,
            date        TEXT NOT NULL,
            run         INTEGER NOT NULL DEFAULT 1,
            tweet_n     INTEGER,                 -- position [N] in digest
            tweet_id    TEXT,
            url         TEXT,
            author      TEXT,
            text        TEXT,
            likes       INTEGER,
            retweets    INTEGER,
            replies     INTEGER,
            topic_tag   TEXT,
            has_media   INTEGER                  -- 0/1
        );

        CREATE INDEX IF NOT EXISTS tweet_refs_date   ON tweet_refs (date);
        CREATE INDEX IF NOT EXISTS tweet_refs_author ON tweet_refs (author);
    """)
    db.commit()
    return db


def _next_run(db: sqlite3.Connection, date: str) -> int:
    """Return 1 for first digest of the day, 2 for second, etc."""
    row = db.execute(
        "SELECT COALESCE(MAX(run), 0) FROM digests WHERE date = ?", (date,)
    ).fetchone()
    return (row[0] or 0) + 1


def index_digest(
    data: dict,
    raw_analysis: str,
    raw_md: str,
    db_path: str,
) -> int:
    """
    Parse the digest, embed sections, and write everything to SQLite.
    Returns the new digest_id.
    """
    db   = init_db(db_path)
    # Derive date from the digest itself, not wall-clock time
    # so indexing a past digest (e.g. replayed from raw_json) lands on the right date
    generated_at_str = data.get("generated_at", "")
    try:
        date = datetime.fromisoformat(generated_at_str).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run  = _next_run(db, date)

    log.info("Indexing digest into %s  (date=%s  run=%d)", db_path, date, run)

    # ── 1. Insert digest row ──────────────────────────────────────────────────
    cur = db.execute(
        """
        INSERT INTO digests
            (date, run, generated_at, topics, tweet_count,
             llm_model, embed_model, raw_md, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            date, run,
            data.get("generated_at"),
            json.dumps(data.get("topics", [])),
            data.get("tweet_count", 0),
            LLM_MODEL, EMBED_MODEL,
            raw_md,
            json.dumps(data, ensure_ascii=False),
        ),
    )
    digest_id = cur.lastrowid
    db.commit()

    # ── 2. Parse + embed sections ─────────────────────────────────────────────
    sections = _parse_sections(raw_analysis)
    bodies   = [s["body"] for s in sections]
    blobs    = _embed_texts(bodies)

    log.info("Embedding %d sections with %s…", len(sections), EMBED_MODEL)

    for section, blob in zip(sections, blobs):
        cur = db.execute(
            """
            INSERT INTO sections
                (digest_id, date, run, section, heading, body, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (digest_id, date, run,
             section["slug"], section["heading"], section["body"], blob),
        )
        section_id = cur.lastrowid
        # Keep FTS index in sync
        db.execute(
            "INSERT INTO sections_fts (rowid, heading, body) VALUES (?, ?, ?)",
            (section_id, section["heading"], section["body"]),
        )

    # ── 3. Insert tweet refs ──────────────────────────────────────────────────
    for i, t in enumerate(data.get("tweets", []), 1):
        db.execute(
            """
            INSERT INTO tweet_refs
                (digest_id, date, run, tweet_n, tweet_id, url, author,
                 text, likes, retweets, replies, topic_tag, has_media)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                digest_id, date, run, i,
                t.get("tweet_id"), t.get("url"),
                t.get("author_screen_name") or t.get("author_name"),
                t.get("text"),
                t.get("likes"), t.get("retweets"), t.get("replies"),
                t.get("topic"),
                int(t.get("has_media", False)),
            ),
        )

    db.commit()
    db.close()

    log.info(
        "Indexed: digest_id=%d  sections=%d  tweets=%d",
        digest_id, len(sections), data.get("tweet_count", 0),
    )
    return digest_id


# ─────────────────────────────────────────────────────────────────────────────
# Output formatters
# ─────────────────────────────────────────────────────────────────────────────

def format_json(data: dict, raw_analysis: str) -> str:
    doc = {**data, "analysis": raw_analysis}
    return json.dumps(doc, ensure_ascii=False, indent=2)


def format_md(data: dict, raw_analysis: str) -> str:
    provider = os.getenv("PROVIDER", "anthropic").upper()
    header = (
        f"# AI/LLM Twitter Digest\n\n"
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"**Provider:** {provider}\n"
        f"**Model:** {LLM_MODEL}\n"
        f"**Tweets analysed:** {data.get('tweet_count', '?')}\n"
        f"**Search topics:** {', '.join(data.get('topics', []))}\n\n"
        f"---\n\n"
    )
    index           = _tweet_index(data.get("tweets", []))
    linked_analysis = _inject_links(raw_analysis, index)
    references      = _references_section(raw_analysis, index)
    return header + linked_analysis + ("\n\n" + references if references else "")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_digest",
        description="Search X for AI/LLM tweets, enrich, analyse with an LLM, and index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Environment variables:

  Retrieval:
    XAI_API_KEY         xAI / Grok API key (required for tweet search)

  LLM provider:
    PROVIDER            Preset name: anthropic (default), openai, xai,
                        mistral, together, ollama, lmstudio
    ANTHROPIC_API_KEY   API key when PROVIDER=anthropic
    OPENAI_API_KEY      API key when PROVIDER=openai
    XAI_API_KEY         API key when PROVIDER=xai
    MISTRAL_API_KEY     API key when PROVIDER=mistral
    TOGETHER_API_KEY    API key when PROVIDER=together
    LLM_BASE_URL        Override the provider base URL
    LLM_API_KEY         Override the provider API key
    LLM_MODEL           Override the model name

  Embeddings (only used with --db):
    EMBED_PROVIDER      Preset: openai (default), ollama, lmstudio
                        Falls back to PROVIDER when not set
    EMBED_MODEL         Override the embedding model name
    EMBED_BASE_URL      Override the embedding base URL
    EMBED_API_KEY       Override the embedding API key

Examples:
  uv run ai_digest.py                            # JSON -> stdout
  uv run ai_digest.py --format md                # Markdown -> stdout
  uv run ai_digest.py --format md -o out.md      # Markdown -> file
  uv run ai_digest.py -o digest.json             # JSON -> file
  uv run ai_digest.py --format md --db           # index into ai_digest.db
  uv run ai_digest.py --format md --db my.db     # index into custom path
  PROVIDER=ollama LLM_MODEL=llama3.2 uv run ai_digest.py --format md --db
""",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "md"],
        default="json",
        metavar="FORMAT",
        help="Output format: json (default) or md",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        default=None,
        help="Write output to FILE instead of stdout",
    )
    parser.add_argument(
        "--db",
        nargs="?",
        const=DEFAULT_DB,
        default=None,
        metavar="DB_PATH",
        help=(
            f"Index digest into SQLite database. "
            f"Optional path (default: {DEFAULT_DB})"
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not XAI_API_KEY:
        raise SystemExit(
            "ERROR: XAI_API_KEY not set.\n"
            "Add to .env:  XAI_API_KEY=xai-xxxxxxxxxxxxxxxxxxxx"
        )

    log.info(
        "Starting pipeline | format=%s | output=%s | db=%s | provider=%s | model=%s",
        args.format,
        args.output or "stdout",
        args.db or "disabled",
        os.getenv("PROVIDER", "anthropic"),
        LLM_MODEL,
    )

    # ── Step 1: Retrieve & enrich ─────────────────────────────────────────────
    data = retrieve_tweets()
    if data["tweet_count"] == 0:
        raise SystemExit("ERROR: No tweets retrieved. Check XAI_API_KEY and topics.")

    # ── Step 2: Analyse ───────────────────────────────────────────────────────
    raw_analysis = analyse_tweets(data)

    # ── Step 3: Format ────────────────────────────────────────────────────────
    if args.format == "md":
        output_str = format_md(data, raw_analysis)
    else:
        output_str = format_json(data, raw_analysis)

    # ── Step 4: Output ────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str)
        log.info("Output written → %s", args.output)
    else:
        sys.stdout.write(output_str + "\n")
        log.info("Output written → stdout")

    # ── Step 5: Index (optional) ──────────────────────────────────────────────
    if args.db:
        # Always pass the full Markdown for raw_md storage
        raw_md = output_str if args.format == "md" else format_md(data, raw_analysis)
        index_digest(data, raw_analysis, raw_md, args.db)


if __name__ == "__main__":
    main()
