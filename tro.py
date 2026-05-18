#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "httpx",
#     "sqlite-vec",
#     "python-dotenv",
# ]
# ///

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

LOG = logging.getLogger("tro")

QUERIES = [
        "(Claude OR GPT OR Gemini OR DeepSeek OR Qwen OR Grok OR Kimi) release min_faves:20",
        "(\"Model Context Protocol\" OR MCP) (server OR client OR agents) min_faves:10",
        "(agentic OR \"AI agents\" OR orchestration) (framework OR runtime) min_faves:15",
        "(arxiv OR benchmark OR SOTA) (LLM OR reasoning OR multimodal) min_faves:25",
        "(open-source OR weights) model release min_faves:20",
        "(\"tool calling\" OR MCP OR function calling) agents min_faves:10",
        "(coding agent OR SWE-agent OR autonomous coding) min_faves:15",
        "\"MCP server\" repository min_faves:25"
]

SEARCH_PROMPT = (
    "Search X (Twitter) for recent, high-quality posts about: {query}\n\n"
    "Return the full tweet URLs for each result. "
    "Focus on original insights, not retweets. "
    "Prefer posts from the last {num_of_days} days. "
    "Include at least {num_of_tweets} tweet URLs if possible."
)

XAI_API_URL = "https://api.x.ai/v1/responses"
SEARCH_MODEL = "grok-4-1-fast"
FX_API_BASE = "https://api.fxtwitter.com"
VX_API_BASE = "https://api.vxtwitter.com"
REQUEST_DELAY = 0.5
MAX_RETRIES = 3
DEFAULT_NUM_TWEETS = 50
DEFAULT_NUM_DAYS = 7

ANALYSIS_SYSTEM_PROMPT = """You are an AI industry analyst. Analyze the following tweets about AI/LLM topics and produce a structured report.

Guiding principles:
- Prioritise signal over noise.  Many tweets repeat the same news; surface the underlying development once, concisely.
- Distinguish between hype and substance.  Flag when something is a PR announcement vs a genuine technical advancement.
- Note emerging patterns that appear across multiple tweets, even if each individual tweet seems minor.
- Be precise about model names, version numbers, and organisations.
- Keep the tone analytical and neutral, not cheerleader-ish.
- If a tweet's claim is unverified or speculative, say so explicitly.

Report structure:

## 1. Executive Summary (3-5 sentences)
What is the overall state of the AI/LLM space based on this snapshot?

## 2. Topic Clusters
Group the tweets into 3-6 meaningful clusters (e.g. "New model releases", "Tooling & frameworks", "Research papers", "Industry moves", "Community debate").
For each cluster:
- Give it a descriptive heading
- List the key developments in bullet points
- Note the most credible / high-signal sources

## 3. Top 5 Most Significant Tweets
Pick the 5 tweets with the highest signal value. For each:
- Quote the tweet number [N] and the @handle
- Explain in 1-2 sentences WHY it is significant

## 4. Emerging Themes & Weak Signals
What patterns appear across multiple tweets that might not be obvious from any single tweet?  What could these signal about near-term trends?

## 5. Noise / Hype to Ignore
Which tweets (or patterns) appear to be marketing noise, repetitive, unverified, or otherwise low-value?  Be specific.

## 6. Recommended Follow-Up
- 3 specific X search queries that would enrich this digest further
- 3 other sources (papers, blogs, GitHub repos) worth checking based on what you saw"""


def load_env() -> None:
    env_file = Path(".env")
    if env_file.is_file():
        LOG.info("Loading environment from .env")
        load_dotenv(env_file, override=False)

        # Map TPO_ legacy prefix to TRO_ for any not already set
        for key, value in os.environ.items():
            if key.startswith("TPO_"):
                target = "TRO_" + key[4:]
                if target not in os.environ:
                    os.environ[target] = value


def parse_args() -> argparse.Namespace:
    default_db = str(Path(__file__).resolve().parent / "tro.db")

    parser = argparse.ArgumentParser(
        prog="tro",
        description="Twitter Research OS — Analyze X.com tweets related to AI topics",
        epilog=(
            "Environment variables (all prefixed TRO_):\n"
            "  TRO_XAI_API_KEY          x.ai API key (required)\n"
            "  TRO_ANALYSIS_API_KEY     Analysis API key (required)\n"
            "  TRO_ANALYSIS_BASE_URL    Analysis base URL "
            "(default: https://api.openai.com/v1)\n"
            "  TRO_ANALYSIS_MODEL       Analysis model (default: gpt-4o)\n"
            "  TRO_EMBEDDING_API_KEY    Embedding API key (optional)\n"
            "  TRO_EMBEDDING_BASE_URL   Embedding base URL "
            "(default: https://api.openai.com/v1)\n"
            "  TRO_EMBEDDING_MODEL      Embedding model "
            "(default: text-embedding-3-small)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-tweets",
        type=int,
        default=DEFAULT_NUM_TWEETS,
        help=f"Number of tweets to analyze (default: {DEFAULT_NUM_TWEETS})",
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        default=DEFAULT_NUM_DAYS,
        help=f"Number of days to look back (default: {DEFAULT_NUM_DAYS})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--db",
        nargs="?",
        const=default_db,
        default=None,
        metavar="PATH",
        help=(
            "Store digest in SQLite database. "
            "Without argument uses ./tro.db in the script directory. "
            "If omitted, no DB storage."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_env(key: str, required: bool = True) -> str | None:
    value = os.environ.get(key)
    if required and not value:
        LOG.error("Environment variable %s is required but not set", key)
        sys.exit(1)
    return value


def parse_tweet_url(url: str) -> tuple[str, str] | None:
    patterns = [
        r"(?:https?://)?(?:www\.)?(?:twitter|x)\.com/(\w+)/status/(\d+)",
        r"(?:https?://)?(?:www\.)?(?:twitter|x)\.com/i/status/(\d+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            groups = m.groups()
            if len(groups) == 2 and groups[0] != "i":
                return groups[0], groups[1]
            if len(groups) == 1:
                return "", groups[0]
    return None


TWEET_URL_RE = re.compile(r"https?://(?:www\.)?(?:twitter|x)\.com/(?:\w+|i)/status/\d+")


def extract_tweet_urls(citations: list[Any]) -> list[str]:
    urls: list[str] = []
    for item in citations:
        if isinstance(item, str) and re.search(r"/status/\d+", item):
            urls.append(item)
    return urls


def extract_tweet_urls_from_text(text: str) -> list[str]:
    return TWEET_URL_RE.findall(text)


def get_assistant_text(data: dict[str, Any]) -> str:
    output = data.get("output", [])
    for item in output:
        if item.get("type") == "message" and item.get("role") == "assistant":
            content_list = item.get("content", [])
            for c in content_list:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    return c.get("text", "")
    return ""


async def search_tweets_for_query(
    client: httpx.AsyncClient,
    query: str,
    num_tweets: int,
    num_days: int,
    api_key: str,
) -> list[str]:
    prompt = SEARCH_PROMPT.format(
        query=query,
        num_of_tweets=num_tweets,
        num_of_days=num_days,
    )
    from_date = (datetime.now(timezone.utc) - timedelta(days=num_days)).strftime(
        "%Y-%m-%d"
    )
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    payload: dict[str, Any] = {
        "model": SEARCH_MODEL,
        "input": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "type": "x_search",
                "from_date": from_date,
                "to_date": to_date,
            }
        ],
    }

    LOG.info("Searching: %s (%s to %s)", query, from_date, to_date)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.post(
                XAI_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", "5"))
                LOG.warning(
                    "Rate limited (429), waiting %ds (attempt %d/%d)",
                    retry_after,
                    attempt,
                    MAX_RETRIES,
                )
                await asyncio.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()

            tweet_urls: list[str] = []

            # Extract from citations array
            citations = data.get("citations", [])
            tweet_urls.extend(extract_tweet_urls(citations))

            # Extract from assistant message output_text
            full_text = get_assistant_text(data)
            tweet_urls.extend(extract_tweet_urls_from_text(full_text))

            # Deduplicate preserving order
            seen: set[str] = set()
            unique_urls: list[str] = []
            for url in tweet_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)

            LOG.info(
                "  -> %d tweet URLs found for: %s",
                len(unique_urls),
                query,
            )
            return unique_urls

        except httpx.HTTPStatusError as e:
            LOG.error(
                "HTTP error for '%s': %s (attempt %d/%d)",
                query,
                e,
                attempt,
                MAX_RETRIES,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2**attempt)
            else:
                LOG.warning("Exhausted retries for query: %s", query)
                return []
        except Exception as e:
            LOG.error("Error for '%s': %s", query, e)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2**attempt)
            else:
                return []

    return []


async def search_all_tweets(
    num_tweets: int,
    num_days: int,
    api_key: str,
) -> list[str]:
    tweets_per_query = max(1, num_tweets // len(QUERIES))
    LOG.info(
        "Searching ~%d tweets per query across %d topics",
        tweets_per_query,
        len(QUERIES),
    )

    async with httpx.AsyncClient() as client:
        tasks = [
            search_tweets_for_query(client, q, tweets_per_query, num_days, api_key)
            for q in QUERIES
        ]
        results = await asyncio.gather(*tasks)

    all_urls: list[str] = []
    seen: set[str] = set()
    for urls in results:
        for url in urls:
            if url not in seen:
                seen.add(url)
                all_urls.append(url)

    LOG.info("Total unique tweet URLs found: %d", len(all_urls))

    if len(all_urls) > num_tweets:
        LOG.info("Capping to %d tweets (from %d total)", num_tweets, len(all_urls))
        all_urls = all_urls[:num_tweets]

    return all_urls


async def fetch_tweet_from_api(
    client: httpx.AsyncClient,
    api_base: str,
    screen_name: str,
    tweet_id: str,
    index: int,
) -> dict[str, Any] | None:
    if screen_name:
        api_url = f"{api_base}/{screen_name}/status/{tweet_id}"
    else:
        api_url = f"{api_base}/i/status/{tweet_id}"
    try:
        response = await client.get(
            api_url,
            headers={"User-Agent": "tro-twitter-research-os/1.0"},
            timeout=30.0,
        )
        if response.status_code == 404:
            LOG.debug("[%d] Tweet not found: %s", index, tweet_id)
            return None
        if response.status_code == 401:
            LOG.debug("[%d] Private tweet: %s", index, tweet_id)
            return None
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 200:
            LOG.debug(
                "[%d] Non-200 code from %s: %s", index, api_base, data.get("message")
            )
            return None
        return data
    except Exception as e:
        LOG.debug("[%d] API error (%s): %s", index, api_base, e)
        return None


def build_enriched_tweet(
    raw: dict[str, Any], url: str, tweet_id: str, index: int
) -> dict[str, Any]:
    tweet_data = raw.get("tweet", {})
    author = tweet_data.get("author", {})
    return {
        "index": index,
        "id": tweet_data.get("id", tweet_id),
        "url": tweet_data.get("url", url),
        "text": tweet_data.get("text", ""),
        "author_name": author.get("name", ""),
        "author_handle": author.get("screen_name", ""),
        "created_at": tweet_data.get("created_at", ""),
        "likes": tweet_data.get("likes", 0),
        "retweets": tweet_data.get("retweets", 0),
        "replies": tweet_data.get("replies", 0),
        "views": tweet_data.get("views") or 0,
    }


async def enrich_single_tweet(
    client: httpx.AsyncClient,
    url: str,
    index: int,
) -> dict[str, Any] | None:
    parsed = parse_tweet_url(url)
    if not parsed:
        LOG.warning("[%d] Could not parse URL: %s", index, url)
        return None

    screen_name, tweet_id = parsed

    result = await fetch_tweet_from_api(
        client, FX_API_BASE, screen_name, tweet_id, index
    )
    if result is None:
        LOG.debug("[%d] FxTwitter failed, trying vxTwitter fallback", index)
        result = await fetch_tweet_from_api(
            client, VX_API_BASE, screen_name, tweet_id, index
        )

    if result is None:
        LOG.warning("[%d] Failed to enrich tweet %s", index, tweet_id)
        return None

    return build_enriched_tweet(result, url, tweet_id, index)


async def enrich_tweets(urls: list[str]) -> list[dict[str, Any]]:
    LOG.info("Enriching %d tweets...", len(urls))
    enriched: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    skipped = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        for i, url in enumerate(urls):
            if i > 0:
                await asyncio.sleep(REQUEST_DELAY)

            tweet = await enrich_single_tweet(client, url, i + 1)
            if tweet is None:
                failed += 1
                continue

            if tweet["id"] in seen_ids:
                LOG.debug("Skipping duplicate tweet: %s", tweet["id"])
                skipped += 1
                continue

            seen_ids.add(tweet["id"])
            enriched.append(tweet)

            if (i + 1) % 10 == 0:
                LOG.info(
                    "  Enriched %d/%d (%d failed, %d skipped)",
                    len(enriched),
                    len(urls),
                    failed,
                    skipped,
                )

    LOG.info(
        "Enrichment complete: %d unique tweets (%d failed, %d duplicates)",
        len(enriched),
        failed,
        skipped,
    )
    return enriched


def build_analysis_payload(tweets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "number": t["index"],
            "handle": f"@{t['author_handle']}",
            "name": t["author_name"],
            "text": t["text"],
            "likes": t["likes"],
            "retweets": t["retweets"],
            "replies": t["replies"],
            "date": t["created_at"],
        }
        for t in tweets
    ]


async def analyze_tweets(
    tweets: list[dict[str, Any]],
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    LOG.info("Analyzing %d tweets with model %s...", len(tweets), model)

    tweets_for_analysis = build_analysis_payload(tweets)
    tweets_json = json.dumps(tweets_for_analysis, indent=2, ensure_ascii=False)
    user_prompt = (
        f"Here are {len(tweets_for_analysis)} tweets about AI/LLM topics. "
        f"Analyze them and produce the structured report.\n\n```json\n{tweets_json}\n```"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 8000,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    transport = httpx.AsyncHTTPTransport(retries=2)

    async with httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(300.0, connect=15.0),
    ) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                )
                if response.status_code == 429:
                    retry_after = int(response.headers.get("retry-after", "10"))
                    LOG.warning(
                        "Analysis rate limited (429), waiting %ds (attempt %d/%d)",
                        retry_after,
                        attempt,
                        MAX_RETRIES,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location", "")
                    LOG.error(
                        "Analysis endpoint redirected (%d -> %s). "
                        "Check TRO_ANALYSIS_BASE_URL is a valid OpenAI-compatible base URL "
                        "(e.g. https://api.openai.com/v1, http://localhost:11434/v1)",
                        response.status_code,
                        location,
                    )
                    raise RuntimeError(
                        f"Invalid TRO_ANALYSIS_BASE_URL — redirected to {location}"
                    )

                response.raise_for_status()
                data = response.json()

                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("No choices in analysis response")

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise RuntimeError("Empty content in analysis response")

                usage = data.get("usage", {})
                LOG.info(
                    "Analysis complete (%d tokens)",
                    usage.get("total_tokens", 0),
                )
                return content

            except httpx.HTTPStatusError as e:
                LOG.error(
                    "Analysis HTTP error: %s (attempt %d/%d)",
                    e,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2**attempt)
                else:
                    raise
            except Exception as e:
                LOG.error(
                    "Analysis error: %s (attempt %d/%d)",
                    e,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

    raise RuntimeError("Analysis failed after all retries")


def build_json_output(
    tweets: list[dict[str, Any]],
    analysis: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "num_tweets_requested": args.num_tweets,
            "num_days": args.days,
            "num_tweets_found": len(tweets),
            "search_model": SEARCH_MODEL,
        },
        "tweets": tweets,
        "analysis": analysis,
    }


def build_markdown_output(
    tweets: list[dict[str, Any]],
    analysis: str,
    args: argparse.Namespace,
) -> str:
    header = (
        "# Twitter Research OS — AI/LLM Digest\n\n"
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"**Tweets analyzed:** {len(tweets)} (requested: {args.num_tweets})\n"
        f"**Lookback period:** {args.days} days\n\n"
        "---\n\n"
    )

    summary_parts = [f"## Sampled Tweets ({len(tweets)})\n\n"]
    for t in tweets:
        summary_parts.append(f"- [{t['index']}] {t['url']}\n")
    summary_parts.append("\n---\n\n")
    summary = "".join(summary_parts)

    return header + summary + analysis + "\n"


def write_output(output: str, path: str | None) -> None:
    if path:
        Path(path).write_text(output, encoding="utf-8")
        LOG.info("Output written to %s", path)
    else:
        print(output)


def parse_digest(markdown: str) -> dict[str, list[dict[str, Any]]]:
    section_pattern = re.compile(r"^## \d\.\s", re.MULTILINE)
    sub_pattern = re.compile(r"^### (.+?)$", re.MULTILINE)
    tweet_pattern = re.compile(
        r"^- \[(\d+)\] (https?://(?:twitter|x)\.com/\S+)", re.MULTILINE
    )
    heading_pattern = re.compile(r"^## (\d)\.\s*(.+?)(?:\s*\(.*?\))?\s*$", re.MULTILINE)

    analysis_body = markdown
    # Strip the leading h1 report title if present
    h1_idx = markdown.find("\n## ")
    if h1_idx != -1 and markdown.startswith("# "):
        after_header = markdown[:h1_idx]
        if "Twitter Research OS" not in after_header:
            analysis_body = markdown[h1_idx + 1 :]

    tweets: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []

    # Extract sampled tweets
    tweets_section_match = re.search(
        r"^## Sampled Tweets.*?\n((?:\s*- \[.*\n)*)", analysis_body, re.MULTILINE
    )
    if tweets_section_match:
        for m in tweet_pattern.finditer(tweets_section_match.group(1)):
            tweets.append({"index": int(m.group(1)), "url": m.group(2)})

    # Find major section boundaries
    section_matches = list(section_pattern.finditer(analysis_body))
    type_map = {
        "1": "executive_summary",
        "2": "topic_cluster",
        "3": "top_5",
        "4": "themes",
        "5": "noise",
        "6": "followup",
    }

    for i, match in enumerate(section_matches):
        heading_line = match.group(0).strip()
        number = heading_line.split(".")[0].split("## ")[1]
        section_type = type_map.get(number, heading_line)

        title_match = heading_pattern.search(heading_line)
        section_title = title_match.group(2).strip() if title_match else None

        start = match.end()
        end = (
            section_matches[i + 1].start()
            if i + 1 < len(section_matches)
            else len(analysis_body)
        )
        body = analysis_body[start:end].strip()

        if section_type == "topic_cluster":
            sub_matches = list(sub_pattern.finditer(body))
            if sub_matches:
                for j, sm in enumerate(sub_matches):
                    cluster_title = sm.group(1).strip()
                    s_start = sm.end()
                    s_end = (
                        sub_matches[j + 1].start()
                        if j + 1 < len(sub_matches)
                        else len(body)
                    )
                    sections.append(
                        {
                            "section_type": "topic_cluster",
                            "section_title": cluster_title,
                            "content": body[s_start:s_end].strip(),
                        }
                    )
                continue

        sections.append(
            {
                "section_type": section_type,
                "section_title": section_title,
                "content": body.strip(),
            }
        )

    return {"tweets": tweets, "sections": sections}


# --- Database storage ---

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS digests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at    TEXT NOT NULL,
    num_tweets      INTEGER,
    num_days        INTEGER,
    search_model    TEXT,
    analysis_model  TEXT,
    embedding_model TEXT,
    raw_markdown    TEXT
);

CREATE TABLE IF NOT EXISTS tweets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_id       INTEGER NOT NULL REFERENCES digests(id),
    tweet_index     INTEGER NOT NULL,
    url             TEXT NOT NULL,
    text            TEXT NOT NULL,
    author_handle   TEXT,
    likes           INTEGER DEFAULT 0,
    retweets        INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS tweets_fts USING fts5(
    text, author_handle,
    content='', content_rowid='',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS sections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_id       INTEGER NOT NULL REFERENCES digests(id),
    section_type    TEXT NOT NULL,
    section_title   TEXT,
    content         TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
    content, section_title,
    content='', content_rowid='',
    tokenize='porter unicode61'
);
"""


def init_db(path: str):
    import sqlite3

    import sqlite_vec

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(DB_SCHEMA)
    _create_vec_tables(conn)
    return conn


def _create_vec_tables(conn) -> None:
    for name in ("tweets_vec", "sections_vec"):
        cur = conn.execute(
            "SELECT name FROM pragma_module_list() WHERE name=?",
            (name,),
        )
        if not cur.fetchone():
            try:
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} "
                    "USING vec0(embedding float[1536])"
                )
            except Exception:
                pass


def store_digest(
    conn,
    generated_at: str,
    num_tweets: int,
    num_days: int,
    search_model: str,
    analysis_model: str,
    embedding_model: str | None,
    raw_markdown: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO digests(generated_at, num_tweets, num_days, "
        "search_model, analysis_model, embedding_model, raw_markdown) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            generated_at,
            num_tweets,
            num_days,
            search_model,
            analysis_model,
            embedding_model,
            raw_markdown,
        ),
    )
    return cur.lastrowid


def store_tweets(
    conn,
    digest_id: int,
    enriched: list[dict[str, Any]],
    parsed_tweets: list[dict[str, Any]],
) -> list[int]:
    url_to_enriched: dict[str, dict[str, Any]] = {}
    for t in enriched:
        url_to_enriched[t["url"]] = t

    row_ids: list[int] = []
    for pt in parsed_tweets:
        et = url_to_enriched.get(pt["url"])
        if et is None:
            continue
        cur = conn.execute(
            "INSERT INTO tweets(digest_id, tweet_index, url, text, "
            "author_handle, likes, retweets) VALUES (?,?,?,?,?,?,?)",
            (
                digest_id,
                pt["index"],
                pt["url"],
                et["text"],
                et["author_handle"],
                et["likes"],
                et["retweets"],
            ),
        )
        row_ids.append(cur.lastrowid)

    # Populate FTS5 explicitly
    for rid in row_ids:
        t = conn.execute(
            "SELECT text, author_handle FROM tweets WHERE id=?", (rid,)
        ).fetchone()
        if t:
            conn.execute(
                "INSERT INTO tweets_fts(rowid, text, author_handle) VALUES (?,?,?)",
                (rid, t[0], t[1]),
            )

    return row_ids


def store_sections(
    conn,
    digest_id: int,
    sections: list[dict[str, Any]],
) -> list[int]:
    row_ids: list[int] = []
    for s in sections:
        cur = conn.execute(
            "INSERT INTO sections(digest_id, section_type, section_title, content) "
            "VALUES (?,?,?,?)",
            (digest_id, s["section_type"], s.get("section_title"), s["content"]),
        )
        row_ids.append(cur.lastrowid)

    # Populate FTS5 explicitly
    for rid in row_ids:
        s = conn.execute(
            "SELECT content, section_title FROM sections WHERE id=?", (rid,)
        ).fetchone()
        if s:
            conn.execute(
                "INSERT INTO sections_fts(rowid, content, section_title) VALUES (?,?,?)",
                (rid, s[0], s[1] or ""),
            )

    return row_ids


def store_vectors(
    conn,
    row_ids: list[int],
    vectors: list[list[float] | None],
    table_name: str,
) -> None:
    if not vectors or not row_ids:
        return
    for rid, vec in zip(row_ids, vectors):
        if rid and vec is not None:
            vec_json = json.dumps(vec)
            conn.execute(
                f"INSERT INTO {table_name}(rowid, embedding) VALUES (?,?)",
                (rid, vec_json),
            )


# --- Embedding client ---

EMBEDDING_BATCH_SIZE = 100
_EMBEDDING_DIM_CACHE: int | None = None


async def _get_embedding_dim(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
) -> int:
    global _EMBEDDING_DIM_CACHE
    if _EMBEDDING_DIM_CACHE is not None:
        return _EMBEDDING_DIM_CACHE

    response = await client.post(
        f"{base_url.rstrip('/')}/embeddings",
        json={"model": model, "input": "dim probe"},
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    dim = len(data["data"][0]["embedding"])
    _EMBEDDING_DIM_CACHE = dim
    LOG.info("Embedding dimension detected: %d", dim)
    return dim


async def embed_texts(
    texts: list[str],
    api_key: str,
    base_url: str,
    model: str,
) -> list[list[float] | None]:
    all_embeddings: list[list[float] | None] = []
    total = len(texts)

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
        await _get_embedding_dim(client, base_url, api_key, model)

        for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
            batch = texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = await client.post(
                        f"{base_url.rstrip('/')}/embeddings",
                        json={"model": model, "input": batch},
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("retry-after", "5"))
                        LOG.warning("Embedding rate limited, waiting %ds", retry_after)
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    LOG.error(
                        "Embedding batch error (attempt %d/%d): %s",
                        attempt,
                        MAX_RETRIES,
                        e,
                    )
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(2**attempt)
                    else:
                        # Mark failed batch as None so vector tables stay clean
                        for _ in batch:
                            all_embeddings.append(None)

            if (batch_start + EMBEDDING_BATCH_SIZE) < total:
                await asyncio.sleep(0.1)

    return all_embeddings


# --- DB storage integration ---


async def _store_embeddings(
    conn,
    parsed: dict[str, Any],
    enriched: list[dict[str, Any]],
    tweet_ids: list[int],
    section_ids: list[int],
    emb_api_key: str,
) -> None:
    emb_base_url = os.environ.get("TRO_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    emb_model = os.environ.get("TRO_EMBEDDING_MODEL", "text-embedding-3-small")

    section_texts = [s["content"] for s in parsed["sections"]]
    if section_texts and section_ids:
        section_vectors = await embed_texts(
            section_texts, emb_api_key, emb_base_url, emb_model
        )
        store_vectors(conn, section_ids, section_vectors, "sections_vec")

    tweet_texts = [
        t["text"]
        for t in enriched
        if any(pt["url"] == t["url"] for pt in parsed["tweets"])
    ]
    if tweet_texts and tweet_ids:
        tweet_vectors = await embed_texts(
            tweet_texts, emb_api_key, emb_base_url, emb_model
        )
        store_vectors(conn, tweet_ids, tweet_vectors, "tweets_vec")


async def store_digest_in_db(
    enriched: list[dict[str, Any]],
    args: argparse.Namespace,
    analysis_model: str,
    markdown: str,
    db_path: str | None,
) -> None:
    if not db_path:
        return

    LOG.info("Storing digest in SQLite: %s", db_path)
    conn = init_db(db_path)
    digest_id = -1

    try:
        conn.execute("BEGIN")

        generated_at = datetime.now(timezone.utc).isoformat()
        digest_id = store_digest(
            conn,
            generated_at,
            args.num_tweets,
            args.days,
            SEARCH_MODEL,
            analysis_model,
            None,
            markdown,
        )

        parsed = parse_digest(markdown)
        tweet_ids = store_tweets(conn, digest_id, enriched, parsed["tweets"])
        section_ids = store_sections(conn, digest_id, parsed["sections"])

        emb_api_key = os.environ.get("TRO_EMBEDDING_API_KEY")
        if emb_api_key:
            emb_model = os.environ.get("TRO_EMBEDDING_MODEL", "text-embedding-3-small")
            conn.execute(
                "UPDATE digests SET embedding_model=? WHERE id=?",
                (emb_model, digest_id),
            )
            try:
                await _store_embeddings(
                    conn,
                    parsed,
                    enriched,
                    tweet_ids,
                    section_ids,
                    emb_api_key,
                )
            except Exception as exc:
                LOG.warning(
                    "Failed to store embeddings (%s: %s). "
                    "Structured data and FTS5 were saved successfully. "
                    "Check TRO_EMBEDDING_BASE_URL and TRO_EMBEDDING_MODEL.",
                    type(exc).__name__,
                    exc,
                )

        conn.commit()
        LOG.info(
            "Digest stored: id=%d, tweets=%d, sections=%d",
            digest_id,
            len(tweet_ids),
            len(section_ids),
        )

    except Exception:
        LOG.exception("Failed to store digest in database")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


async def main() -> None:
    load_env()

    args = parse_args()
    setup_logging(args.verbose)

    xai_api_key = get_env("TRO_XAI_API_KEY")
    analysis_api_key = get_env("TRO_ANALYSIS_API_KEY")
    analysis_base_url = os.environ.get(
        "TRO_ANALYSIS_BASE_URL", "https://api.openai.com/v1"
    )
    analysis_model = os.environ.get("TRO_ANALYSIS_MODEL", "gpt-4o")

    LOG.info(
        "TRO pipeline start: %d tweets, %d days, format=%s",
        args.num_tweets,
        args.days,
        args.format,
    )

    urls = await search_all_tweets(args.num_tweets, args.days, xai_api_key)
    if not urls:
        LOG.error("No tweets found. Check API key, date range, and query terms.")
        sys.exit(1)

    enriched = await enrich_tweets(urls)
    if not enriched:
        LOG.error("Failed to enrich any tweets.")
        sys.exit(1)

    analysis = await analyze_tweets(
        enriched, analysis_api_key, analysis_base_url, analysis_model
    )

    markdown_output = build_markdown_output(enriched, analysis, args)

    await store_digest_in_db(
        enriched,
        args,
        analysis_model,
        markdown_output,
        args.db,
    )

    if args.format == "json":
        output = json.dumps(
            build_json_output(enriched, analysis, args),
            indent=2,
            ensure_ascii=False,
        )
    else:
        output = markdown_output

    write_output(output, args.output)
    LOG.info("TRO pipeline complete.")


if __name__ == "__main__":
    asyncio.run(main())
