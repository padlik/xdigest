# AI Digest Index — Schema & Query Reference

This document describes the SQLite database schema created by `ai_digest.py --db`
and provides ready-to-run query examples for trend analysis, semantic search, and
automated report generation.

---

## Setup

```bash
# Create / update the database on every digest run
uv run ai_digest.py --format md --db ai_digest.db

# The DB is a single portable file — copy, back up, or version-control freely
```

---

## Schema

### `digests` — one row per pipeline run

| Column        | Type    | Description |
|---------------|---------|-------------|
| `id`          | INTEGER | Primary key |
| `date`        | TEXT    | `YYYY-MM-DD` (UTC) |
| `run`         | INTEGER | `1` or `2` when run twice per day |
| `generated_at`| TEXT    | ISO-8601 timestamp |
| `topics`      | TEXT    | JSON array of search topics |
| `tweet_count` | INTEGER | Number of enriched tweets |
| `llm_model`   | TEXT    | Model used for analysis |
| `embed_model` | TEXT    | Model used for embeddings |
| `raw_md`      | TEXT    | Full Markdown digest |
| `raw_json`    | TEXT    | Full JSON dataset including all tweet fields |

### `sections` — one row per Markdown section within a digest

| Column      | Type    | Description |
|-------------|---------|-------------|
| `id`        | INTEGER | Primary key (also the FTS rowid) |
| `digest_id` | INTEGER | FK → `digests.id` |
| `date`      | TEXT    | `YYYY-MM-DD` — denormalised for fast date filtering |
| `run`       | INTEGER | `1` or `2` |
| `section`   | TEXT    | Canonical slug (see table below) |
| `heading`   | TEXT    | Original heading text from the LLM |
| `body`      | TEXT    | Section body text |
| `embedding` | BLOB    | `float32` vector serialised by `sqlite-vec` |

**Section slugs**

| Slug                | Typical heading |
|---------------------|-----------------|
| `executive_summary` | Executive Summary |
| `topic_clusters`    | Topic Clusters |
| `top_tweets`        | Top 5 Most Significant Tweets |
| `emerging_themes`   | Emerging Themes & Weak Signals |
| `noise`             | Noise / Hype to Ignore |
| `follow_up`         | Recommended Follow-Up |

### `sections_fts` — FTS5 virtual table (auto-synced with `sections`)

Columns mirrored: `heading`, `body`.  
Query via `sections_fts MATCH '...'` — ranks by BM25 automatically.

### `tweet_refs` — one row per tweet per digest

| Column      | Type    | Description |
|-------------|---------|-------------|
| `id`        | INTEGER | Primary key |
| `digest_id` | INTEGER | FK → `digests.id` |
| `date`      | TEXT    | `YYYY-MM-DD` |
| `run`       | INTEGER | `1` or `2` |
| `tweet_n`   | INTEGER | Position `[N]` as cited in the digest |
| `tweet_id`  | TEXT    | X/Twitter numeric tweet ID |
| `url`       | TEXT    | Full `https://x.com/...` URL |
| `author`    | TEXT    | `screen_name` (without `@`) |
| `text`      | TEXT    | Full tweet text |
| `likes`     | INTEGER | Like count at time of retrieval |
| `retweets`  | INTEGER | Retweet count |
| `replies`   | INTEGER | Reply count |
| `topic_tag` | TEXT    | Search topic that surfaced this tweet |
| `has_media` | INTEGER | `1` if tweet has photos or videos |

---

## Query Examples

All examples use plain SQLite (Python `sqlite3` module or any SQL client).
For vector queries, the `sqlite-vec` extension must be loaded first.

```python
import sqlite3, sqlite_vec, struct

db = sqlite3.connect("ai_digest.db")
db.row_factory = sqlite3.Row
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
```

---

### 1. Latest digest summary

```sql
SELECT date, run, tweet_count, llm_model,
       substr(raw_md, 1, 500) AS preview
FROM   digests
ORDER  BY date DESC, run DESC
LIMIT  1;
```

---

### 2. Keyword trend — mentions per day

How often did a term appear in section bodies over the past 30 days?

```sql
SELECT s.date,
       count(*) AS mentions
FROM   sections_fts f
JOIN   sections s ON s.id = f.rowid
WHERE  sections_fts MATCH 'MCP OR "model context protocol"'
  AND  s.date >= date('now', '-30 days')
GROUP  BY s.date
ORDER  BY s.date;
```

---

### 3. Keyword trend — weekly aggregation

```sql
SELECT strftime('%Y-W%W', s.date) AS week,
       count(*)                   AS mentions
FROM   sections_fts f
JOIN   sections s ON s.id = f.rowid
WHERE  sections_fts MATCH 'agent OR "agentic"'
GROUP  BY week
ORDER  BY week;
```

---

### 4. Most mentioned authors across all digests

```sql
SELECT author,
       count(*)        AS appearances,
       sum(likes)      AS total_likes,
       sum(retweets)   AS total_retweets,
       max(date)       AS last_seen
FROM   tweet_refs
WHERE  author IS NOT NULL
GROUP  BY author
ORDER  BY appearances DESC
LIMIT  20;
```

---

### 5. Top tweets by engagement for a date range

```sql
SELECT date, author, url,
       likes + retweets * 2 AS score,
       substr(text, 1, 120) AS preview
FROM   tweet_refs
WHERE  date BETWEEN '2026-05-01' AND '2026-05-31'
ORDER  BY score DESC
LIMIT  10;
```

---

### 6. Semantic search — find sections similar to a concept

```python
def search(query_text: str, top_k: int = 5) -> list[sqlite3.Row]:
    # Embed the query with the same model used at index time
    from openai import OpenAI
    client   = OpenAI(api_key="...", base_url="...")
    response = client.embeddings.create(model="text-embedding-3-small",
                                        input=[query_text])
    floats   = response.data[0].embedding
    blob     = struct.pack(f"{len(floats)}f", *floats)

    return db.execute("""
        SELECT s.date, s.run, s.section, s.heading,
               substr(s.body, 1, 300)              AS preview,
               vec_distance_cosine(s.embedding, ?) AS distance
        FROM   sections s
        WHERE  s.embedding IS NOT NULL
        ORDER  BY distance
        LIMIT  ?
    """, (blob, top_k)).fetchall()

results = search("inference cost optimisation at scale")
for r in results:
    print(r["date"], r["section"], r["distance"], r["preview"])
```

---

### 7. Weekly trend report — pull last 7 executive summaries for LLM synthesis

```python
rows = db.execute("""
    SELECT d.date, d.run, s.body
    FROM   sections s
    JOIN   digests  d ON d.id = s.digest_id
    WHERE  s.section = 'executive_summary'
      AND  d.date   >= date('now', '-7 days')
    ORDER  BY d.date, d.run
""").fetchall()

context = "\n\n---\n\n".join(
    f"### {r['date']} (run {r['run']})\n{r['body']}" for r in rows
)

# Feed to Claude or any LLM
prompt = f"""
You are an AI research analyst.
Below are the executive summaries from the last 7 daily AI/LLM digests.
Write a concise weekly trend report (400-600 words) highlighting:
- The dominant themes this week
- Notable shifts or accelerations vs. previous days
- The 2-3 most significant developments

{context}
"""
```

---

### 8. Emerging themes — cross-digest signal detection

Find terms that appear in **emerging_themes** sections across multiple days.

```sql
SELECT fts.term,
       count(DISTINCT s.date) AS days_mentioned,
       min(s.date)            AS first_seen,
       max(s.date)            AS last_seen
FROM   sections_fts,
       -- FTS auxiliary function for term extraction
       json_each(
           '[' || replace(sections_fts.body, ' ', '","') || ']'
       ) AS fts
JOIN   sections s ON s.id = sections_fts.rowid
WHERE  s.section = 'emerging_themes'
  AND  length(fts.value) > 4
GROUP  BY fts.term
HAVING days_mentioned >= 3
ORDER  BY days_mentioned DESC
LIMIT  30;
```

> **Note:** For production term extraction, prefer Python `collections.Counter`
> over SQLite string splitting (cleaner tokenisation, stop-word removal).

---

### 9. Topic velocity — which search topic is accelerating?

```sql
SELECT topic_tag,
       strftime('%Y-W%W', date) AS week,
       count(*)                 AS tweet_count,
       avg(likes)               AS avg_likes
FROM   tweet_refs
GROUP  BY topic_tag, week
ORDER  BY topic_tag, week;
```

---

### 10. Full-text search across all digest bodies (agent query)

```sql
-- Useful for agents answering "what have we said about X?"
SELECT d.date, d.run, s.section, s.heading,
       snippet(sections_fts, 1, '**', '**', '…', 20) AS excerpt
FROM   sections_fts
JOIN   sections s ON s.id    = sections_fts.rowid
JOIN   digests  d ON d.id    = s.digest_id
WHERE  sections_fts MATCH '"tool calling" OR "function calling"'
ORDER  BY d.date DESC, rank;
```

---

### 11. Re-analyse a past digest with a new prompt (no re-fetching)

```python
# Pull raw JSON for any past date
row = db.execute("""
    SELECT raw_json FROM digests
    WHERE  date = '2026-05-10' AND run = 1
""").fetchone()

data         = json.loads(row["raw_json"])
new_analysis = analyse_tweets(data)          # calls LLM with new prompt
print(new_analysis)
```

---

### 12. Export last N days to Markdown for sharing

```python
rows = db.execute("""
    SELECT raw_md FROM digests
    ORDER  BY date DESC, run DESC
    LIMIT  7
""").fetchall()

combined = "\n\n---\n\n".join(r["raw_md"] for r in reversed(rows))
with open("weekly_digest.md", "w") as f:
    f.write(combined)
```

---

## Agent Integration Notes

When building an agent that queries this database, provide the agent with:

1. **This document** as context (or a condensed version of the schema section).
2. The **section slugs table** so it knows which `section` value to filter on.
3. The **date format** (`YYYY-MM-DD`) to avoid ambiguity in date comparisons.
4. Tell the agent to **always join `sections` → `digests`** when it needs the date,
   since `sections.date` is denormalised for convenience but `digests` is the source
   of truth for model/run metadata.
5. For vector queries, the agent must call the embedding API with the **same model**
   stored in `digests.embed_model` for the rows it is searching.

### Minimal agent system prompt snippet

```
You have access to a SQLite database `ai_digest.db` containing daily AI/LLM
Twitter digests. Tables: digests, sections, sections_fts, tweet_refs.
Key columns:
  sections.section  — one of: executive_summary, topic_clusters, top_tweets,
                      emerging_themes, noise, follow_up
  sections.date     — YYYY-MM-DD
  tweet_refs.author — screen name without @
  tweet_refs.url    — full x.com URL

For keyword search use: SELECT ... FROM sections_fts WHERE sections_fts MATCH '...'
For date ranges use:    WHERE date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
For trends use:         GROUP BY strftime('%Y-W%W', date)
Always ORDER BY date DESC unless asked for oldest-first.
```
