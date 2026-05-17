# TRO Digest Database — Schema & Query Reference

> For AI agents and tools querying the TRO SQLite digest database.

---

## 1. Schema Overview

```
digests ──┐
          ├──< tweets ──< tweets_fts     (FTS5 full-text)
          │         └──< tweets_vec      (vec0 vector)
          │
          └──< sections ──< sections_fts (FTS5 full-text)
                       └──< sections_vec (vec0 vector)
```

Every run of `tro.py` produces one row in `digests`. That row has many `tweets` and many `sections`. FTS5 and vec0 virtual tables link to their parent rows by `rowid`.

**Section types** stored in `sections.section_type`:

| Value | Description | Split behavior |
|---|---|---|
| `executive_summary` | Digest section 1 | One row per digest |
| `topic_cluster` | One row per `### Heading` under section 2 | Multiple rows |
| `top_5` | Digest section 3 | One row |
| `themes` | Digest section 4 | One row |
| `noise` | Digest section 5 | One row |
| `followup` | Digest section 6 | One row |

---

## 2. Table Reference

### `digests`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment ID |
| `generated_at` | TEXT | ISO 8601 timestamp |
| `num_tweets` | INTEGER | Requested tweet count |
| `num_days` | INTEGER | Lookback window in days |
| `search_model` | TEXT | x.ai model used for search |
| `analysis_model` | TEXT | OpenAI-compatible model used for analysis |
| `embedding_model` | TEXT | Model used for vector embeddings; NULL if not configured |
| `raw_markdown` | TEXT | Full generated digest in Markdown |

### `tweets`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment ID |
| `digest_id` | INTEGER FK | References `digests(id)` |
| `tweet_index` | INTEGER | `[N]` from the Sampled Tweets list |
| `url` | TEXT | Full `https://x.com/user/status/…` URL |
| `text` | TEXT | Full tweet content |
| `author_handle` | TEXT | `@username` (without the @) |
| `likes` | INTEGER | Like count at time of fetch |
| `retweets` | INTEGER | Retweet count at time of fetch |

### `sections`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment ID |
| `digest_id` | INTEGER FK | References `digests(id)` |
| `section_type` | TEXT | One of the six types above |
| `section_title` | TEXT NULLABLE | Cluster name; `NULL` for single-item sections |
| `content` | TEXT | Full section text |

### `tweets_fts` / `sections_fts` (FTS5 virtual tables)

| Column | Source |
|---|---|
| `text` / `content` | Full tweet or section text |
| `author_handle` / `section_title` | Author or section title |
| **`rowid`** | == `tweets.id` or `sections.id` (the join key) |

Tokenizer: `porter unicode61` (stemming + Unicode-aware tokenization).

### `tweets_vec` / `sections_vec` (vec0 virtual tables)

| Column | Description |
|---|---|
| `embedding` | `float[1536]` (default) or per-model dimension |
| **`rowid`** | == `tweets.id` or `sections.id` (the join key) |

Dimension is set by the configured embedding model (env `TRO_EMBEDDING_MODEL`). If embeddings are not configured, these tables exist but may be empty.

---

## 3. Search Capabilities

### 3.1 Full-text search (FTS5)

```sql
-- Search tweets by keyword
SELECT t.id, t.url, t.text, t.author_handle
FROM tweets_fts fts
JOIN tweets t ON t.id = fts.rowid
WHERE tweets_fts MATCH 'reasoning OR architecture'
ORDER BY rank;

-- Search sections
SELECT s.section_type, s.section_title, s.content
FROM sections_fts fts
JOIN sections s ON s.id = fts.rowid
WHERE sections_fts MATCH 'MCP OR "model context protocol"'
ORDER BY rank;

-- Search with negation
WHERE sections_fts MATCH 'agent -hypothesis'
```

FTS5 supports: boolean operators (`AND`, `OR`, `NOT`), phrase queries (`"exact phrase"`), prefix queries (`token*`), and column-specific matches (`content:keyword`).

### 3.2 Semantic vector search (vec0)

```sql
-- K-nearest-neighbour on sections
SELECT s.id, s.section_type, s.section_title, distance
FROM sections_vec v
JOIN sections s ON s.id = v.rowid
WHERE embedding MATCH '[0.1, 0.2, ...]'
ORDER BY distance
LIMIT 5;

-- KNN on tweets
SELECT t.id, t.url, t.text, distance
FROM tweets_vec v
JOIN tweets t ON t.id = v.rowid
WHERE embedding MATCH '<query_embedding_vector>'
ORDER BY distance
LIMIT 10;
```

Distance is cosine distance (0 = identical, 2 = opposite). Lower is better.

### 3.3 Combined FTS5 + vector

```sql
-- FTS5 pre-filter, then vector refine
SELECT s.id, s.section_type, s.content, distance
FROM sections_fts fts
JOIN sections s ON s.id = fts.rowid
JOIN sections_vec v ON v.rowid = s.id
WHERE sections_fts MATCH 'MCP'
  AND v.embedding MATCH '<query_embedding>'
ORDER BY distance
LIMIT 5;
```

---

## 4. Using This Database with AI Agents

### 4.1 Quick start: bootstrap an agent connection

Generate a digest and store it:

```bash
python tro.py -n 50 -d 7 -f markdown --db       # stores in ./tro.db
python tro.py -n 50 -d 7 -f markdown --db my.db  # stores in my.db
python tro.py -n 50 -d 7 -f markdown              # no DB storage
```

Connect from Python:

```python
import sqlite3
import sqlite_vec

DB_PATH = "tro.db"

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn
```

Give the agent the file path and the schema reference above. Start with a context summary query to orient the agent before deeper queries:

```sql
-- Agent orientation: what's available?
SELECT
    COUNT(*)                        AS total_digests,
    MIN(generated_at)               AS earliest,
    MAX(generated_at)               AS latest,
    COUNT(DISTINCT analysis_model)  AS models_used
FROM digests;
```

### 4.2 Database topology (for agent reasoning)

The database models a **temporal tree**:

```
Time axis: generated_at (in digests)
           │
           ▼
  digest_2026-05-10  digest_2026-05-11  digest_2026-05-12  ...
       │                   │                   │
       ├── tweets          ├── tweets          ├── tweets
       │   ├── [1] url     │   ├── [1] url     │   ...
       │   ├── [2] url     │   ├── [2] url
       │   └── ...         │   └── ...
       │                   │
       └── sections        └── sections
           ├── exec_summary    ├── exec_summary
           ├── cluster: MCP    ├── cluster: Tools
           ├── cluster: OSS    ├── cluster: Agents
           ├── top_5           ├── top_5
           ├── themes          ├── themes
           ├── noise           ├── noise
           └── followup        └── followup
```

An agent can traverse this tree:
- **Down**: from a digest to its tweets and sections
- **Across**: comparing the same section type across digests
- **Up**: from a tweet URL to the digest context that produced it

### 4.3 Recommended agent workflow

| Phase | Goal | Queries |
|---|---|---|
| **1. Orient** | Understand the dataset scope | `SELECT COUNT(*), MIN/MAX date FROM digests` |
| **2. Frame** | Read recent executive summaries | Query 5.5 — gives high-level context |
| **3. Explore** | Identify recurring topics | Query 5.4 — trending clusters |
| **4. Search** | Drill into a specific topic | FTS5 (section 3.1) + vec0 KNN (section 3.2) |
| **5. Evidence** | Pull source tweets backing a claim | Query 5.8 — high-engagement tweets on topic |
| **6. Cross-ref** | Check if finding persists across time | Repeat step 4 with `generated_at >= date('now', '-30 days')` |
| **7. Synthesize** | Combine exec summaries + clusters + themes | Query 5.5 + 4.4 + 4.9 into LLM context window |

### 4.4 Building context windows for LLM synthesis

When an agent needs to answer a user question, build a context payload from multiple queries:

```python
def build_context_for_question(conn, question: str, days: int = 14) -> dict:
    """Assemble structured context for an LLM to answer a question."""
    return {
        "recent_state": [
            dict(r) for r in conn.execute("""
                SELECT d.generated_at, s.content
                FROM sections s JOIN digests d ON d.id = s.digest_id
                WHERE s.section_type = 'executive_summary'
                  AND d.generated_at >= date('now', ?)
                ORDER BY d.generated_at DESC
                LIMIT 3
            """, (f'-{days} days',)).fetchall()
        ],
        "trending_clusters": [
            dict(r) for r in conn.execute("""
                SELECT s.section_title, COUNT(*) AS freq
                FROM sections s JOIN digests d ON d.id = s.digest_id
                WHERE s.section_type = 'topic_cluster'
                  AND d.generated_at >= date('now', ?)
                GROUP BY s.section_title ORDER BY freq DESC LIMIT 10
            """, (f'-{days} days',)).fetchall()
        ],
        "key_tweets": [
            dict(r) for r in conn.execute("""
                SELECT d.generated_at, s.content
                FROM sections s JOIN digests d ON d.id = s.digest_id
                WHERE s.section_type = 'top_5'
                  AND d.generated_at >= date('now', ?)
                ORDER BY d.generated_at DESC LIMIT 5
            """, (f'-{days} days',)).fetchall()
        ],
        "weak_signals": [
            dict(r) for r in conn.execute("""
                SELECT d.generated_at, s.content
                FROM sections s JOIN digests d ON d.id = s.digest_id
                WHERE s.section_type = 'themes'
                  AND d.generated_at >= date('now', ?)
                ORDER BY d.generated_at DESC LIMIT 5
            """, (f'-{days} days',)).fetchall()
        ],
    }
```

Then pass to the LLM:

```python
context = build_context_for_question(conn, user_question, days=14)
system_prompt = """You are an AI industry analyst with access to a database 
of daily AI/LLM tweet digests. Use the structured context below to answer 
the user's question. Cite specific digests and dates when making claims."""

user_prompt = f"""Question: {user_question}

Database context:
{json.dumps(context, indent=2, default=str)}
"""
```

### 4.5 Semantic agent search (vector workflow)

For "find me everything like this" queries where keywords are insufficient:

```
User asks: "What are the latest memory management approaches for LLMs?"

Agent pipeline:
  1. Query digests to discover which embedding_model was used:
     `SELECT DISTINCT embedding_model FROM digests WHERE embedding_model IS NOT NULL LIMIT 1`
  2. Embed the user's question using the same model
  3. KNN on sections_vec (section 3.2) — find top 5 matching sections
  4. Join back to digests to get dates and full context
  5. KNN on tweets_vec — find top 10 matching tweets
  6. Combine results into a single context payload
  7. Send to LLM with instructions to synthesize and cite
```

```sql
-- Step 2: semantic section search
SELECT s.section_type, s.section_title, s.content, d.generated_at, distance
FROM sections_vec v
JOIN sections s ON s.id = v.rowid
JOIN digests d ON d.id = s.digest_id
WHERE v.embedding MATCH '<question_embedding>'
ORDER BY distance
LIMIT 5;

-- Step 4: semantic tweet search
SELECT t.url, t.text, t.author_handle, t.likes, d.generated_at, distance
FROM tweets_vec v
JOIN tweets t ON t.id = v.rowid
JOIN digests d ON d.id = t.digest_id
WHERE v.embedding MATCH '<question_embedding>'
ORDER BY distance
LIMIT 10;
```

### 4.6 Multi-turn agent interactions

For follow-up questions, maintain conversation state by narrowing the temporal or topic scope:

```sql
-- "Tell me more about that" → narrow to specific cluster
SELECT s.content FROM sections s
WHERE s.section_type = 'topic_cluster'
  AND s.section_title = 'Memory & State Management'
  AND s.digest_id IN (
    SELECT id FROM digests WHERE generated_at >= date('now', '-14 days')
  )
ORDER BY s.digest_id DESC;

-- "Who said that?" → resolve tweet reference
SELECT t.text, t.author_handle, t.url, d.generated_at
FROM tweets t JOIN digests d ON d.id = t.digest_id
WHERE t.tweet_index = 3        -- from [N] in digest
  AND d.generated_at = '...';  -- from the cited digest
```

### 4.7 Example system prompt for a research agent

```
You are an AI research analyst with SQL access to a database of daily 
AI/LLM tweet digests (table: tro.db). 

Database contents:
- One digest per day, each containing ~5-50 analyzed tweets about AI
- Each digest is parsed into sections: executive_summary, topic_cluster 
  (6+), top_5, themes, noise, followup
- Full-text search (FTS5) is available on tweets and sections
- Vector search (vec0) is available if embeddings were generated

Schema reference: see sections 1-2 of the reference document.

Your workflow:
1. Start by checking the date range and digest count (Query 5.1)
2. Read the 2-3 most recent executive summaries for context (Query 5.5)
3. For specific topics, run FTS5 keyword search (Section 3.1)
4. For "similar to" or "trends like", use vector KNN (Section 3.2)
5. Always cite which digest and date a finding came from
6. If you find an interesting cluster (Query 5.4), cross-reference 
   the underlying tweets (Query 5.8) for evidence
7. Use the noise section (Query 5.9) to avoid amplifying hype
8. Synthesize across time: a single digest is a snapshot; 7+ digests 
   is a trend

Output expectations:
- Start with a 2-3 sentence bottom line
- Support claims with specific digest dates and tweet references
- Distinguish between consensus (appears in 3+ digests) and outlier 
  (appears in 1)
- Call out when data is stale (digest older than 7 days) vs fresh
```

### 4.8 Verifying agent output

Spot-check the agent's claims by running the source query directly:

```sql
-- "The agent says MCP was dominant May 10-15. Verify."
SELECT date(d.generated_at) AS day, COUNT(*) AS mcp_mentions
FROM sections_fts fts
JOIN sections s ON s.id = fts.rowid
JOIN digests d ON d.id = s.digest_id
WHERE sections_fts MATCH 'MCP'
  AND d.generated_at BETWEEN '2026-05-10' AND '2026-05-16'
GROUP BY day
ORDER BY day;
```

---

## 5. Agent Query Examples

### 5.1 List recent digests

```sql
SELECT id, generated_at, num_tweets, num_days, search_model, analysis_model
FROM digests
ORDER BY generated_at DESC
LIMIT 20;
```

### 5.2 Keyword search across all sections in recent digests

```sql
SELECT s.section_type, s.section_title, s.content, d.generated_at
FROM sections_fts fts
JOIN sections s ON s.id = fts.rowid
JOIN digests d ON d.id = s.digest_id
WHERE sections_fts MATCH 'open source'
  AND d.generated_at >= date('now', '-30 days')
ORDER BY rank, d.generated_at DESC
LIMIT 20;
```

### 5.3 Find tweets citing a specific handle across all digests

```sql
SELECT t.tweet_index, t.url, t.text, t.likes, t.retweets, d.generated_at
FROM tweets t
JOIN digests d ON d.id = t.digest_id
WHERE t.author_handle = 'karpathy'
ORDER BY d.generated_at DESC;
```

### 5.4 Trend analysis: topic cluster frequency over time

```sql
SELECT s.section_title, COUNT(*) AS occurrences
FROM sections s
JOIN digests d ON d.id = s.digest_id
WHERE s.section_type = 'topic_cluster'
  AND d.generated_at >= date('now', '-14 days')
GROUP BY s.section_title
ORDER BY occurrences DESC;
```

### 5.5 Extract all executive summaries from the past 7 days

```sql
SELECT d.id, d.generated_at, s.content
FROM sections s
JOIN digests d ON d.id = s.digest_id
WHERE s.section_type = 'executive_summary'
  AND d.generated_at >= date('now', '-7 days')
ORDER BY d.generated_at DESC;
```

### 5.6 Find tweets that were marked as most significant

```sql
SELECT d.generated_at, s.content
FROM sections s
JOIN digests d ON d.id = s.digest_id
WHERE s.section_type = 'top_5'
  AND d.generated_at >= date('now', '-7 days')
ORDER BY d.generated_at DESC;
```

### 5.7 Discover digests where a specific model was mentioned

```sql
SELECT d.id, d.generated_at
FROM digests d
JOIN sections_fts fts ON fts.rowid IN (
    SELECT id FROM sections WHERE digest_id = d.id
)
WHERE sections_fts MATCH 'Claude OR Gemini OR GPT'
GROUP BY d.id
ORDER BY d.generated_at DESC;
```

### 5.8 Compare engagement of tweets about a topic across digests

```sql
SELECT d.generated_at, t.tweet_index, t.author_handle, t.likes, t.retweets, t.text
FROM tweets t
JOIN tweets_fts fts ON fts.rowid = t.id
JOIN digests d ON d.id = t.digest_id
WHERE tweets_fts MATCH 'agent*'
ORDER BY t.likes DESC
LIMIT 25;
```

### 5.9 Get all noise/hype signals from recent digests

```sql
SELECT d.generated_at, s.content
FROM sections s
JOIN digests d ON d.id = s.digest_id
WHERE s.section_type = 'noise'
  AND d.generated_at >= date('now', '-14 days')
ORDER BY d.generated_at DESC;
```

### 5.10 Find recommended follow-up queries across digests

```sql
SELECT d.generated_at, s.content
FROM sections s
JOIN digests d ON d.id = s.digest_id
WHERE s.section_type = 'followup'
  AND d.generated_at >= date('now', '-14 days')
ORDER BY d.generated_at DESC;
```

### 5.11 Retrieve the full digest as Markdown

```sql
SELECT raw_markdown FROM digests WHERE id = 5;
```

### 5.12 Count tweets per day to see coverage trends

```sql
SELECT date(d.generated_at) AS day, COUNT(*) AS tweet_count
FROM tweets t
JOIN digests d ON d.id = t.digest_id
GROUP BY day
ORDER BY day DESC
LIMIT 30;
```

---

## 6. Extension: Loading sqlite-vec

When connecting to the database from outside `tro.py`, load the sqlite-vec extension first:

```python
import sqlite3
import sqlite_vec

conn = sqlite3.connect("tro.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)
```

From raw SQL: `.load ./vec0` (if the shared library is in the path).

---

## 7. Schema DDL (for reference)

```sql
CREATE TABLE digests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at    TEXT NOT NULL,
    num_tweets      INTEGER,
    num_days        INTEGER,
    search_model    TEXT,
    analysis_model  TEXT,
    embedding_model TEXT,
    raw_markdown    TEXT
);

CREATE TABLE tweets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_id       INTEGER NOT NULL REFERENCES digests(id),
    tweet_index     INTEGER NOT NULL,
    url             TEXT NOT NULL,
    text            TEXT NOT NULL,
    author_handle   TEXT,
    likes           INTEGER DEFAULT 0,
    retweets        INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE tweets_fts USING fts5(
    text, author_handle,
    content='', content_rowid='',
    tokenize='porter unicode61'
);

CREATE TABLE sections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_id       INTEGER NOT NULL REFERENCES digests(id),
    section_type    TEXT NOT NULL,
    section_title   TEXT,
    content         TEXT NOT NULL
);

CREATE VIRTUAL TABLE sections_fts USING fts5(
    content, section_title,
    content='', content_rowid='',
    tokenize='porter unicode61'
);

-- Created by sqlite-vec on first tro.py run:
CREATE VIRTUAL TABLE tweets_vec USING vec0(embedding float[1536]);
CREATE VIRTUAL TABLE sections_vec USING vec0(embedding float[1536]);
```
