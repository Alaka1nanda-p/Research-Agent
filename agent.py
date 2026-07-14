"""
agent.py — Core AI agent logic: query understanding, search, summarization,
           report generation, hypothesis suggestion, and data extraction.
           All LLM calls go through this module with caching + token tracking.
"""
import os
import re
import json
import time
import hashlib
import requests
import feedparser
from typing import Optional

from dotenv import load_dotenv

import database as db

load_dotenv()

# ── AGENT INSTRUCTIONS (customise these to change agent behaviour) ──────────
AGENT_INSTRUCTIONS = {
    "tone": "academic and concise",
    "default_citation_style": "APA",
    "summarization_depth": "brief",           # brief | detailed
    "report_structure": ["Introduction", "Related Work", "Summary Table", "Conclusion"],
    "language": "English",
    "hypothesis_count": 3,
}

# ── Model / token config ─────────────────────────────────────────────────────
# NOTE: ibm/granite-8b-code-instruct is the model actually available/working
# in this project's region (Sydney / au-syd). It is instruct-tuned, so it
# DOES follow instructions, but it responds much more reliably to a plain
# "Question: ... \n\nAnswer:" style prompt than to the <|system|>/<|user|>/
# <|assistant|> chat-template style (which returned empty completions in
# testing). This model is also flagged deprecated by IBM (withdrawn from
# 2026-08-08) and its decoding parameters are not fully honoured by the
# API ("decoding_method is ignored and set automatically" warning), which
# makes long, multi-section generations less reliable than short ones —
# see generate_report() below for how this is worked around.
GRANITE_MODEL_ID   = os.getenv("GRANITE_MODEL_ID", "ibm/granite-8b-code-instruct")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://au-syd.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
IBM_API_KEY        = os.getenv("IBM_API_KEY", "")
MAX_TOKENS_DEFAULT = int(os.getenv("GRANITE_MAX_NEW_TOKENS_DEFAULT", "300"))
MAX_PAPERS         = int(os.getenv("MAX_PAPERS_PER_SEARCH", "5"))
ABSTRACT_MAX_CHARS = int(os.getenv("ABSTRACT_MAX_CHARS", "1000"))
MAX_TOKENS_PER_DAY = int(os.getenv("MAX_TOKENS_PER_DAY", "50000"))

# Per-call token caps
TOKEN_CAPS = {
    "summary":     150,
    "report":      400,
    "hypothesis":  280,
    "extraction":  250,
    "query_parse": 150,
    "draft":       350,
}

# In-memory IAM token cache
_iam_token_cache = {"token": None, "expires_at": 0}


# ═══════════════════════════════════════════════════════════════════════════
# IAM Auth helper
# ═══════════════════════════════════════════════════════════════════════════

def _get_iam_token() -> str:
    """Return a cached or freshly-fetched IAM bearer token."""
    now = time.time()
    if _iam_token_cache["token"] and now < _iam_token_cache["expires_at"] - 30:
        return _iam_token_cache["token"]

    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": IBM_API_KEY,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    _iam_token_cache["token"] = data["access_token"]
    _iam_token_cache["expires_at"] = now + int(data.get("expires_in", 3600))
    return _iam_token_cache["token"]


# ═══════════════════════════════════════════════════════════════════════════
# Core LLM caller (with cache + token tracking)
# ═══════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str, call_type: str, session_id: str,
              max_new_tokens: Optional[int] = None) -> str:
    """
    Send a prompt to Granite via the Watsonx.ai REST API.
    Responses are cached in SQLite by prompt hash.
    Token usage is recorded per session.
    """
    max_tokens = max_new_tokens or TOKEN_CAPS.get(call_type, MAX_TOKENS_DEFAULT)

    # Check cache first
    cache_key = db.cache_key_for(f"{call_type}::{GRANITE_MODEL_ID}::{prompt}")
    cached = db.get_cached_response(cache_key)
    if cached:
        return cached

    # Check daily limit
    daily_used = db.get_daily_token_total()
    if daily_used >= MAX_TOKENS_PER_DAY:
        return "[Token limit reached for today. Please try again tomorrow.]"

    if not IBM_API_KEY or not WATSONX_PROJECT_ID:
        return "[LLM not configured — set IBM_API_KEY and WATSONX_PROJECT_ID in .env]"

    try:
        token = _get_iam_token()
        url = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"
        payload = {
            "model_id": GRANITE_MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": max_tokens,
                "stop_sequences": ["Question:", "\n\nQuestion"],
                "repetition_penalty": 1.15 if call_type == "report" else 1.1,
            },
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        result_text = data["results"][0]["generated_text"].strip()

        usage      = data.get("results", [{}])[0]
        in_count   = usage.get("input_token_count", len(prompt.split()) * 4 // 3)
        out_count  = usage.get("generated_token_count", max_tokens // 2)
        db.record_token_usage(session_id, call_type, in_count, out_count, GRANITE_MODEL_ID)

        if not result_text:
            return "[LLM returned an empty response. Try rephrasing your query.]"

        db.set_cached_response(cache_key, result_text)
        return result_text

    except requests.exceptions.Timeout:
        return "[LLM request timed out. Please try again.]"
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else 0
        detail = ""
        try:
            detail = e.response.text[:200] if e.response is not None else ""
        except Exception:
            pass
        if status == 429:
            return "[Rate limited by Watsonx. Please wait a moment and retry.]"
        return f"[LLM API error {status}: {detail or str(e)[:120]}]"
    except Exception as e:
        return f"[LLM error: {str(e)[:120]}]"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Query Understanding
# ═══════════════════════════════════════════════════════════════════════════

def _simple_keyword_extract(query: str) -> list[str]:
    """
    Pure-Python keyword extraction. Strips stopwords and punctuation.
    Used as primary extractor; LLM is only the fallback.
    """
    stopwords = {
        "a", "an", "the", "in", "on", "at", "to", "of", "for", "is", "are",
        "was", "were", "be", "been", "with", "and", "or", "but", "that",
        "this", "it", "its", "from", "by", "as", "how", "what", "when",
        "where", "which", "who", "do", "does", "use", "using", "can", "will",
        "would", "could", "should", "have", "has", "had", "about", "into",
        "through", "during", "between", "each", "over", "under",
    }
    words = re.sub(r"[^\w\s]", " ", query.lower()).split()
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    seen, unique = set(), []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique[:10]


def _is_ambiguous_query(query: str) -> bool:
    """Return True if the query is short/vague and likely needs LLM parsing."""
    words = query.strip().split()
    return len(words) <= 3 or "?" not in query and len(words) <= 5


def parse_research_query(query: str, session_id: str) -> dict:
    """
    Extract keywords, topic area, and sub-questions from a research query.
    Uses plain NLP first; falls back to Granite only for ambiguous queries.
    """
    simple_kw = _simple_keyword_extract(query)

    if not _is_ambiguous_query(query):
        return {
            "keywords": simple_kw,
            "topic_area": " ".join(simple_kw[:3]).title(),
            "sub_questions": [],
            "original_query": query,
            "llm_used": False,
        }

    prompt = f"""Question: You are a research assistant. Extract structured metadata from the research query below. Respond ONLY with a single valid JSON object — no markdown, no explanation, no extra text before or after.

Query: "{query}"

Return exactly this JSON shape:
{{"keywords": ["term1", "term2"], "topic_area": "one-line description", "sub_questions": ["optional sub-question"]}}

Answer:"""
    raw = _call_llm(prompt, "query_parse", session_id, max_new_tokens=150)
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            return {
                "keywords":      parsed.get("keywords", simple_kw),
                "topic_area":    parsed.get("topic_area", ""),
                "sub_questions": parsed.get("sub_questions", []),
                "original_query": query,
                "llm_used": True,
            }
    except (json.JSONDecodeError, AttributeError):
        pass

    return {
        "keywords": simple_kw,
        "topic_area": " ".join(simple_kw[:3]).title(),
        "sub_questions": [],
        "original_query": query,
        "llm_used": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Literature Search
# ═══════════════════════════════════════════════════════════════════════════

def _search_arxiv(keywords: list[str], max_results: int = 5) -> list[dict]:
    """Query the arXiv Atom API (no key needed)."""
    query_str = "+".join(keywords[:6])
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=all:{query_str}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )
    try:
        feed = feedparser.parse(url)
        papers = []
        for entry in feed.entries[:max_results]:
            arxiv_id = entry.get("id", "").split("/abs/")[-1].split("v")[0]
            authors = [a.get("name", "") for a in entry.get("authors", [])]
            year = None
            published = entry.get("published", "")
            if published:
                try:
                    year = int(published[:4])
                except ValueError:
                    pass
            papers.append({
                "paper_id": f"arxiv:{arxiv_id}",
                "title":    entry.get("title", "").replace("\n", " ").strip(),
                "authors":  authors,
                "year":     year,
                "abstract": entry.get("summary", "").replace("\n", " ").strip(),
                "url":      entry.get("link", f"https://arxiv.org/abs/{arxiv_id}"),
                "venue":    "arXiv",
                "source":   "arxiv",
            })
        return papers
    except Exception as e:
        print(f"[arXiv search error] {e}")
        return []


def _search_semantic_scholar(keywords: list[str], max_results: int = 5) -> list[dict]:
    """Query the Semantic Scholar public API (no key needed)."""
    query_str = " ".join(keywords[:6])
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":  query_str,
        "limit":  max_results,
        "fields": "title,authors,year,abstract,externalIds,venue,openAccessPdf,url",
    }
    headers = {"User-Agent": "ResearchAgent/1.0 (academic research tool)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 429:
            time.sleep(3)
            resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        papers = []
        for item in data.get("data", [])[:max_results]:
            ss_id   = item.get("paperId", "")
            authors = [a.get("name", "") for a in item.get("authors", [])]
            pdf_url = (item.get("openAccessPdf") or {}).get("url", "")
            pub_url = item.get("url", f"https://www.semanticscholar.org/paper/{ss_id}")
            ext_ids = item.get("externalIds", {}) or {}
            arxiv_id = ext_ids.get("ArXiv", "")
            final_url = pdf_url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else pub_url)
            papers.append({
                "paper_id": f"ss:{ss_id}",
                "title":    item.get("title", ""),
                "authors":  authors,
                "year":     item.get("year"),
                "abstract": (item.get("abstract") or "").strip(),
                "url":      final_url,
                "venue":    item.get("venue", ""),
                "source":   "semantic_scholar",
            })
        return papers
    except Exception as e:
        print(f"[Semantic Scholar error] {e}")
        return []


def _fallback_search(keywords: list[str]) -> list[dict]:
    """
    Keyword-based fallback returning stub results so the app still works
    when both external APIs fail (e.g., during a demo with no internet).
    """
    return [
        {
            "paper_id": f"fallback:{hashlib.md5(k.encode()).hexdigest()[:8]}",
            "title":    f"[Demo] Research overview: {k.title()}",
            "authors":  ["Demo Author"],
            "year":     2024,
            "abstract": f"This is a placeholder result for the keyword '{k}'. "
                        "External search APIs are currently unreachable.",
            "url":      "#",
            "venue":    "Fallback",
            "source":   "fallback",
        }
        for k in keywords[:3]
    ]


def search_literature(keywords: list[str], session_id: str) -> dict:
    """
    Search arXiv + Semantic Scholar. Merge and de-duplicate results.
    Falls back to stubs if both APIs fail.
    """
    arxiv_papers = _search_arxiv(keywords, MAX_PAPERS)
    ss_papers    = _search_semantic_scholar(keywords, MAX_PAPERS)

    seen_titles: set[str] = set()
    merged = []
    for p in arxiv_papers + ss_papers:
        norm_title = re.sub(r"\s+", " ", p["title"].lower().strip())
        if norm_title not in seen_titles and p["title"]:
            seen_titles.add(norm_title)
            merged.append(p)
        if len(merged) >= MAX_PAPERS:
            break

    used_fallback = False
    if not merged:
        merged = _fallback_search(keywords)
        used_fallback = True

    return {
        "papers":       merged,
        "used_fallback": used_fallback,
        "arxiv_count":  len(arxiv_papers),
        "ss_count":     len(ss_papers),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Summarization (batched)
# ═══════════════════════════════════════════════════════════════════════════

def _truncate_abstract(text: str) -> str:
    return text[:ABSTRACT_MAX_CHARS] + ("…" if len(text) > ABSTRACT_MAX_CHARS else "")


def summarize_papers(papers: list[dict], query: str, session_id: str) -> dict[str, str]:
    """
    Summarize a batch of papers in a SINGLE LLM call when possible.
    Falls back to per-paper calls if the batch would exceed ~3000 input chars.
    Returns a dict mapping paper_id → summary string.
    """
    to_summarize = []
    results: dict[str, str] = {}
    for p in papers:
        if p.get("summary"):
            results[p["paper_id"]] = p["summary"]
            continue
        cache_key = db.cache_key_for(f"summary::{GRANITE_MODEL_ID}::{p['paper_id']}::{query[:100]}")
        cached = db.get_cached_response(cache_key)
        if cached:
            results[p["paper_id"]] = cached
        else:
            to_summarize.append(p)

    if not to_summarize:
        return results

    combined_text = " ".join(
        _truncate_abstract(p.get("abstract", "")) for p in to_summarize
    )
    use_batch = len(combined_text) < 3000

    if use_batch:
        summaries = _batch_summarize(to_summarize, query, session_id)
    else:
        summaries = {}
        for p in to_summarize:
            summaries[p["paper_id"]] = _single_summarize(p, query, session_id)

    results.update(summaries)
    return results


def _batch_summarize(papers: list[dict], query: str, session_id: str) -> dict[str, str]:
    """One LLM call for all papers together."""
    entries = []
    for i, p in enumerate(papers, 1):
        abstract = _truncate_abstract(p.get("abstract", "(no abstract)"))
        entries.append(f"Paper {i} [{p['paper_id']}]:\nTitle: {p['title']}\nAbstract: {abstract}")

    prompt = f"""Question: You are a scientific research assistant. Summarize each paper below in 80 words or fewer, highlighting its method, findings, and relevance to the research query. Be {AGENT_INSTRUCTIONS['tone']}.

Research query: "{query}"

{chr(10).join(entries)}

For each paper, write exactly one block in this format (repeat for every paper, in order):
PAPER_ID: <paper_id>
SUMMARY: <your summary>

Answer:"""
    raw = _call_llm(prompt, "summary", session_id,
                    max_new_tokens=TOKEN_CAPS["summary"] * len(papers))

    summaries: dict[str, str] = {}
    blocks = re.split(r"\nPAPER_ID:", "\n" + raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        id_match  = re.match(r"^([\w:\-]+)", block)
        sum_match = re.search(r"SUMMARY:\s*(.+)", block, re.DOTALL)
        if id_match and sum_match:
            pid     = id_match.group(1).strip()
            summary = sum_match.group(1).strip()[:600]
            summaries[pid] = summary
            cache_key = db.cache_key_for(f"summary::{GRANITE_MODEL_ID}::{pid}::{query[:100]}")
            db.set_cached_response(cache_key, summary)

    if not summaries:
        lines = [l for l in raw.split("\n") if l.strip()]
        for i, p in enumerate(papers):
            fallback = lines[i] if i < len(lines) else "Summary not available."
            summaries[p["paper_id"]] = fallback[:300]

    return summaries


def _single_summarize(paper: dict, query: str, session_id: str) -> str:
    """One LLM call for a single paper."""
    abstract = _truncate_abstract(paper.get("abstract", "(no abstract)"))
    prompt = f"""Question: You are a research assistant. Summarize the paper below in 80 words or fewer, highlighting its method, findings, and relevance to the query. Be {AGENT_INSTRUCTIONS['tone']}.

Query: "{query}"
Title: {paper['title']}
Abstract: {abstract}

Answer:"""
    return _call_llm(prompt, "summary", session_id)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Citation Formatting (pure Python — NO LLM)
# ═══════════════════════════════════════════════════════════════════════════

def _author_list(authors: list[str], style: str, max_authors: int = 6) -> str:
    if not authors:
        return "Unknown Author(s)"
    if len(authors) > max_authors:
        authors = authors[:max_authors] + ["et al."]
    if style == "APA":
        formatted = []
        for a in authors:
            parts = a.strip().split()
            if len(parts) >= 2:
                last  = parts[-1]
                inits = " ".join(p[0] + "." for p in parts[:-1])
                formatted.append(f"{last}, {inits}")
            else:
                formatted.append(a)
        if len(formatted) > 1:
            return ", ".join(formatted[:-1]) + ", & " + formatted[-1]
        return formatted[0]
    elif style == "IEEE":
        formatted = []
        for a in authors:
            parts = a.strip().split()
            if len(parts) >= 2:
                inits = ". ".join(p[0] for p in parts[:-1]) + "."
                formatted.append(f"{inits} {parts[-1]}")
            else:
                formatted.append(a)
        return ", ".join(formatted)
    else:
        return " and ".join(authors)


def format_citation(paper: dict, style: str) -> str:
    """Return a formatted citation string. 100% pure Python."""
    title   = paper.get("title", "Untitled")
    year    = paper.get("year") or "n.d."
    url     = paper.get("url", "")
    venue   = paper.get("venue", "")
    authors = paper.get("authors", [])

    if style == "BibTeX":
        pid     = re.sub(r"[^a-zA-Z0-9]", "", paper.get("paper_id", "paper"))
        first   = (authors[0].split()[-1] if authors else "unknown").lower()
        bib_key = f"{first}{year}"
        author_str = _author_list(authors, "BibTeX")
        return (
            f"@article{{{bib_key},\n"
            f"  author  = {{{author_str}}},\n"
            f"  title   = {{{title}}},\n"
            f"  year    = {{{year}}},\n"
            f"  journal = {{{venue or 'arXiv'}}},\n"
            f"  url     = {{{url}}}\n"
            f"}}"
        )

    elif style == "APA":
        author_str = _author_list(authors, "APA")
        venue_str  = f" *{venue}*." if venue else ""
        url_str    = f" {url}" if url else ""
        return f"{author_str} ({year}). {title}.{venue_str}{url_str}"

    elif style == "IEEE":
        author_str = _author_list(authors, "IEEE")
        venue_str  = f" in *{venue}*," if venue else ""
        url_str    = f". [Online]. Available: {url}" if url else ""
        return f"{author_str}, \"{title}\",{venue_str} {year}{url_str}"

    return f"{', '.join(authors[:3])} ({year}). {title}."


def export_references(papers: list[dict], style: str) -> str:
    """Export full reference list in the given citation style."""
    lines = []
    for i, p in enumerate(papers, 1):
        citation = format_citation(p, style)
        if style == "BibTeX":
            lines.append(citation)
        else:
            lines.append(f"[{i}] {citation}")
    sep = "\n\n" if style == "BibTeX" else "\n"
    return sep.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Report Generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(papers: list[dict], query: str, citation_style: str,
                    session_id: str) -> str:
    """
    Generate a structured literature-review report from ALREADY-SAVED summaries.
    Papers without summaries get a placeholder; no re-summarization occurs.

    NOTE: Each section is generated with its own short, focused LLM call
    rather than asking for the entire multi-section report in one shot.
    The model available in this project (ibm/granite-8b-code-instruct, a
    smaller, deprecated instruct model whose decoding parameters are not
    fully honoured by the API) proved unreliable when asked to write
    several structured sections in a single long completion — it would
    sometimes emit only headings, hallucinated meta-commentary, or an
    empty response. Splitting into one simpler call per section costs
    more tokens overall, but is far more reliable in practice.
    """
    if not papers:
        return "No saved references found. Please save some papers first."

    paper_context = []
    for i, p in enumerate(papers[:10], 1):
        summary = p.get("summary") or "(summary not available)"
        citation = format_citation(p, citation_style)
        paper_context.append(
            f"[{i}] {p['title']} ({p.get('year', 'n.d.')})\n"
            f"    Summary: {summary[:300]}\n"
            f"    Citation ({citation_style}): {citation}"
        )
    context_str = "\n\n".join(paper_context)

    # Section-specific token budgets. Related Work needs the most room since
    # it discusses multiple papers; the others need enough to complete
    # 3-5 full sentences without being cut off mid-thought.
    section_token_caps = {
        "Introduction":   220,
        "Related Work":   320,
        "Summary Table":  260,
        "Conclusion":     300,
    }
    default_section_tokens = 200

    section_instructions = {
        "Introduction": (
            f'Write the Introduction paragraph of a literature review about "{query}". '
            f"Be {AGENT_INSTRUCTIONS['tone']}. Write 3-5 sentences of real prose only "
            f"(no headings, no instructions, no bullet points)."
        ),
        "Summary Table": (
            f"Create a markdown table with EXACTLY 3 columns: Paper, Year, Key Contribution. "
            f"One row per paper below. In the 'Paper' column, write ONLY the short title "
            f"(no citation, no URL, no author names). In 'Key Contribution', write ONE "
            f"short sentence (max 15 words). Do not add any extra columns or information "
            f"beyond these three.\n\nPapers:\n{context_str}"
        ),
        "Conclusion": (
            f'Write a short Conclusion paragraph (exactly 3-4 sentences, no more) for '
            f'a literature review about "{query}". Do NOT describe each paper individually '
            f"one by one — that has already been covered in the Related Work section. "
            f"Instead, briefly synthesize the overall theme or trend across the papers "
            f"below, then note one direction for future work. Refer to papers by "
            f"citation number only (e.g. [1], [2]), never by author name. "
            f"Be {AGENT_INSTRUCTIONS['tone']}.\n\nPapers:\n{context_str}"
        ),
    }

    parts = []
    for section_name in AGENT_INSTRUCTIONS["report_structure"]:
        if section_name == "Related Work":
            # Built deterministically — see earlier note.
            rw_lines = []
            for i, p in enumerate(papers[:10], 1):
                summary = p.get("summary") or "(summary not available)"
                rw_lines.append(f"**[{i}] {p['title']}** ({p.get('year', 'n.d.')}): {summary}")
            section_text = "\n\n".join(rw_lines)

        elif section_name == "Summary Table":
            # Built deterministically from paper data — no LLM call. This
            # model (a code-instruct model) proved unreliable at this task,
            # repeatedly producing hallucinated code/JS snippets instead of
            # a table when asked to generate one via prompt.
            table_lines = ["| Paper | Year | Key Contribution |", "|---|---|---|"]
            for i, p in enumerate(papers[:10], 1):
                summary = p.get("summary") or ""
                # Use the first sentence of the summary as the "key contribution"
                first_sentence = summary.split(". ")[0].strip()
                if first_sentence and not first_sentence.endswith("."):
                    first_sentence += "."
                first_sentence = first_sentence[:120]
                title_short = p['title'][:60] + ("…" if len(p['title']) > 60 else "")
                table_lines.append(f"| [{i}] {title_short} | {p.get('year', 'n.d.')} | {first_sentence} |")
            section_text = "\n".join(table_lines)

        else:
            body_instructions = section_instructions.get(
                section_name,
                f'Write the "{section_name}" section of a literature review about "{query}" '
                f"based on the papers below.\n\nPapers:\n{context_str}"
            )
            prompt = f"Question: {body_instructions}\n\nAnswer:"
            section_text = _call_llm(
                prompt, "report", f"{session_id}:{section_name}",
                max_new_tokens=section_token_caps.get(section_name, default_section_tokens)
            )
        parts.append(f"## {section_name}\n\n{section_text}")

    return "\n\n".join(parts)
# ═══════════════════════════════════════════════════════════════════════════
# 6. Hypothesis Suggestion
# ═══════════════════════════════════════════════════════════════════════════

def suggest_hypotheses(papers: list[dict], session_id: str) -> str:
    """
    Analyze patterns across saved summaries and suggest research hypotheses.
    Single LLM call; uses summaries only (not raw abstracts).
    """
    if not papers:
        return "No saved references available for hypothesis generation."

    summaries_text = "\n".join(
        f"- [{p['title'][:60]}]: {(p.get('summary') or p.get('abstract', ''))[:200]}"
        for p in papers[:8]
    )
    n = AGENT_INSTRUCTIONS["hypothesis_count"]

    prompt = f"""Question: You are a research hypothesis generator. Based on the paper summaries below, identify research gaps and propose exactly {n} novel research hypotheses. Be {AGENT_INSTRUCTIONS['tone']}. For each hypothesis, give a clear statement and a one-sentence justification.

Summaries:
{summaries_text}

Write your answer in exactly this format:
H1: <hypothesis>
Justification: <one sentence>

H2: <hypothesis>
Justification: <one sentence>

(continue for all {n} hypotheses)

Answer:"""
    return _call_llm(prompt, "hypothesis", session_id,
                     max_new_tokens=TOKEN_CAPS["hypothesis"])


# ═══════════════════════════════════════════════════════════════════════════
# 7. Draft Section Writer
# ═══════════════════════════════════════════════════════════════════════════

def write_draft_section(section_name: str, papers: list[dict],
                        query: str, session_id: str) -> str:
    """Draft a specific paper section using saved references."""
    if not papers:
        return "No saved references to base the draft on."

    n = min(len(papers), 6)
    ref_lines = []
    for i, p in enumerate(papers[:6], 1):
        summary = (p.get('summary') or '')[:150]
        ref_lines.append(f"[{i}] {p['title']}: {summary}")
    citations = "\n".join(ref_lines)

    prompt = f"""Question: Write the "{section_name}" section of a research paper about "{query}", using only the {n} references below. Cite them as [1] through [{n}] where relevant. Be {AGENT_INSTRUCTIONS['tone']}. Write about 120 words of real prose, finishing your last sentence completely.

References:
{citations}

Answer:"""
    return _call_llm(prompt, "draft", session_id, max_new_tokens=300)
# ═══════════════════════════════════════════════════════════════════════════
# 8. Data Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_paper_data(text: str, session_id: str) -> dict:
    """
    Extract structured fields from a pasted abstract/paper excerpt.
    Single LLM call; result is cached by hash of input text.
    """
    truncated = text[:ABSTRACT_MAX_CHARS * 2]
    prompt = f"""Question: You are a scientific information extractor. Extract structured fields from the text below. Respond ONLY with a single valid JSON object — no markdown, no explanation, no extra text before or after.

Text:
{truncated}

Return exactly this JSON shape:
{{"objective": "...", "method": "...", "dataset": "...", "results": "...", "limitations": "..."}}

Answer:"""
    raw = _call_llm(prompt, "extraction", session_id,
                    max_new_tokens=TOKEN_CAPS["extraction"])
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"raw_response": raw, "error": "Could not parse structured output."}


# ═══════════════════════════════════════════════════════════════════════════
# Utility: token usage summary
# ═══════════════════════════════════════════════════════════════════════════

def get_usage_summary(session_id: str) -> dict:
    session = db.get_session_token_total(session_id)
    daily   = db.get_daily_token_total()
    return {
        "session_tokens_in":  session["tokens_in"],
        "session_tokens_out": session["tokens_out"],
        "session_calls":      session["calls"],
        "session_total":      session["tokens_in"] + session["tokens_out"],
        "daily_total":        daily,
        "daily_limit":        MAX_TOKENS_PER_DAY,
        "daily_pct":          round(min(daily / max(MAX_TOKENS_PER_DAY, 1) * 100, 100), 1),
        "warning":            daily >= MAX_TOKENS_PER_DAY * 0.8,
    }
