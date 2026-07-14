"""
app.py — Flask application for the AI Research Agent
"""
import os
import uuid
import json

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, Response
)
from dotenv import load_dotenv

import database as db
import agent

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["DEBUG"] = os.getenv("FLASK_DEBUG", "False").lower() == "true"

# ── Initialise database on startup ───────────────────────────────────────────
db.init_db()


def get_session_id() -> str:
    """Return (or create) a stable session ID for this browser session."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


# ════════════════════════════════════════════════════════════════════════════
# MAIN PAGES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/references")
def references_page():
    papers = db.get_all_papers()
    return render_template("references.html", papers=papers)


@app.route("/reports")
def reports_page():
    reports = db.get_all_reports()
    papers  = db.get_all_papers()
    return render_template("reports.html", reports=reports, papers=papers)


@app.route("/extract")
def extract_page():
    return render_template("extract.html")


# ════════════════════════════════════════════════════════════════════════════
# API — CHAT / QUERY UNDERSTANDING
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/query", methods=["POST"])
def api_query():
    """
    Parse a research query → keywords + topic + sub-questions.
    Then search and return papers.
    """
    sid  = get_session_id()
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query is required."}), 400

    # 1. Parse query
    parsed = agent.parse_research_query(query, sid)
    keywords = parsed.get("keywords", [])

    # 2. Search literature
    search_result = agent.search_literature(keywords, sid)
    papers        = search_result["papers"]

    # 3. Summarize papers (batched / cached)
    summaries = agent.summarize_papers(papers, query, sid)
    for p in papers:
        p["summary"] = summaries.get(p["paper_id"], "")

    # 4. Persist search history
    db.save_search(query, keywords, [p["paper_id"] for p in papers])

    # 5. Usage stats
    usage = agent.get_usage_summary(sid)

    return jsonify({
        "parsed_query":   parsed,
        "papers":         papers,
        "used_fallback":  search_result["used_fallback"],
        "usage":          usage,
    })


# ════════════════════════════════════════════════════════════════════════════
# API — PAPERS / REFERENCES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/papers", methods=["GET"])
def api_list_papers():
    q = request.args.get("q", "").strip()
    papers = db.search_papers(q) if q else db.get_all_papers()
    return jsonify({"papers": papers})


@app.route("/api/papers/save", methods=["POST"])
def api_save_paper():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No data provided."}), 400

    paper_id = data.get("paper_id", "").strip()
    if not paper_id:
        return jsonify({"error": "paper_id required."}), 400

    db.save_paper(data)
    return jsonify({"ok": True, "paper_id": paper_id})


@app.route("/api/papers/<path:paper_id>", methods=["DELETE"])
def api_delete_paper(paper_id):
    db.delete_paper(paper_id)
    return jsonify({"ok": True})


@app.route("/api/papers/<path:paper_id>/tags", methods=["PUT"])
def api_update_tags(paper_id):
    data = request.get_json(force=True)
    tags = data.get("tags", [])
    db.update_paper_tags(paper_id, tags)
    return jsonify({"ok": True})


@app.route("/api/papers/export", methods=["GET"])
def api_export_references():
    """Export all saved papers in BibTeX, APA, or IEEE format."""
    style  = request.args.get("style", "APA").upper()
    if style not in ("BIBTEX", "APA", "IEEE"):
        style = "APA"
    if style == "BIBTEX":
        style = "BibTeX"

    papers   = db.get_all_papers()
    exported = agent.export_references(papers, style)

    ext = "bib" if style == "BibTeX" else "txt"
    return Response(
        exported,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename=references_{style.lower()}.{ext}"},
    )


# ════════════════════════════════════════════════════════════════════════════
# API — REPORTS
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/reports/generate", methods=["POST"])
def api_generate_report():
    sid    = get_session_id()
    data   = request.get_json(force=True)
    query  = (data.get("query") or "Literature Review").strip()
    style  = data.get("citation_style", "APA")
    papers = db.get_all_papers()

    if not papers:
        return jsonify({"error": "No saved references found."}), 400

    content = agent.generate_report(papers, query, style, sid)

    # Save to DB
    report_id = db.save_report(
        title=f"Literature Review — {query[:60]}",
        query=query,
        citation_style=style,
        content=content,
    )
    usage = agent.get_usage_summary(sid)
    return jsonify({"report_id": report_id, "content": content, "usage": usage})


@app.route("/api/reports/<int:report_id>", methods=["GET"])
def api_get_report(report_id: int):
    report = db.get_report(report_id)
    if not report:
        return jsonify({"error": "Report not found."}), 404
    return jsonify(report)


# ════════════════════════════════════════════════════════════════════════════
# API — HYPOTHESES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/hypotheses", methods=["POST"])
def api_hypotheses():
    sid    = get_session_id()
    papers = db.get_all_papers()
    if not papers:
        return jsonify({"error": "No saved references found."}), 400

    result = agent.suggest_hypotheses(papers, sid)
    usage  = agent.get_usage_summary(sid)
    return jsonify({"hypotheses": result, "usage": usage})


# ════════════════════════════════════════════════════════════════════════════
# API — DRAFT SECTION WRITER
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/draft", methods=["POST"])
def api_draft():
    sid    = get_session_id()
    data   = request.get_json(force=True)
    section = (data.get("section") or "Introduction").strip()
    query   = (data.get("query") or "").strip()
    papers  = db.get_all_papers()

    if not papers:
        return jsonify({"error": "No saved references found."}), 400

    draft = agent.write_draft_section(section, papers, query, sid)
    usage = agent.get_usage_summary(sid)
    return jsonify({"draft": draft, "usage": usage})


# ════════════════════════════════════════════════════════════════════════════
# API — DATA EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/extract", methods=["POST"])
def api_extract():
    sid  = get_session_id()
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) < 50:
        return jsonify({"error": "Text too short for extraction."}), 400

    result = agent.extract_paper_data(text, sid)
    usage  = agent.get_usage_summary(sid)
    return jsonify({"extracted": result, "usage": usage})


# ════════════════════════════════════════════════════════════════════════════
# API — USAGE
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/usage", methods=["GET"])
def api_usage():
    return jsonify(agent.get_usage_summary(get_session_id()))


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=app.config["DEBUG"])
