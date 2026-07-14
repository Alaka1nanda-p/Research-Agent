# 🔬 AI Research Agent

A full-stack AI-powered research assistant built with **Python Flask** and **IBM Watsonx.ai** It automates academic research workflows, including literature search, AI-powered summarization, citation management, and report generation.

## 🛠️ Built With

- Python 3.11
- Flask
- IBM watsonx.ai (Granite Models)
- IBM Cloud
- IBM Bob (development environment)
- SQLite
- arXiv API
- Semantic Scholar API

---

---

## ✨ Features

| Feature | Description |
|---|---|
| **Chat & Query** | Natural-language research questions with intelligent keyword extraction |
| **Literature Search** | Searches arXiv + Semantic Scholar APIs (keyless, free) |
| **AI Summarization** | Batch-summarizes papers using Granite with token-conserving prompting |
| **References Library** | SQLite-backed library with tag, search, delete, and export |
| **Citation Export** | BibTeX, APA, IEEE — pure Python, no LLM needed |
| **Report Generator** | Structured literature-review (Intro → Related Work → Table → Conclusion) |
| **Hypothesis Suggestion** | 2–3 novel hypotheses from saved summaries in a single LLM call |
| **Draft Section Writer** | AI-drafted Introduction, Related Work, Methodology, etc. |
| **Data Extraction** | Paste any abstract → extract objective/method/dataset/results/limitations |
| **Token Tracker** | Real-time usage bar, daily cap, LLM response cache in SQLite |


---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <your-repo>
cd ResearchAgent
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
Copy .env.example to .env
```

Edit `.env` with your credentials:

```ini
IBM_API_KEY=your_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://au-syd.ml.cloud.ibm.com
FLASK_SECRET_KEY=any-random-string
```

**Where to get IBM credentials:**
1. Sign up at [cloud.ibm.com](https://cloud.ibm.com) (Lite plan is free)
2. Create a **Watsonx.ai** service instance
3. Go to **IAM → API keys** and generate an API key
4. Open your Watsonx project → **Settings → General** to find the Project ID

### 3. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000)

---

## 📁 Project Structure

```
ResearchAgent/
├── app.py              # Flask routes & API endpoints
├── agent.py            # Core AI logic: search, summarize, generate
├── database.py         # SQLite models: papers, cache, usage, reports
├── requirements.txt
├── .gitignore
├── .env.example
├── .env                # (create from .env.example — never commit this)
├── research_agent.db   # Auto-created on first run
└── templates/
    ├── base.html       # Sidebar layout, dark theme, usage widget
    ├── index.html      # Chat interface + search results
    ├── references.html # Library with tag/search/export
    ├── reports.html    # Report generator, hypotheses, draft writer
    └── extract.html    # Data extraction tool
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `IBM_API_KEY` | — | IBM Cloud API Key |
| `WATSONX_PROJECT_ID` | — | Watsonx.ai Project ID |
| `WATSONX_URL` | `https://au-syd.ml.cloud.ibm.com` | Regional endpoint |
| `GRANITE_MODEL_ID` | `ibm/granite-8b-code-instruct` | Model to use |
| `GRANITE_MAX_NEW_TOKENS_DEFAULT` | `300` | Global token cap for LLM calls |
| `MAX_TOKENS_PER_DAY` | `50000` | Daily usage warning threshold |
| `MAX_TOKENS_PER_SESSION` | `10000` | Session-level cap |
| `MAX_PAPERS_PER_SEARCH` | `5` | Papers returned per search |
| `ABSTRACT_MAX_CHARS` | `1000` | Truncation limit for abstracts sent to LLM |
| `DATABASE_PATH` | `research_agent.db` | SQLite file path |
| `FLASK_SECRET_KEY` | — | Session encryption key |
| `FLASK_DEBUG` | `False` | Enable Flask debug mode |

---

## 🧠 Token Conservation Strategy

The app is designed for IBM Cloud Lite plan limits:

- **SQLite cache**: Every LLM response is cached by SHA-256 of the prompt. The same input is **never sent twice**.
- **Batch summarization**: Multiple abstracts are batched into one LLM call (when combined length < 3000 chars).
- **Abstract truncation**: Long abstracts are truncated to `ABSTRACT_MAX_CHARS` before sending.
- **Per-call token caps**: Each call type has its own conservative `max_new_tokens` limit.
- **Pure Python fallback**: Citation formatting, keyword extraction, sorting — never touches the LLM.
- **Usage indicator**: Real-time bar in the sidebar shows daily token usage vs. limit.

---

## 🛠 Customizing Agent Behavior

In [`agent.py`](agent.py), edit the `AGENT_INSTRUCTIONS` dict at the top:

```python
AGENT_INSTRUCTIONS = {
    "tone": "academic and concise",          # Change tone of all LLM outputs
    "default_citation_style": "APA",         # Default for reports
    "summarization_depth": "brief",          # "brief" or "detailed"
    "report_structure": [                    # Sections in generated reports
        "Introduction", "Related Work",
        "Summary Table", "Conclusion"
    ],
    "language": "English",
    "hypothesis_count": 3,                   # How many hypotheses to suggest
}
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/query` | Parse query + search + summarize |
| `GET` | `/api/papers` | List saved papers (optional `?q=filter`) |
| `POST` | `/api/papers/save` | Save a paper to references |
| `DELETE` | `/api/papers/<id>` | Delete a saved paper |
| `PUT` | `/api/papers/<id>/tags` | Update tags |
| `GET` | `/api/papers/export` | Export refs (?style= APA | IEEE | BibTeX )
| `POST` | `/api/reports/generate` | Generate literature review report |
| `GET` | `/api/reports/<id>` | Fetch a saved report |
| `POST` | `/api/hypotheses` | Suggest research hypotheses |
| `POST` | `/api/draft` | Draft a paper section |
| `POST` | `/api/extract` | Extract structured data from abstract |
| `GET` | `/api/usage` | Token usage summary |

---

## 🐳 Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
docker build -t research-agent .
docker run -p 5000:5000 --env-file .env research-agent
```

---

## ☁️ Deploy to IBM Code Engine

```bash
# Build and push image
ibmcloud ce application create \
  --name research-agent \
  --image icr.io/<namespace>/research-agent \
  --port 5000 \
  --env-from-secret research-agent-secrets
```

---

## 📝 Notes

- The app works even without Watsonx credentials — search, citation export, filtering, and the references library all work offline / without LLM calls.
- If both arXiv and Semantic Scholar are unreachable (e.g., no internet during a demo), the app falls back to stub placeholder results.
- The SQLite database is created automatically on first run.

---


