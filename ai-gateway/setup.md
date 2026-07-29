# AI Gateway + RAG Pipeline

A learning project implementing:
- Multi-provider LLM gateway (OpenAI, Anthropic, AWS Bedrock)
- Retry + fallback logic
- Rate limiting
- RAG pipeline (Phase 2)

## Setup

```bash
# 1. Create a virtual environment (like node_modules but for Python)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies (like npm install)
pip install -r requirements.txt
pip install pydantic-settings    # settings management

# 3. Copy and fill in your API keys
cp .env.example .env

# 4. Run the server (like npm run dev)
uvicorn main:app --reload
```

## Test it
Open http://localhost:8000/docs — FastAPI gives you a Swagger UI for free.

Or use curl:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "provider": "openai"
  }'
```

## Project structure
```
ai-gateway/
├── main.py              # FastAPI app (like index.ts)
├── config/
│   └── settings.py      # Env config (like a .env loader with types)
├── gateway/
│   ├── models.py         # Pydantic models (like TypeScript interfaces)
│   ├── providers.py      # OpenAI + Anthropic clients
│   ├── router.py         # Routing + retry + fallback
│   └── rate_limiter.py   # Sliding window rate limiter
├── rag/                  # Phase 2 — RAG pipeline (coming soon)
└── tests/
    └── test_gateway.py   # pytest tests (like Jest)
```

## TypeScript → Python cheatsheet
| TypeScript | Python |
|---|---|
| `interface Foo {}` | `class Foo(BaseModel):` |
| `npm install` | `pip install` |
| `package.json` | `requirements.txt` |
| `node_modules/` | `venv/` |
| `ts-node index.ts` | `uvicorn main:app` |
| `jest` | `pytest` |
| `async/await` | `async/await` (identical!) |
| `console.log()` | `print()` |
