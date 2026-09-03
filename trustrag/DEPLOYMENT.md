# TrustRAG Deployment Notes

Recommended architecture:

- GitHub: source repository
- Vercel: `frontend/`
- Render: FastAPI backend
- Render persistent disk: SQLite accounts/chats + Hugging Face model cache

## Render backend

Root directory: repository root

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Health check path:

```text
/api/health
```

Environment variables:

```text
PYTHON_VERSION=3.12.7
TRUSTRAG_ALLOWED_ORIGINS=https://YOUR-VERCEL-DOMAIN.vercel.app
TRUSTRAG_DB_PATH=/var/data/users.db
HF_HOME=/var/data/huggingface
```

For persistent accounts/chat history, attach a Render persistent disk at `/var/data`.
The local NLI model is large, so use a backend instance with sufficient RAM rather
than the 512 MB free instance.

## Vercel frontend

Import the same GitHub repository and set the project Root Directory to:

```text
frontend
```

Add:

```text
VITE_API_URL=https://YOUR-RENDER-SERVICE.onrender.com
```

Then redeploy.

After Vercel gives you the final domain, set that exact URL in Render's
`TRUSTRAG_ALLOWED_ORIGINS` variable and redeploy the backend.
