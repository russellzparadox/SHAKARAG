# SHAKARAG — Docker deployment

## Files
- `Dockerfile.webapp` — single image: Django webapp + RAG core + CLI scripts
- `docker-compose.webapp.yml` — service definition

## Run on the server

```bash
cd /path/to/rag
docker compose -f docker-compose.webapp.yml up -d --build
```

Port defaults to 8111; override with `SHAKARAG_PORT=9000` if taken:

```bash
SHAKARAG_PORT=8112 docker compose -f docker-compose.webapp.yml up -d
```

## What's inside the image
- `python:3.12-slim-bookworm` (pinned because Docker Hub is unreachable from some hosts;
  it must exist in the server's local cache or registry mirror)
- requirements.txt + `pymssql` (SQL Server) + `onnxruntime` (local embedding fallback)
- repo code at `/app`, workdir `/app/webapp`
- entrypoint runs `manage.py migrate` then `runserver 0.0.0.0:8111`

> Dev server is used intentionally (matches current setup). For production put nginx in front.

## Volumes
| Volume | Container path | Purpose |
|---|---|---|
| `shakarag_web_data` | /app/webapp | sqlite db: users, profiles, chat history |
| `shakarag_chroma` | /data/chroma | default chroma dir when CHROMA_DIR unset |
| bind mount `./.chroma_Shaka` | /app/.chroma_Shaka | the host-built schema index referenced by .env |
| `shakarag_logs` | /app/webapp/logs | webapp.log / errors.log |

The sqlite DB ships inside the image on first run (copied from repo), so users/profiles/chats
carry over from your dev environment. Afterwards it lives only in the volume — **back up
`shakarag_web_data`** or you lose accounts and chat memory.

## Networking (important)
Inside a container, `localhost` is the container itself. The compose file adds:
```
extra_hosts: ["host.docker.internal:host-gateway"]
```
and overrides `LLM_BASE_URL=http://host.docker.internal:20128/v1`.

BUT: LLM and database connection details are stored in **DatabaseProfile / LLMProfile rows**
(sqlite), which take precedence over env vars. After first boot, rewrite any profile rows that
point at localhost:

```bash
docker exec shakarag_web python - <<'EOF'
import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
django.setup()
from chat.models import LLMProfile, DatabaseProfile
for p in LLMProfile.objects.all():
    p.base_url = p.base_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
    p.save()
for d in DatabaseProfile.objects.all():
    if d.host in ("localhost", "127.0.0.1"):
        d.host = "host.docker.internal"
        d.save()
EOF
```

(Already applied once for this deployment; needed again only after re-copying a fresh sqlite.)

LAN databases (e.g. the SQL Server warehouse at 192.168.2.11) need no changes — routable IPs
work as-is.

## Health & logs
```bash
docker ps --filter name=shakarag            # status shows (healthy)
docker logs -f shakarag_web                 # request log
curl http://localhost:8112/accounts/login/  # quick check
```

## Rebuild after code changes
```bash
docker compose -f docker-compose.webapp.yml build
docker compose -f docker-compose.webapp.yml up -d --force-recreate
```
Code is baked into the image; volumes persist data across rebuilds.
