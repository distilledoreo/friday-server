#!/usr/bin/env bash
# Smoke-test each backend independently.
cd "$(dirname "$0")"; set -a; . ./.env; set +a
echo "== llama.cpp :8080"
curl -s -m 120 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with just: pong"}],"max_tokens":200}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
echo "== SearXNG :8888"
curl -s -m 30 'http://127.0.0.1:8888/search?q=latest+nvidia+news&format=json' \
  | python3 -c 'import sys,json;r=json.load(sys.stdin)["results"];print(len(r),"results");[print(" -",x["title"][:80]) for x in r[:3]]'
echo "== Crawl4AI :11235"
if ! curl -sf -m 5 http://127.0.0.1:11235/health >/dev/null; then echo "(not running — start with: docker compose --profile crawl up -d)"; exit 0; fi
curl -s -m 90 http://127.0.0.1:11235/crawl -H "Authorization: Bearer $CRAWL4AI_API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"urls":["https://example.com"]}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);r=d["results"][0];m=r.get("markdown");m=m.get("raw_markdown",m) if isinstance(m,dict) else m;print("success:",r["success"]);print((m or "")[:200])'
