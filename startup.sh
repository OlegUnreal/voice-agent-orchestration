#!/bin/sh
set -eu
cd /workspace
node scripts/preview.mjs stop || true

if ! curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8090/health; then
  if [ ! -x python/.venv/bin/python ]; then
    python3 -m venv python/.venv
    python/.venv/bin/pip install -e "/workspace/python[dev]"
  fi
  PYTHONPATH=/workspace/python python/.venv/bin/python -m uvicorn helix.gateway.app:app --host 127.0.0.1 --port 8090 >>/tmp/helix-engine.log 2>&1 &
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -sf -o /dev/null --max-time 1 http://127.0.0.1:8090/health; then
      break
    fi
    sleep 0.4
  done
fi

if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8080/; then
  exit 0
fi
HELIX_ENGINE_URL=http://127.0.0.1:8090 npm run dev >>/tmp/app-startup.log 2>&1 &
