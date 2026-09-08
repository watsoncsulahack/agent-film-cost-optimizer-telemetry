#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PORT=${PORT:-8000}

echo "=========================================================================="
echo "🎬  Agentic Cinema: Unified Cost Optimization & Telemetry Suite"
echo "    - Agent 1: Cost Optimization Agent (ai-film / Parallel API Track)"
echo "    - Agent 2: Cost Optimization Telemetry Agent (ai-films-telemetry / ClickHouse MCP)"
echo "=========================================================================="

export PYTHONPATH="$DIR:/home/allan/ai-film:$PYTHONPATH"

echo "Launching web server on http://localhost:$PORT ..."
exec uvicorn web_app:app --host 0.0.0.0 --port "$PORT"
