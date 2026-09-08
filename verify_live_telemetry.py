#!/usr/bin/env python3
"""Script to verify that the live Cost Optimizer Agent is using ClickHouse telemetry."""

import json
import urllib.request

URL = "https://agent-film-cost-optimizer-709949980336.us-central1.run.app/api/analyze"

payload = {
    "shot_description": "Cinematic slow pan across a snow-covered mountain ridge",
    "duration_seconds": 5.0,
    "rerun_multiplier": 2.2,
}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

print("🚀 Querying live Cost Optimizer Agent on Cloud Run...\n")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))

print("=" * 70)
print("1. EXECUTION TIMELINE EVENTS (ClickHouse MCP Closed-Loop)")
print("=" * 70)
for ev in data.get("execution_timeline", []):
    svc = ev.get("service")
    msg = ev.get("message")
    icon = "📊" if svc == "ClickHouse MCP" else ("⚡" if "Gemini" in svc else "🎬")
    print(f" {icon} [{svc}] {msg}")

print("\n" + "=" * 70)
print("2. VIABLE MODEL EVALUATION & GROUND-TRUTH EMPIRICAL CALIBRATION")
print("=" * 70)
viable = data.get("evaluation", {}).get("viable_models", [])
for m in viable:
    name = m.get("model_name")
    cost = m.get("total_estimated_cost_usd")
    is_emp = m.get("is_empirical")
    emp_mult = m.get("empirical_multiplier")
    
    if is_emp:
        status_tag = f"✅ EMPIRICAL (ClickHouse {emp_mult}x multiplier applied)"
    else:
        status_tag = "⚪ BASELINE (Default slider multiplier)"
    
    print(f" • {name:<20} | Total Cost: ${cost:.4f} | {status_tag}")

rec = data.get("evaluation", {}).get("recommended_model", {})
print("\n" + "=" * 70)
print(f"3. TOP RECOMMENDED MODEL: {rec.get('model_name')} (${rec.get('total_estimated_cost_usd', 0):.4f})")
print("=" * 70)
