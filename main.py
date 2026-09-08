"""Main CLI entrypoint and evaluation runner for CostOptimizationTelemetryAgent."""

import argparse
import asyncio
import json
import os
import sys
from uuid import uuid4

from telemetry_agent.agent import root_agent
from telemetry_agent.client import default_client
from telemetry_agent.mcp_client import ClickHouseMCPClient
from telemetry_agent.schemas import SessionInitPayload, RerunPayload, TerminalPayload
from telemetry_agent.tools import (
    get_telemetry_pipeline_summary,
    query_model_empirical_insights,
)

SAMPLE_TELEMETRY_WORKFLOWS = [
    {
        "title": "Scene 1: Drone Skyline Shot (High Quality - 2 Reruns)",
        "prompt": "Cinematic aerial drone shot sweeping over a modern metropolis skyline at golden hour sunset",
        "model": "Runway Gen-3 Alpha",
        "base_cost": 0.0500,
        "reruns": [
            {"incremental_cost": 0.0500, "reason": "Lighting too dark on skyscrapers"},
            {"incremental_cost": 0.0500, "reason": "Adjust camera lens flare intensity"},
        ],
        "accepted": True,
        "terminal_reason": "Approved and downloaded final render",
    },
    {
        "title": "Scene 2: Cyberpunk Character Dialogue (High Motion - 1 Rerun)",
        "prompt": "Medium close-up of a cyberpunk detective in a neon-lit alleyway arguing with a robotic informant",
        "model": "Luma Ray 2",
        "base_cost": 0.0350,
        "reruns": [
            {"incremental_cost": 0.0350, "reason": "Robotic eye glow color incorrect"},
        ],
        "accepted": True,
        "terminal_reason": "Approved for sequence assembly",
    },
    {
        "title": "Scene 3: Complex Water Fluid Dynamics (Failed & Abandoned)",
        "prompt": "Hyper-realistic slow motion macro shot of stormy ocean water splashing against rocky cliffs",
        "model": "Kling 1.5 Pro",
        "base_cost": 0.0280,
        "reruns": [
            {"incremental_cost": 0.0280, "reason": "Water physics unnatural, foam missing"},
            {"incremental_cost": 0.0280, "reason": "Distortion artifacts on rock surface"},
        ],
        "accepted": False,
        "terminal_reason": "User canceled generation session after failed retries",
    },
]


async def run_adk_agent_query(prompt: str, user_id: str = "filmmaker", session_id: str = "session_001"):
    """Runs a prompt through the Google ADK InMemoryRunner."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    print(f"\n{'='*70}")
    print(f"🤖 ADK TELEMETRY AGENT QUERY: {prompt}")
    print(f"{'='*70}\n")

    runner = InMemoryRunner(agent=root_agent)
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_content,
    ):
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text, end="", flush=True)
                elif getattr(part, "function_call", None):
                    print(f"\n[Tool Call] -> {part.function_call.name}({part.function_call.args})\n")
                elif getattr(part, "function_response", None):
                    print(f"\n[Tool Response] -> {part.function_response.name} completed.\n")
    print("\n")


async def run_simulation_demo():
    """Runs an automated showcase simulating real filmmaking sessions with telemetry capture."""
    print("\n" + "=" * 75)
    print("🎬  AGENTIC CINEMA - COST OPTIMIZATION TELEMETRY AGENT DEMO")
    print("    Target Track: ClickHouse MCP (mcp-clickhouse)")
    print("=" * 75 + "\n")

    print("[Step 1] Ensuring ClickHouse 'generation_telemetry' table is initialized...")
    await default_client.mcp_client.ensure_table_exists()
    print("✓ ClickHouse table schema verified.\n")

    print("[Step 2] Simulating Video Generation Sessions & Telemetry Ingestion...\n")

    for idx, workflow in enumerate(SAMPLE_TELEMETRY_WORKFLOWS, 1):
        print(f"--- Workflow {idx}: {workflow['title']} ---")
        session_id = uuid4()

        # 1. Session Initialization (FR-1)
        print(f"  [FR-1 Initializing Session {session_id}]")
        print(f"    Prompt: '{workflow['prompt'][:60]}...'")
        print(f"    Model: {workflow['model']} | Quoted Base API Cost: ${workflow['base_cost']:.4f}")
        await default_client.async_start_session(
            prompt_text=workflow["prompt"],
            suggested_model=workflow["model"],
            base_api_cost=workflow["base_cost"],
            session_id=session_id,
        )

        # 2. Iterations / Reruns (FR-2)
        for r_idx, rerun in enumerate(workflow["reruns"], 1):
            print(f"  [FR-2 Rerun #{r_idx}] Reason: '{rerun['reason']}' (+ ${rerun['incremental_cost']:.4f})")
            await default_client.async_record_rerun(
                session_id=session_id,
                incremental_cost=rerun["incremental_cost"],
                reason=rerun["reason"],
            )

        # 3. Terminal Outcome (FR-3)
        outcome_str = "ACCEPTED (user_accepted=1)" if workflow["accepted"] else "ABANDONED (user_accepted=0)"
        print(f"  [FR-3 Terminal Outcome] {outcome_str} | Reason: {workflow['terminal_reason']}")
        record = await default_client.async_complete_session(
            session_id=session_id,
            accepted=workflow["accepted"],
            reason=workflow["terminal_reason"],
        )

        if record:
            print(f"  [FR-4 Persisted via MCP] Realized Session Cost: ${record.total_session_cost:.4f} | Total Reruns: {record.total_rerun_count}")
        print()

    # Step 3: Closed-Loop Feedback Query
    print("[Step 3] Querying ClickHouse Empirical Insights for Cost Optimizer Feedback Loop...\n")
    stats_json = query_model_empirical_insights()
    print(stats_json)

    print("\n" + "=" * 75)
    print("✓ DEMO COMPLETE: Telemetry successfully captured & closed-loop data available.")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Cost Optimization Telemetry Agent - ClickHouse MCP")
    parser.add_argument("--demo", action="store_true", help="Run automated showcase simulating filmmaking telemetry")
    parser.add_argument("--stats", action="store_true", help="Query and display historical ClickHouse empirical stats")
    parser.add_argument("--summary", action="store_true", help="Display global pipeline telemetry summary")
    parser.add_argument("--query", type=str, help="Query the ADK Telemetry Agent with natural language")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI Telemetry REST server")
    parser.add_argument("--port", type=int, default=8001, help="Port for FastAPI server")

    args = parser.parse_args()

    if args.serve:
        import uvicorn
        from fastapi import FastAPI
        app = FastAPI(title="Cost Optimization Telemetry Agent Bridge")
        from telemetry_agent.web_bridge import telemetry_router
        app.include_router(telemetry_router)
        print(f"Starting Telemetry REST Server on http://0.0.0.0:{args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return

    if args.stats:
        print(query_model_empirical_insights())
        return

    if args.summary:
        print(get_telemetry_pipeline_summary())
        return

    if args.query:
        asyncio.run(run_adk_agent_query(args.query))
        return

    # Default to automated demo
    asyncio.run(run_simulation_demo())


if __name__ == "__main__":
    main()
