#!/usr/bin/env python3
"""Standalone CLI tool to inspect ClickHouse ground-truth telemetry records and empirical statistics.

Separated from the main filmmaker UI to keep the interface streamlined.
Usage:
    python inspect_telemetry.py
    python inspect_telemetry.py --raw
"""

import argparse
import asyncio
import json
import os
import sys

# Ensure local package path is available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telemetry_agent.mcp_client import mcp_client
from telemetry_agent.schemas import ModelEmpiricalStats


async def display_telemetry(raw: bool = False):
    print("=" * 80)
    print(" 🏛️  CLICKHOUSE GROUND-TRUTH TELEMETRY INSPECTOR")
    print("=" * 80)
    
    # 1. Fetch Aggregated Empirical Metrics
    stats_list = await mcp_client.get_empirical_model_stats()
    
    if not stats_list:
        print("\nNo telemetry records found yet in ClickHouse or local fallback.")
        print("Run a generation session in the web app to record empirical metrics.\n")
        return

    print(f"\n📊 EMPIRICAL MODEL PERFORMANCE & SPEND SUMMARY ({len(stats_list)} models tracked):\n")
    header = f"{'Model Name':<22} | {'Sessions':<8} | {'Accepted':<8} | {'Accept %':<8} | {'Avg Reruns':<10} | {'Base Quote':<10} | {'Avg Cost':<10} | {'Multiplier':<10}"
    print(header)
    print("-" * len(header))
    
    for s in stats_list:
        acc_pct = f"{s.acceptance_rate * 100:.1f}%"
        base_q = f"${s.catalog_base_cost:.4f}"
        avg_c = f"${s.avg_realized_cost:.4f}"
        mult = f"{s.effective_rerun_multiplier:.2f}x"
        print(f"{s.model_name:<22} | {s.total_sessions:<8} | {s.accepted_sessions:<8} | {acc_pct:<8} | {s.avg_rerun_count:<10.2f} | {base_q:<10} | {avg_c:<10} | {mult:<10}")

    print("\n" + "-" * len(header))

    # 2. Raw Log Summary
    if raw or os.path.exists("telemetry_fallback.jsonl"):
        print("\n📁 RAW PERSISTED TELEMETRY LOGS (telemetry_fallback.jsonl):")
        try:
            with open("telemetry_fallback.jsonl", "r", encoding="utf-8") as f:
                lines = f.readlines()
                print(f"Total Logged Records: {len(lines)}")
                for i, line in enumerate(lines[-5:], start=max(1, len(lines) - 4)):
                    record = json.loads(line.strip())
                    print(f"  [{i}] Session: {record.get('session_id')[:8]}... | Model: {record.get('model_name')} | Cost: ${record.get('total_session_cost_usd', 0):.4f} | Reruns: {record.get('rerun_count')} | Accepted: {record.get('user_accepted')}")
        except Exception as e:
            print(f"Could not read raw fallback log: {e}")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect ClickHouse Telemetry Data")
    parser.add_argument("--raw", action="store_true", help="Display raw JSON records from local fallback")
    args = parser.parse_args()

    asyncio.run(display_telemetry(raw=args.raw))


if __name__ == "__main__":
    main()
