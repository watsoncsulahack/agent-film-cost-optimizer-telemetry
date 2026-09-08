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

from telemetry_agent.mcp_client import ClickHouseMCPClient
from telemetry_agent.schemas import ModelEmpiricalStats

mcp_client = ClickHouseMCPClient()


async def display_telemetry(raw: bool = False):
    print("=" * 88)
    print(" 🏛️  CLICKHOUSE GROUND-TRUTH TELEMETRY & DIRECTOR DEFECT DIAGNOSTICS")
    print("=" * 88)
    
    # 1. Fetch Aggregated Empirical Metrics
    stats_list = await mcp_client.fetch_model_empirical_stats()
    
    if not stats_list:
        print("\nNo telemetry records found yet in ClickHouse or local fallback.")
        print("Run a generation session in the web app to record empirical metrics.\n")
    else:
        print(f"\n📊 EMPIRICAL MODEL PERFORMANCE & SPEND SUMMARY ({len(stats_list)} models tracked):\n")
        header = f"{'Model Name':<24} | {'Sessions':<8} | {'Accepted':<8} | {'Accept %':<8} | {'Avg Reruns':<10} | {'Base Quote':<10} | {'Avg Cost':<10} | {'Multiplier':<10}"
        print(header)
        print("-" * len(header))
        
        for s in stats_list:
            acc_pct = f"{s.acceptance_rate * 100:.1f}%"
            base_q = f"${s.avg_base_cost:.4f}"
            avg_c = f"${s.avg_total_cost:.4f}"
            mult = f"{s.effective_cost_multiplier:.2f}x"
            print(f"{s.suggested_model:<24} | {s.total_sessions:<8} | {s.accepted_sessions:<8} | {acc_pct:<8} | {s.avg_rerun_count:<10.2f} | {base_q:<10} | {avg_c:<10} | {mult:<10}")

        print("\n" + "-" * len(header))

    # 2. Raw Log Summary
    fallback_file = mcp_client.fallback_file
    if raw or os.path.exists(fallback_file):
        print(f"\n📁 PERSISTED TELEMETRY LOGS ({fallback_file}):")
        try:
            if os.path.exists(fallback_file):
                with open(fallback_file, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                    print(f"Total Logged Records: {len(lines)}")
                    for i, line in enumerate(lines[-5:], start=max(1, len(lines) - 4)):
                        record = json.loads(line)
                        cat = record.get("feedback_category", "unspecified")
                        fb = record.get("director_feedback", "")
                        fb_str = f" | Defect: [{cat}] {fb[:30]}..." if fb or cat != "unspecified" else ""
                        print(f"  [{i}] Session: {str(record.get('session_id'))[:8]}... | Model: {record.get('suggested_model')} | Cost: ${record.get('total_session_cost', 0):.4f} | Reruns: {record.get('total_rerun_count')} | Accepted: {record.get('user_accepted')}{fb_str}")
            else:
                print("No fallback records found.")
        except Exception as e:
            print(f"Could not read fallback log: {e}")

    print("\n" + "=" * 88 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect ClickHouse Telemetry Data")
    parser.add_argument("--raw", action="store_true", help="Display raw JSON records from local fallback")
    args = parser.parse_args()

    asyncio.run(display_telemetry(raw=args.raw))


if __name__ == "__main__":
    main()
