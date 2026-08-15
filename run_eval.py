"""Run all 10 assignment questions and print answers + sources.
Usage: python run_eval.py
Requires GEMINI_API_KEY (or OPENAI_API_KEY) in .env and indexed chroma_db.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ingest import chunk_count_fast, ingest_data_folder
from rag import ask

QUESTIONS = [
    "Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?",
    "How many line stoppages happened in Q1, what was the total downtime, and what caused them?",
    "What is the approval authority for a purchase order worth ₹1.4 crore?",
    "What are the four supplier classification categories, and what qualifies a supplier as Critical?",
    "Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million in Q1. Which policy clauses does this trigger, and what exactly must the buyer do?",
    "The microcontroller supplier is single-source. What does the sourcing policy require in this situation, and what is the company already doing about it?",
    "Microcontrollers are imported with a 46-day lead time. Using the safety-stock policy, how many days of stock should be held for this part?",
    "Trident Circuit Boards had a defect rate of 640 parts per million. What is the cost consequence under the policy?",
    "Which suppliers would fall below the B rating band on on-time delivery alone, and what is the escalation path for them?",
    "What is the annual salary of the Head of Procurement?",
]


def main() -> None:
    if chunk_count_fast() == 0:
        print("Indexing data/ …")
        print(ingest_data_folder(clear_first=True, skip_if_unchanged=False))

    rows = []
    for i, q in enumerate(QUESTIONS, start=1):
        print(f"\n=== Q{i} ===\n{q}")
        result = ask(q, top_k=6)
        print(result["answer"])
        print("Sources:", result["sources"])
        print("Docs:", list((result.get("grouped_sources") or {}).keys()))
        rows.append(
            {
                "id": i,
                "question": q,
                "answer": result["answer"],
                "sources": result["sources"],
                "docs": list((result.get("grouped_sources") or {}).keys()),
                "latency_ms": result.get("latency_ms"),
            }
        )

    out = Path("eval_results.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
