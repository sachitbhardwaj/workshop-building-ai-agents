"""Example 2: shrink what the agent reads with Headroom (github.com/headroomlabs-ai/headroom).

The system prompt is only one part of what the model reads on every step; tool output is usually
the biggest part. Headroom compresses tool outputs, logs and documents on your machine before they
reach the model, and never touches the system prompt or the user's question.

Run:  python example2_context_headroom.py          # same question, without and then with Headroom
      python example2_context_headroom.py --peek   # also show what the model actually receives
"""
import argparse
import json
import os
import random

os.environ.setdefault("HEADROOM_TELEMETRY", "off")  # Headroom's anonymous usage beacon
from headroom import compress  # noqa: E402

from common import MODEL, Meter, assistant_entry, get_client  # noqa: E402

SYSTEM = ("You are an on-call assistant. Always call get_job_runs before answering. "
          "Answer in two short lines: which job failed, and why.")
QUESTION = "Why did last night's sync job fail?"

TOOLS = [{"type": "function", "function": {
    "name": "get_job_runs",
    "description": "Return all of last night's sync job runs as JSON.",
    "parameters": {"type": "object", "properties": {}}}}]


def get_job_runs():
    """A typical monitoring API: 300 near-identical records and one that matters."""
    rng = random.Random(7)
    jobs = [{"job_id": f"sync-{i:04d}", "started": f"2026-10-20T02:{i // 60:02d}:{i % 60:02d}Z",
             "status": "success", "rows": rng.randint(900, 1100), "latency_ms": rng.randint(80, 140),
             "worker": "sync-worker-2", "region": "westeurope", "error": None} for i in range(300)]
    jobs[211].update(status="failed", error="db write failed: disk quota exceeded on /data (used 100%)")
    return {"jobs": jobs}


def ask(client, use_headroom, peek):
    meter = Meter()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    for _ in range(4):
        sent = messages
        if use_headroom:
            # The model name only picks the tokenizer Headroom counts with
            result = compress(messages, model="gpt-4o")
            sent = result.messages
            if result.tokens_saved:
                print(f"  [headroom] {result.tokens_before:,} -> {result.tokens_after:,} tokens "
                      f"({result.compression_ratio:.0%} smaller) via {', '.join(result.transforms_applied)}")
                if peek:
                    print(f"  [headroom] the model now reads:\n{sent[-1]['content'][:400]}\n  ...")
        resp = client.chat.completions.create(model=MODEL, messages=sent, tools=TOOLS)
        meter.add(resp.usage)
        msg = resp.choices[0].message
        messages.append(assistant_entry(msg))
        if not msg.tool_calls:
            return msg.content, meter
        for call in msg.tool_calls:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(get_job_runs())})
    return "(no answer within 4 steps)", meter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--peek", action="store_true", help="print the start of the compressed tool output")
    a = p.parse_args()
    client = get_client()
    print(f"QUESTION: {QUESTION}\n")

    results = {}
    for label, use_headroom in (("WITHOUT Headroom", False), ("WITH Headroom", True)):
        print(f"=== {label}")
        answer, meter = ask(client, use_headroom, a.peek)
        print(f"ANSWER: {answer}\n")
        results[label] = meter

    print("=== Comparison")
    for label, m in results.items():
        print(f"  {label:<17} {m.calls} calls | {m.prompt:>6,} input tokens | {m.completion:>4,} output tokens")


if __name__ == "__main__":
    main()
