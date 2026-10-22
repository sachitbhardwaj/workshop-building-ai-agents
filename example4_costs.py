"""Example 4: put a meter on it. Same stock agent as Example 1, now with token counting.

Run:  python example4_costs.py
      python example4_costs.py --bloat                    # news tool passes the raw API response through
      python example4_costs.py --bloat --budget 3000      # stop when the budget runs out
      python example4_costs.py --model qwen2.5-1.5b       # compare a smaller or bigger model
      python example4_costs.py --price-in 0.15 --price-out 0.60   # what the same tokens would cost in the cloud

On Foundry Local the tokens are free; you pay in laptop time and memory instead.
"""
import argparse

import example1_first_agent as stocks
from common import MODEL, Meter, get_client, resolve_model, run_agent


def bloated_get_company_news(symbol, days=7):
    # A lazy tool: raw articles with summaries, image and link URLs. Capped at 20 so it fits a laptop model.
    return stocks.fetch_news(symbol, days)[:20]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bloat", action="store_true")
    p.add_argument("--budget", type=int)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--price-in", type=float, help="$ per 1M input tokens (check your provider)")
    p.add_argument("--price-out", type=float, help="$ per 1M output tokens (check your provider)")
    a = p.parse_args()

    funcs = dict(stocks.TOOL_FUNCS)
    if a.bloat:
        funcs["get_company_news"] = bloated_get_company_news

    meter = Meter(budget_tokens=a.budget)
    model = resolve_model(a.model)
    client = get_client(model)
    print(f"MODEL: {model} | bloat={'ON' if a.bloat else 'OFF'} | budget={a.budget or 'none'}\n")
    run_agent(client, stocks.SYSTEM, stocks.TASK, stocks.TOOLS, funcs, meter=meter, model=model, verbose=False)

    print(f"\nTASK TOTAL: {meter.summary()}")
    if a.price_in is not None and a.price_out is not None:
        cost = meter.prompt / 1e6 * a.price_in + meter.completion / 1e6 * a.price_out
        print(f"Cost for this task: ${cost:.5f}  ->  x1,000 tasks a day = ${cost * 1000:.2f}/day")


if __name__ == "__main__":
    main()
