"""Example 1: your first agent. A stock research assistant with live market data from Finnhub.

Needs a free API key from https://finnhub.io in workshop_labs/.env:   FINNHUB_API_KEY=...

Run:  python example1_first_agent.py
      python example1_first_agent.py "your own question"
      python example1_first_agent.py --approve "Add Nvidia to my watchlist with a one-line note"
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import truststore

from common import MODEL, approve, assistant_entry, get_client

# Trust the OS certificate store, so HTTPS also works behind corporate proxies
truststore.inject_into_ssl()

MAX_STEPS = 8
TODAY = date.today()
NEEDS_APPROVAL = "--approve" in sys.argv
WATCHLIST = Path(__file__).with_name("watchlist.json")


def finnhub(endpoint, **params):
    """One GET to the Finnhub REST API. The key travels in a header, so it never appears in URLs or logs."""
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise SystemExit("Set FINNHUB_API_KEY in workshop_labs/.env (free key at https://finnhub.io).")
    url = f"https://finnhub.io/api/v1/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"X-Finnhub-Token": key})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


# ---------- tools: plain Python functions ----------
def search_symbol(query):
    data = finnhub("search", q=query)
    return {"matches": [{"symbol": r["symbol"], "name": r["description"], "type": r["type"]}
                        for r in data.get("result", [])[:5]]}


def get_quote(symbol):
    q = finnhub("quote", symbol=symbol.upper())
    if not q.get("t"):
        return {"error": f"No quote for '{symbol}'. Use search_symbol to find the right ticker."}
    return {"symbol": symbol.upper(), "price": q["c"], "change": q["d"], "change_percent": round(q["dp"], 2),
            "day_high": q["h"], "day_low": q["l"], "previous_close": q["pc"],
            "as_of": datetime.fromtimestamp(q["t"]).isoformat(timespec="minutes")}


def fetch_news(symbol, days=7):
    """The raw API response: every article, with summaries, image and link URLs."""
    start = TODAY - timedelta(days=int(days))
    return finnhub("company-news", symbol=symbol.upper(), **{"from": start.isoformat(), "to": TODAY.isoformat()})


def get_company_news(symbol, days=7):
    articles = fetch_news(symbol, days)
    return {"symbol": symbol.upper(), "articles_found": len(articles),
            "latest": [{"date": date.fromtimestamp(a["datetime"]).isoformat(), "source": a["source"],
                        "headline": a["headline"]} for a in articles[:5]]}


def add_to_watchlist(symbol, note):
    """The only tool that changes anything, so it's the one behind the approval gate."""
    if NEEDS_APPROVAL and not approve(f"add {symbol.upper()} to your watchlist: '{note}'"):
        return {"status": "NOT added", "reason": "The user declined. Tell them it was not added."}
    items = json.loads(WATCHLIST.read_text()) if WATCHLIST.exists() else []
    items.append({"symbol": symbol.upper(), "note": note, "added": TODAY.isoformat()})
    WATCHLIST.write_text(json.dumps(items, indent=2))
    print(f"    >> saved to {WATCHLIST.name}: {symbol.upper()} - {note}")
    return {"status": "added", "symbol": symbol.upper(), "watchlist_size": len(items)}


TOOL_FUNCS = {"search_symbol": search_symbol, "get_quote": get_quote,
              "get_company_news": get_company_news, "add_to_watchlist": add_to_watchlist}

# ---------- tool schemas: what the model is told about each tool ----------
TOOLS = [
    {"type": "function", "function": {
        "name": "search_symbol",
        "description": "Find the stock ticker for a company name. Use when you don't know the ticker.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Company name, e.g. Microsoft"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_quote",
        "description": "Get the latest price and today's change for a US stock ticker.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Ticker, e.g. MSFT"}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "get_company_news",
        "description": "Get the latest news headlines about a company.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Ticker, e.g. MSFT"},
            "days": {"type": "integer", "description": "How many days back to look (default 7)"}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "add_to_watchlist",
        "description": "Save a ticker to the user's watchlist with a short note.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"},
            "note": {"type": "string", "description": "One line on why it's worth watching"}},
            "required": ["symbol", "note"]}}},
]

SYSTEM = f"""You are a stock research assistant. Today is {TODAY}.
- Do the work by calling tools: your first reply must be a tool call. Never just describe what you are about to do.
- If you only have a company name, call search_symbol first to get the ticker.
- Use only the numbers and headlines the tools return, copied exactly. Never invent prices or news.
- You give information, not financial advice. Never tell the user to buy or sell.
- Finish every part of the request before you reply. If the user mentions the watchlist,
  call add_to_watchlist before your final answer.
- When you are done, reply with a short summary (at most 5 lines): price, today's change,
  and up to 3 headlines if you fetched news."""

TASK = "How is Microsoft's stock doing today, and what's in the news this week?"


def main():
    client = get_client()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    task = " ".join(args) or TASK
    print(f"TASK: {task}\n")

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        messages.append(assistant_entry(msg))          # remember what it decided

        if not msg.tool_calls:                         # no tool needed: done
            print(f"[step {step}] FINAL: {msg.content}")
            break

        for call in msg.tool_calls:                    # DO: run each tool it asked for
            args = json.loads(call.function.arguments)
            print(f"[step {step}] DO: {call.function.name}({args})")
            result = TOOL_FUNCS[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)})   # CHECK: feed the result back
    else:
        print("Max steps reached: ask a human.")

    print(f"\nModel calls used: {step}")


if __name__ == "__main__":
    main()
