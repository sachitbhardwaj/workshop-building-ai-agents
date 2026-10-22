"""Example 3: grounding the agent in your documents, and what messy data does to it.

Run:  python example3_grounding.py messy                 # old versions included, no metadata
      python example3_grounding.py clean                 # metadata on, superseded docs filtered out
      python example3_grounding.py clean "What warranty do headphones have?"
"""
import sys

from common import get_client, run_agent

TOP_K = 2

# A tiny knowledge base. Note the two out-of-date documents: every company has them.
DOCS = [
    {"id": "faq-2023", "title": "Returns FAQ", "effective": "2023-05-01", "status": "superseded",
     "text": "Can I return an item for a refund? Yes, returns are accepted within 60 days of delivery."},
    {"id": "policy-2024", "title": "Refund policy", "effective": "2024-01-01", "status": "superseded",
     "text": "Customers can return items within 60 days of delivery for a full refund. "
             "Refunds up to $200 are approved automatically."},
    {"id": "policy-2026", "title": "Refund policy", "effective": "2026-07-01", "status": "current",
     "text": "Customers can return items within 30 days of delivery for a full refund. "
             "Refunds over $100 need human approval."},
    {"id": "shipping-2026", "title": "Shipping policy", "effective": "2026-03-01", "status": "current",
     "text": "Standard delivery takes 3 to 5 working days. Express delivery takes 1 to 2 working days."},
    {"id": "warranty-2025", "title": "Warranty", "effective": "2025-05-01", "status": "current",
     "text": "Electronics such as headphones have a 1-year warranty covering manufacturing defects."},
]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MODE = args[0] if args and args[0] in ("messy", "clean") else "clean"
QUESTION = " ".join(args[1:]) or "I received my headphones 45 days ago. Can I still return them for a refund?"


def search_docs(query):
    """Naive keyword search. Real systems use embeddings, but the data problems are identical."""
    words = {w.strip(".,?!'\"").lower() for w in query.split() if len(w) > 2}

    def score(doc):
        haystack = f"{doc['title']} {doc['text']}".lower()
        return sum(w in haystack for w in words)

    docs = DOCS if MODE == "messy" else [d for d in DOCS if d["status"] == "current"]
    top = sorted(docs, key=score, reverse=True)[:TOP_K]
    if MODE == "messy":
        return {"results": [{"text": d["text"]} for d in top]}
    return {"results": top}


TOOLS = [{"type": "function", "function": {
    "name": "search_docs",
    "description": "Search Acme's policy documents. Use it before answering any policy question.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

SYSTEM = """You are a customer support assistant for Acme Online Store.
- Answer only from search_docs results; never from memory.
- If results conflict, prefer status "current" and the latest "effective" date.
- If the documents don't answer the question, say you don't know.
- Cite the document id in square brackets when one is available."""


def main():
    print(f"MODE: {MODE}\nQUESTION: {QUESTION}\n")
    run_agent(get_client(), SYSTEM, QUESTION, TOOLS, {"search_docs": search_docs})


if __name__ == "__main__":
    main()
