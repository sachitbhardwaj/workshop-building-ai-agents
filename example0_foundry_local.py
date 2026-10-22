"""Example 0: meet Foundry Local, the engine that runs every example on this laptop.

Foundry Local is Microsoft's on-device AI runtime. It downloads open models (Qwen, Phi, Mistral...),
runs them on your GPU, NPU or CPU, and serves them on localhost with the same API as OpenAI.
No cloud, no API key, and prompts never leave the machine.

Run:  python example0_foundry_local.py            # guided tour + setup check
      python example0_foundry_local.py --all      # list every model in the catalog
      MODEL=qwen2.5-1.5b python example0_foundry_local.py   # smaller model for slower laptops
"""
import argparse
import os
import time

from foundry_local_sdk import Configuration, FoundryLocalManager
from openai import OpenAI

MODEL_ALIAS = os.getenv("MODEL", "qwen2.5-7b")
APP_NAME = "workshop_rea_demo"  # same as common.FOUNDRY_APP, so every example shares the downloaded model


def step(n, title):
    print(f"\n[{n}] {title}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="list every model, not just chat models with tool calling")
    a = p.parse_args()

    step(1, "Start Foundry Local inside this Python process")
    FoundryLocalManager.initialize(Configuration(app_name=APP_NAME))
    manager = FoundryLocalManager.instance
    print(f"    Models are cached in ~/.{APP_NAME}/ so they download only once.")

    step(2, "Find the hardware it can use (execution providers)")
    for ep in manager.discover_eps():
        print(f"    found {ep.name}")
    result = manager.download_and_register_eps()
    print(f"    ready: {', '.join(result.registered_eps) or 'CPU only'}")

    step(3, "Browse the model catalog")
    catalog = manager.get_catalog()
    seen = set()
    for m in catalog.list_models():
        if m.alias in seen or not (a.all or m.supports_tool_calling):
            continue
        seen.add(m.alias)
        cached = "downloaded" if m.is_cached else ""
        print(f"    {m.alias:<38} {m.info.file_size_mb / 1024:5.1f} GB  {cached}")
    print(f"    {len(seen)} models shown" + ("" if a.all else " (chat models that can call tools; --all for everything)"))

    step(4, f"Download and load {MODEL_ALIAS}")
    model = catalog.get_model(MODEL_ALIAS)
    if model is None:
        raise SystemExit(f"    '{MODEL_ALIAS}' is not in the catalog. Pick an alias from the list above.")
    print(f"    variants: {', '.join(v.id for v in model.variants)}")
    print(f"    chosen for this laptop: {model.id}")
    if not model.is_cached:
        print(f"    downloading {model.info.file_size_mb / 1024:.1f} GB (first run only)...")
        model.download(lambda pct: print(f"\r    {pct:5.1f}%", end="", flush=True))
        print()
    started = time.time()
    model.load()
    print(f"    loaded in {time.time() - started:.1f}s from {model.get_path()}")

    step(5, "Serve it on localhost with an OpenAI-compatible API")
    manager.start_web_service()
    base_url = f"{manager.urls[0]}/v1"
    print(f"    {base_url}  <- any OpenAI SDK, LangChain or agent framework can talk to this")

    step(6, "Ask it something, using the normal OpenAI client")
    client = OpenAI(base_url=base_url, api_key="not-needed-locally")
    started = time.time()
    resp = client.chat.completions.create(
        model=model.id,
        messages=[{"role": "user", "content": "In one sentence: why run an AI model on your own laptop?"}],
    )
    seconds = time.time() - started
    usage = resp.usage
    print(f"    {resp.choices[0].message.content.strip()}")
    print(f"    {usage.prompt_tokens} tokens in, {usage.completion_tokens} out, {seconds:.1f}s "
          f"({usage.completion_tokens / seconds:.0f} tokens/s). Cost: $0.")

    manager.stop_web_service()
    print("\nSetup OK. Every other example uses this same local model through common.py.")


if __name__ == "__main__":
    main()
