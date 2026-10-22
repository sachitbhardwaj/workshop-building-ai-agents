"""Shared helpers for the workshop examples: client setup, the agent loop and a token meter.

By default every example runs on a local model through Foundry Local (no API key, nothing leaves the laptop).
Set AZURE_OPENAI_ENDPOINT or OPENAI_API_KEY to use a cloud model instead.
"""
import atexit
import json
import os

from openai import AzureOpenAI, InternalServerError, OpenAI

LOCAL_MAX_TOKENS = 2048


def _load_dotenv(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Tiny .env reader (KEY=value per line) so no extra package is needed. Real env vars win."""
    if os.path.exists(path):
        for line in open(path):
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


_load_dotenv()

if os.getenv("AZURE_OPENAI_ENDPOINT"):
    BACKEND = "azure"
elif os.getenv("OPENAI_API_KEY"):
    BACKEND = "openai"
else:
    BACKEND = "foundry"

FOUNDRY_APP = "workshop_rea_demo"  # also names the model cache folder: ~/.workshop_rea_demo
_foundry = None


def _foundry_manager():
    global _foundry
    if _foundry is None:
        from foundry_local_sdk import Configuration, FoundryLocalManager
        FoundryLocalManager.initialize(Configuration(app_name=FOUNDRY_APP))
        _foundry = FoundryLocalManager.instance
        # Registers the GPU/NPU runtime; needed on every run before a model can load
        _foundry.download_and_register_eps()
    return _foundry


def _foundry_model(name):
    catalog = _foundry_manager().get_catalog()
    model = catalog.get_model(name) or catalog.get_model_variant(name)
    if model is None:
        raise SystemExit(f"'{name}' is not in the Foundry Local catalog. Try qwen2.5-7b or qwen2.5-1.5b.")
    return model


def resolve_model(name):
    """Foundry Local's server wants the full model id (e.g. qwen2.5-7b-instruct-generic-gpu:4), not the alias."""
    return _foundry_model(name).id if BACKEND == "foundry" else name


MODEL = resolve_model(os.getenv("MODEL", "qwen2.5-7b" if BACKEND == "foundry" else "gpt-4o-mini"))


def start_foundry(name=None):
    """Download (first run only) and load a local model, then serve it on an OpenAI-compatible localhost URL."""
    manager, model = _foundry_manager(), _foundry_model(name or MODEL)
    if not model.is_cached:
        print(f"[foundry] downloading {model.alias} (first run only)...")
        model.download(lambda p: print(f"\r  {p:5.1f}%", end="", flush=True))
        print()
    if not model.is_loaded:
        model.load()
    if not manager.urls:
        manager.start_web_service()
        atexit.register(manager.stop_web_service)
    print(f"[foundry] {model.id} running locally at {manager.urls[0]}/v1")
    return manager, model


def get_client(model=None):
    if BACKEND == "azure":
        return AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    if BACKEND == "openai":
        # Reads OPENAI_API_KEY, plus OPENAI_BASE_URL for any OpenAI-compatible endpoint
        return OpenAI()
    manager, _ = start_foundry(model)
    client = OpenAI(base_url=f"{manager.urls[0]}/v1", api_key="not-needed-locally")
    create = client.chat.completions.create

    def create_local(**kwargs):
        # Local models sometimes run out of room mid tool call (HTTP 500): give more room, retry once
        kwargs.setdefault("max_tokens", LOCAL_MAX_TOKENS)
        try:
            return create(**kwargs)
        except InternalServerError:
            return create(**kwargs)

    client.chat.completions.create = create_local
    return client


def assistant_entry(msg):
    """Turn the model's reply into a plain dict we can send back next step."""
    entry = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        entry["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in msg.tool_calls
        ]
    return entry


def approve(action):
    """Human-in-the-loop gate: the agent only proceeds if a person types y."""
    return input(f"\n  Agent wants to: {action}\n  Allow? [y/N] ").strip().lower() == "y"


class Meter:
    """Counts tokens across every model call in a task."""

    def __init__(self, budget_tokens=None):
        self.budget = budget_tokens
        self.calls = self.prompt = self.completion = 0

    @property
    def total(self):
        return self.prompt + self.completion

    def add(self, usage):
        self.calls += 1
        self.prompt += usage.prompt_tokens
        self.completion += usage.completion_tokens
        print(f"  [meter] call {self.calls}: in={usage.prompt_tokens} out={usage.completion_tokens} "
              f"running total={self.total}")

    def over_budget(self, next_call_tokens=0):
        return self.budget is not None and self.total + next_call_tokens >= self.budget

    def summary(self):
        return (f"{self.calls} calls | {self.prompt} input + {self.completion} output "
                f"= {self.total} tokens")


def run_agent(client, system_prompt, user_message, tools, tool_funcs,
              max_steps=10, meter=None, model=None, verbose=True):
    """THINK -> DO -> CHECK -> REPEAT until the model answers without a tool call."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}]
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(model=model or MODEL, messages=messages, tools=tools)
        if meter:
            meter.add(resp.usage)
        msg = resp.choices[0].message
        messages.append(assistant_entry(msg))

        if not msg.tool_calls:
            if verbose:
                print(f"[step {step}] FINAL: {msg.content}")
            return msg.content, messages

        if verbose and msg.content:
            print(f"[step {step}] THINK: {msg.content}")
        new_chars = 0
        for call in msg.tool_calls:
            name, raw = call.function.name, call.function.arguments
            try:
                args = json.loads(raw or "{}")
                result = tool_funcs[name](**args)
            except Exception as exc:
                # Send the error back so the model can recover instead of crashing the loop
                args, result = raw, {"error": f"{type(exc).__name__}: {exc}"}
            if verbose:
                print(f"[step {step}] DO: {name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
            new_chars += len(messages[-1]["content"])

        # Look ahead: the next call re-sends everything so far plus the new tool results (~4 chars per token)
        next_call = resp.usage.prompt_tokens + resp.usage.completion_tokens + new_chars // 4
        if meter and meter.over_budget(next_call):
            print(f"[budget] Next call needs ~{next_call} tokens: over budget, handing over to a human.")
            return None, messages

    print("[limit] Max steps reached: stopping and handing over to a human.")
    return None, messages
