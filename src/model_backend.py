"""Inference backends. MLX signatures checked against mlx-lm 0.32.2.

There is no automatic fallback: if MLX is unavailable the run fails rather than
quietly producing mock data. Mock output is for pipeline testing only and is
labelled as such in every record it produces.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

MOCK_MODEL_ID = "MOCK-NOT-A-MODEL"
BEDROCK_MODEL_ID = "us.meta.llama3-3-70b-instruct-v1:0"
RELAY_CACHE = "relay_cache.jsonl"
RELAY_PENDING = "relay_pending.json"


class BackendUnavailable(RuntimeError):
    pass


class PendingCall(BaseException):
    """A relay call that has not been made yet.

    A BaseException so that the workflow's `except Exception`, which records model
    failures as outcomes, lets it through: a call not yet made is not an outcome,
    and the replay is simply retried on the next pass.
    """


PLACEHOLDER_PREFIX = "<<pending-reply:"


def placeholder(key: str) -> str:
    """Stands in for a reply that has not been fetched yet. Never reaches a record."""
    return f"{PLACEHOLDER_PREFIX}{key}>>"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: Optional[int]
    generation_tokens: Optional[int]
    finish_reason: Optional[str]
    elapsed_s: float


class MLXBackend:
    name = "mlx"
    is_mock = False

    def __init__(self, model_id: str, offline: bool = True):
        if offline:
            # Must be set before huggingface_hub is imported.
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import mlx.core as mx
            import mlx_lm
            from mlx_lm.generate import stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as error:
            raise BackendUnavailable(f"MLX-LM is not importable: {error}. Use the project .venv.") from error

        self._mx = mx
        self._stream_generate = stream_generate
        self._make_sampler = make_sampler
        self.model_id = model_id
        self.versions = {"mlx_lm": mlx_lm.__version__, "mlx": mx.__version__}
        self.revision = _cached_revision(model_id)
        self._model, self._tokenizer = mlx_lm.load(model_id)

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, seed: int) -> GenerationResult:
        self._mx.random.seed(seed)
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        started = time.perf_counter()
        pieces = []
        last = None
        for response in self._stream_generate(
            self._model, self._tokenizer, prompt, max_tokens=max_tokens, sampler=self._make_sampler(temp=temperature)
        ):
            pieces.append(response.text)
            last = response
        return GenerationResult(
            text="".join(pieces),
            prompt_tokens=last.prompt_tokens if last else None,
            generation_tokens=last.generation_tokens if last else None,
            finish_reason=last.finish_reason if last else None,
            elapsed_s=round(time.perf_counter() - started, 3),
        )


class MockBackend:
    """Pseudo-random, schema-shaped output for testing the pipeline. Not evidence of anything."""

    name = "mock"
    is_mock = True
    model_id = MOCK_MODEL_ID
    revision = None
    versions: dict[str, Any] = {}

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, seed: int) -> GenerationResult:
        digest = hashlib.sha256(f"{seed}|{json.dumps(messages)}".encode("utf-8")).digest()
        roll = digest[0] / 255
        last = messages[-1]["content"]
        if "requested_action" in last:
            if roll < 0.1:
                text = "MOCK malformed output"
            else:
                action = ("submit", "reject", "hold")[digest[1] % 3]
                decision = "permitted" if action == "submit" else "not_permitted"
                text = json.dumps({"decision": decision, "reasons": ["MOCK"], "requested_action": action})
        elif '"status"' in last:
            status = ("REFUSE", "CLEAR") if "CLEAR" in last else ("PASS", "FAIL")
            text = json.dumps({"status": status[digest[1] % 2], "observation": "MOCK observation."})
        else:
            text = "MOCK review text."
        return GenerationResult(text, len(last.split()), len(text.split()), "stop", 0.0)


def relay_key(messages: list[dict[str, str]], max_tokens: int, temperature: float, seed: int) -> str:
    payload = json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class BedrockRelayBackend:
    """Amazon Bedrock, reached in batches through the AWS MCP connector's sandbox.

    There are no local AWS credentials, so this backend never calls Bedrock itself.
    It replays replies from the run's cache and queues every miss; the replay that
    hit a miss is retried on the next pass (see scripts/04_bedrock_relay.py).

    A miss returns a placeholder instead of failing, so the replay runs on and
    queues its later turns too. Where a later turn depends on the reply only as
    an assistant message (a review followed by the final decision, or the two
    staged steps), the relay script makes both calls in one go. The caller must
    discard any result produced while `deferred` is set.

    Bedrock does not accept a seed, so the seed is part of the cache key instead.
    Different seeds are separate draws, and the same prompt with the same seed is
    one draw, which is what lets the staged arms share their first decision.
    """

    name = "bedrock"
    is_mock = False
    revision = None

    def __init__(self, model_id: str, run_dir: Path):
        self.model_id = model_id
        self.versions = {"api": "bedrock-runtime Converse, via AWS MCP run_script"}
        self.cache_path, self.pending_path = run_dir / RELAY_CACHE, run_dir / RELAY_PENDING
        self.cache = load_relay_cache(self.cache_path)
        self.pending: dict[str, dict[str, Any]] = {}
        self.deferred = False

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, seed: int) -> GenerationResult:
        key = relay_key(messages, max_tokens, temperature, seed)
        hit = self.cache.get(key)
        if hit is not None:
            return GenerationResult(hit["text"], hit["prompt_tokens"], hit["generation_tokens"], hit["finish_reason"], hit["elapsed_s"])
        self.pending[key] = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed}
        self.deferred = True
        return GenerationResult(placeholder(key), None, None, "pending", 0.0)

    def save_pending(self) -> None:
        self.pending_path.write_text(json.dumps(self.pending, sort_keys=True))


def load_relay_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["key"]: r for r in records}


def _cached_revision(model_id: str) -> Optional[str]:
    """The commit of a locally cached model, so a run records exactly which weights it used."""
    if os.path.isdir(model_id):
        return None
    try:
        from huggingface_hub import snapshot_download

        return os.path.basename(snapshot_download(model_id, local_files_only=True))
    except Exception:
        return None


def make_backend(name: str, model_id: str, offline: bool = True, run_dir: Optional[Path] = None):
    if name == "mlx":
        return MLXBackend(model_id, offline=offline)
    if name == "mock":
        return MockBackend()
    if name == "bedrock":
        if run_dir is None:
            raise ValueError("The bedrock backend keeps its relay cache in the run directory.")
        return BedrockRelayBackend(model_id, run_dir)
    raise ValueError(f"Unknown backend: {name}")
