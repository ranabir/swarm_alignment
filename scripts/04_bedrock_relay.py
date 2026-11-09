#!/usr/bin/env python3
"""Make the Bedrock calls a relay run has queued, through the AWS MCP sandbox.

There are no local AWS credentials. The AWS MCP connector's run_script sandbox can
call Bedrock but cannot reach this machine, and allows about 100 API calls per
script, so a run advances in passes:

    .venv/bin/python scripts/02_run_experiments.py --backend bedrock --run-name NAME ...
        (replays every cached reply and queues the calls still missing)
    .venv/bin/python scripts/04_bedrock_relay.py script --run-name NAME [--run-name NAME2 ...]
        (writes one sandbox script per batch; run each with run_script)
    .venv/bin/python scripts/04_bedrock_relay.py ingest RESULT_FILE ...
    (repeat until script 02 reports that nothing is pending)

One script can serve several runs, each with its own model. To keep scripts
small, each records block is sent once as compact JSON and rebuilt with the same
json.dumps call the workflow uses, other text is sent as shared lines, and long
lines are split into shared pieces so that labelled and neutral reports share
their wording. A turn whose only dependence on an earlier reply is that reply as
an assistant message (a review then the final decision, or the two staged steps)
is made in the same script, right after the call it follows.

Records are still written only by script 02, through the same code path as any
other backend. Each sandbox script returns the exact system prompt and messages it
sent, and ingest rejects a reply whose prompt differs from the queued one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model_backend import PLACEHOLDER_PREFIX, RELAY_CACHE, RELAY_PENDING, load_relay_cache, placeholder, relay_key  # noqa: E402

CALLS_PER_SCRIPT = 90  # the sandbox allows about 100 API calls per script
CONCURRENCY = 6

TEMPLATE = '''true, false, null = True, False, None
M = __M__
G = __G__
L = __L__
O = __O__
X = __X__
Q = __Q__
P = __P__
C = __C__
def lines(refs):
    out = []
    for r in refs:
        out.extend(L[r[0]:r[1] + 1] if isinstance(r, list) else [L[r]])
    return "\\n".join("".join(G[g] for g in x) if isinstance(x, list) else x for x in out)
def rs(v):
    if isinstance(v, str) and v[:1] == "@" and v[1:].isdigit():
        return rs(O[int(v[1:])])
    if isinstance(v, dict):
        return {k: rs(x) for k, x in v.items()}
    if isinstance(v, list):
        return [rs(x) for x in v]
    return v
def text_of(q, replies):
    spec = Q[q]
    if spec[0] == "R":
        return replies[spec[1]]
    if spec[0] == "J":
        return lines(X[spec[1]]) + json.dumps(rs(O[spec[2]]), indent=1) + lines(X[spec[3]])
    return lines(X[spec[1]])
def parents(p):
    return [Q[q][1] for _, q in P[p][1] if Q[q][0] == "R"]
sem = asyncio.Semaphore(__N__)
T = []
async def one(i, c):
    try:
        replies = {}
        for j in parents(c[0]):
            r = await T[j]
            if "error" in r:
                return {"i": i, "error": "earlier turn failed"}
            replies[j] = r["text"]
        system = text_of(P[c[0]][0], replies)
        messages = [{"role": role, "content": [{"text": text_of(spec, replies)}]} for role, spec in P[c[0]][1]]
        async with sem:
            r = await call_boto3(service_name="bedrock-runtime", operation_name="Converse", region_name="__REGION__",
                params={"modelId": M[c[3]], "system": [{"text": system}], "messages": messages,
                        "inferenceConfig": {"temperature": c[2], "maxTokens": c[1]}})
        # Reasoning models put a reasoningContent block before the text block.
        texts = [b["text"] for b in r["output"]["message"]["content"] if "text" in b]
        return {"i": i, "model": M[c[3]], "system": system, "messages": [[m["role"], m["content"][0]["text"]] for m in messages],
                "text": texts[0] if texts else "", "stop": r.get("stopReason"),
                "in": r["usage"]["inputTokens"], "out": r["usage"]["outputTokens"],
                "ms": r.get("metrics", {}).get("latencyMs", 0)}
    except Exception as e:
        return {"i": i, "error": str(e)[:300]}
for i, c in enumerate(C):
    T.append(asyncio.ensure_future(one(i, c)))
out = await asyncio.gather(*T)
result = {"batch": "__BATCH__", "replies": out, "pad": "." * 120000}
result'''


def _settings() -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / "config.yaml").read_text())["experiment"]


def scripts_dir() -> Path:
    """Shared by all runs, since one script can serve several."""
    path = REPO_ROOT / _settings()["runs_dir"] / "relay_scripts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_paths(run_name: str) -> dict[str, Path]:
    run_dir = REPO_ROOT / _settings()["runs_dir"] / run_name
    if not (run_dir / "run_config.json").exists():
        raise SystemExit(f"No run {run_name}. Start it with script 02 and --backend bedrock first.")
    return {"dir": run_dir, "config": run_dir / "run_config.json", "cache": run_dir / RELAY_CACHE, "pending": run_dir / RELAY_PENDING}


_PLACEHOLDER_RE = re.compile(re.escape(PLACEHOLDER_PREFIX) + r"([0-9a-f]+)>>")
_RECORDS = "RECORDS\n{"


def _compress(refs: list[int]) -> list[Any]:
    """Runs of consecutive line numbers become [first, last]."""
    out: list[Any] = []
    for i in refs:
        if out and isinstance(out[-1], list) and out[-1][1] + 1 == i:
            out[-1][1] = i
        elif out and isinstance(out[-1], int) and out[-1] + 1 == i:
            out[-1] = [out[-1], i]
        else:
            out.append(i)
    return out


def _rows(values: list[Any]) -> str:
    return "[\n" + ",\n".join(json.dumps(v, separators=(",", ":")) for v in values) + "\n]"


Item = tuple[str, str, dict[str, Any], str]  # (run name, key, queued call, model)


def _parent_keys(call: dict[str, Any]) -> Optional[list[str]]:
    """Keys of the queued replies this call depends on, or None if it cannot be chained.

    A placeholder may appear only as a whole assistant message. Anywhere else (a
    generated report built from an unfetched reply, say) the call must wait for a
    later pass, when the real text is known.
    """
    keys = []
    for message in call["messages"]:
        found = _PLACEHOLDER_RE.findall(message["content"])
        if not found:
            continue
        if message["role"] != "assistant" or message["content"] != placeholder(found[0]):
            return None
        keys.append(found[0])
    return keys


class _Encoder:
    def __init__(self) -> None:
        self.segments: list[str] = []
        self.lines: list[Any] = []
        self.objects: list[Any] = []
        self.refs: list[Any] = []
        self.specs: list[Any] = []
        self.prompts: list[Any] = []
        self._index: dict[str, dict[str, int]] = {k: {} for k in ("segment", "line", "object", "refs", "spec", "prompt")}

    def _intern(self, kind: str, table: list[Any], key: str, value: Any) -> int:
        index = self._index[kind]
        if key not in index:
            index[key] = len(table)
            table.append(value)
        return index[key]

    def _line(self, line: str) -> int:
        # Long lines with labels ("Report from Agent B (Availability): PASS: ...")
        # are split after each ": " so that variants share their pieces.
        if len(line) > 60 and ": " in line:
            pieces = [self._intern("segment", self.segments, piece, piece) for piece in re.split(r"(?<=: )", line)]
            return self._intern("line", self.lines, line, pieces)
        return self._intern("line", self.lines, line, line)

    def text(self, content: str) -> int:
        refs = _compress([self._line(line) for line in content.split("\n")])
        return self._intern("refs", self.refs, json.dumps(refs), refs)

    def content(self, content: str, position: dict[str, int]) -> int:
        spec = self._spec(content, position)
        return self._intern("spec", self.specs, json.dumps(spec), spec)

    def _spec(self, content: str, position: dict[str, int]) -> list[Any]:
        found = _PLACEHOLDER_RE.fullmatch(content)
        if found:
            return ["R", position[found.group(1)]]
        start = content.find(_RECORDS)
        if start >= 0:
            start += len(_RECORDS) - 1
            try:
                value, end = json.JSONDecoder().raw_decode(content, start)
            except json.JSONDecodeError:
                value = None
            if value is not None and json.dumps(value, indent=1) == content[start:end] and "@" not in json.dumps(value):
                return ["J", self.text(content[:start]), self._object(value), self.text(content[end:])]
        return ["T", self.text(content)]

    def _object(self, value: Any) -> int:
        """Intern a records object; sizeable nested values become "@n" references."""
        def ref(v: Any) -> Any:
            if isinstance(v, (dict, list)) and len(json.dumps(v)) > 40:
                return f"@{self._object(v)}"
            return v
        shrunk = {k: ref(v) for k, v in value.items()} if isinstance(value, dict) else [ref(v) for v in value]
        return self._intern("object", self.objects, json.dumps(shrunk), shrunk)

    def prompt(self, messages: list[dict[str, str]], position: dict[str, int]) -> int:
        if messages[0]["role"] != "system" or any(m["role"] == "system" for m in messages[1:]):
            raise ValueError("expected exactly one leading system message")
        value = [self.content(messages[0]["content"], position), [[m["role"], self.content(m["content"], position)] for m in messages[1:]]]
        return self._intern("prompt", self.prompts, json.dumps(value), value)


def build_script(batch: list[Item], name: str, region: str) -> str:
    encoder = _Encoder()
    models: list[str] = []
    # Placeholders refer to their call's position in this batch, per run.
    calls, positions = [], {}
    for i, (run, key, call, model) in enumerate(batch):
        if model not in models:
            models.append(model)
        prompt = encoder.prompt(call["messages"], positions.setdefault(run, {}))
        positions[run][key] = i
        calls.append([prompt, call["max_tokens"], call["temperature"], models.index(model)])
    code = (
        TEMPLATE.replace("__M__", json.dumps(models))
        .replace("__G__", _rows(encoder.segments))
        .replace("__L__", _rows(encoder.lines))
        .replace("__O__", _rows(encoder.objects))
        .replace("__X__", _rows(encoder.refs))
        .replace("__Q__", _rows(encoder.specs))
        .replace("__P__", _rows(encoder.prompts))
        .replace("__C__", _rows(calls))
        .replace("__N__", str(CONCURRENCY))
        .replace("__REGION__", region)
        .replace("__BATCH__", name)
    )
    check_script(code, batch)
    return code


def check_script(code: str, batch: list[Item]) -> None:
    """Run the script's data and builders locally and confirm they rebuild every queued call.

    Chained calls are rebuilt with each earlier reply standing in as its placeholder,
    which is exactly how they were queued.
    """
    namespace: dict[str, Any] = {"json": json}  # pre-imported in the sandbox
    exec(code.split("sem = asyncio")[0], namespace)
    text_of, models, prompts = namespace["text_of"], namespace["M"], namespace["P"]
    for i, ((run, key, call, model), (p, max_tokens, temperature, m)) in enumerate(zip(batch, namespace["C"], strict=True)):
        replies = {j: placeholder(batch[j][1]) for j in namespace["parents"](p)}
        if any(j >= i for j in replies):
            raise AssertionError(f"call {run}/{key} comes before a turn it follows")
        system_spec, turns = prompts[p]
        rebuilt = [{"role": "system", "content": text_of(system_spec, replies)}] + [{"role": role, "content": text_of(spec, replies)} for role, spec in turns]
        if rebuilt != call["messages"] or (max_tokens, temperature, models[m]) != (call["max_tokens"], call["temperature"], model):
            raise AssertionError(f"script would not rebuild queued call {run}/{key}")


def _batches(items: list[Item]) -> list[list[Item]]:
    """Each call with the chained calls that follow it, packed into script-sized batches."""
    by_key = {(run, key): (run, key, call, model) for run, key, call, model in items}
    children: dict[tuple[str, str], list[Item]] = {}
    roots = []
    for item in items:
        run, key, call, _ = item
        parents = _parent_keys(call)
        if parents is None or any((run, k) not in by_key for k in parents):
            continue  # waits for a later pass
        if not parents:
            roots.append(item)
        elif len(parents) == 1 and not _parent_keys(by_key[(run, parents[0])][2]):
            children.setdefault((run, parents[0]), []).append(item)
    order = lambda item: (json.dumps(item[2]["messages"]), item[3], item[0], item[1])
    groups = [[root] + sorted(children.get((root[0], root[1]), []), key=order) for root in sorted(roots, key=order)]
    batches: list[list[Item]] = [[]]
    for group in groups:
        if len(batches[-1]) + len(group) > CALLS_PER_SCRIPT:
            batches.append([])
        batches[-1].extend(group)
    return [b for b in batches if b]


def cmd_script(args: argparse.Namespace) -> int:
    items: list[Item] = []
    for run in args.run_name:
        paths = run_paths(run)
        pending = json.loads(paths["pending"].read_text()) if paths["pending"].exists() else {}
        model = json.loads(paths["config"].read_text())["model"]
        items.extend((run, key, call, model) for key, call in pending.items())
    batches = _batches(items)
    if not batches:
        print("Nothing to send. Run script 02 with --backend bedrock first.")
        return 0
    out_dir = scripts_dir()
    first = len(list(out_dir.glob("batch*.keys.json")))
    total = 0
    for number, batch in enumerate(batches, start=first):
        name = f"batch{number:04d}"
        code = build_script(batch, name, _settings().get("bedrock_region", "us-east-1"))
        (out_dir / f"{name}.py").write_text(code)
        (out_dir / f"{name}.keys.json").write_text(json.dumps([[run, key] for run, key, _, _ in batch]))
        total += len(code)
        print(f"{out_dir / (name + '.py')}  ({len(batch)} calls, {len(code)} chars)")
    print(f"{sum(len(b) for b in batches)} of {len(items)} queued calls in {len(batches)} scripts, {total} chars")
    return 0


def _load_result(path: Path) -> dict[str, Any]:
    """A saved run_script result, either bare or wrapped in a list of text blocks."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        data = json.loads(next(block["text"] for block in data if block.get("type") == "text"))
    return data.get("return_value", data)


def cmd_ingest(args: argparse.Namespace) -> int:
    pending: dict[str, dict[str, Any]] = {}
    caches: dict[str, dict[str, Any]] = {}
    models: dict[str, str] = {}
    handles: dict[str, Any] = {}
    ok = errors = rejected = 0
    try:
        for file in args.files:
            result = _load_result(Path(file))
            keys = json.loads((scripts_dir() / f"{result['batch']}.keys.json").read_text())
            texts = {}  # (run, queued key) -> reply text, for the chained calls that follow
            for reply in sorted(result["replies"], key=lambda r: r["i"]):
                run, key = keys[reply["i"]]
                if run not in caches:
                    paths = run_paths(run)
                    pending[run] = json.loads(paths["pending"].read_text()) if paths["pending"].exists() else {}
                    caches[run] = load_relay_cache(paths["cache"])
                    models[run] = json.loads(paths["config"].read_text())["model"]
                    handles[run] = paths["cache"].open("a")
                if reply.get("error"):
                    errors += 1
                    print(f"  {run}/{key}: {reply['error']}", file=sys.stderr)
                    continue
                call = pending[run].get(key)
                if call is None:
                    rejected += 1
                    continue
                # A chained call was queued with placeholders; the real call carried the replies.
                sent = [{"role": m["role"], "content": _PLACEHOLDER_RE.sub(lambda f: texts.get((run, f.group(1)), f.group(0)), m["content"])} for m in call["messages"]]
                echoed = [{"role": "system", "content": reply["system"]}] + [{"role": r, "content": c} for r, c in reply["messages"]]
                if echoed != sent or reply["model"] != models[run] or PLACEHOLDER_PREFIX in json.dumps(sent):
                    rejected += 1
                    continue
                texts[(run, key)] = reply["text"]
                real_key = relay_key(sent, call["max_tokens"], call["temperature"], call["seed"])
                if real_key in caches[run]:
                    continue
                record = {
                    "key": real_key,
                    "text": reply["text"],
                    "prompt_tokens": reply["in"],
                    "generation_tokens": reply["out"],
                    # Converse says max_tokens where MLX says length; the workflow checks for length.
                    "finish_reason": "length" if reply["stop"] == "max_tokens" else "stop",
                    "stop_reason": reply["stop"],
                    "elapsed_s": round(reply["ms"] / 1000, 3),
                }
                caches[run][real_key] = record
                handles[run].write(json.dumps(record, sort_keys=True) + "\n")
                ok += 1
    finally:
        for handle in handles.values():
            handle.close()
    print(f"Ingested {ok} into {len(caches)} runs; errors left queued {errors}; mismatches rejected {rejected}")
    return 1 if rejected else 0


def cmd_cost(args: argparse.Namespace) -> int:
    """Token counts only: prices differ by model, so look them up for the model in each run."""
    for run in args.run_name:
        cache = load_relay_cache(run_paths(run)["cache"])
        tokens_in = sum(r["prompt_tokens"] for r in cache.values())
        tokens_out = sum(r["generation_tokens"] for r in cache.values())
        print(f"{run}: {len(cache)} calls, {tokens_in} in / {tokens_out} out tokens")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["script", "ingest", "cost"])
    parser.add_argument("files", nargs="*", type=Path, help="Saved run_script results, for ingest")
    parser.add_argument("--run-name", action="append", default=[], help="Repeat for several runs (script, cost)")
    # Intermixed, so result files may come before or after --run-name.
    args = parser.parse_intermixed_args()
    if args.command != "ingest" and not args.run_name:
        parser.error("--run-name is required for script and cost")
    return {"script": cmd_script, "ingest": cmd_ingest, "cost": cmd_cost}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
