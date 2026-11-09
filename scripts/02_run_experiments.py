#!/usr/bin/env python3
"""Run experiment replays and append every result to a resumable run directory.

Usage:
    .venv/bin/python scripts/02_run_experiments.py --run-name smoke --limit-pairs 4 --reps 1
    .venv/bin/python scripts/02_run_experiments.py --backend mock --run-name pipeline_check
    .venv/bin/python scripts/02_run_experiments.py --backend bedrock --run-name smoke_llama70b --limit-pairs 2
    .venv/bin/python scripts/02_run_experiments.py --freeze-prompts

Held-out cases are refused until the prompts have been frozen, and refused again
if the prompts change afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agents import load_prompts, prompts_sha256  # noqa: E402
from src.conditions import CONDITIONS, PRIMARY_CONDITIONS, UpstreamReports, controlled_reports  # noqa: E402
from src.metrics import format_outcome_table  # noqa: E402
from src.model_backend import PLACEHOLDER_PREFIX, BackendUnavailable, PendingCall, make_backend  # noqa: E402
from src.policy_engine import POLICY_SHA256  # noqa: E402
from src.schemas import DEV, TEST, Scenario  # noqa: E402
from src.workflow import OUTCOME_MODEL_ERROR, GenerationSettings, generate_upstream, run_replay, stable_seed  # noqa: E402

MAX_CONSECUTIVE_ERRORS = 5


def load_scenarios(scenarios_dir: Path) -> tuple[list[Scenario], str]:
    payload = (scenarios_dir / "scenarios.json").read_text()
    sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    expected = json.loads((scenarios_dir / "manifest.json").read_text())["scenarios_sha256"]
    if sha256 != expected:
        raise SystemExit("scenarios.json does not match its manifest. Regenerate it with script 01.")
    return [Scenario.from_dict(r) for r in json.loads(payload)], sha256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read completed records, dropping a final line cut off by an interrupted run."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise SystemExit(f"{path}: corrupt record on line {index + 1}.")
            print(f"Dropping incomplete final line in {path}", file=sys.stderr)
            path.write_text("".join(l + "\n" for l in lines[:-1]))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_jobs(scenarios: list[Scenario], conditions: list[str], reps: int, seed: int) -> list[tuple[int, Scenario, str]]:
    """Same cases in every condition, in a seeded order that is randomized within each pair."""
    by_pair: dict[str, list[Scenario]] = {}
    for s in scenarios:
        by_pair.setdefault(s.pair_id, []).append(s)
    jobs = []
    for rep in range(reps):
        for pair_id in sorted(by_pair):
            rng = random.Random(stable_seed(seed, "order", pair_id, rep))
            pair_jobs = [(rep, s, c) for s in by_pair[pair_id] for c in conditions]
            rng.shuffle(pair_jobs)
            jobs.extend(pair_jobs)
    return jobs


def check_prompt_freeze(lock_path: Path, current: str) -> None:
    if not lock_path.exists():
        raise SystemExit("Held-out cases require frozen prompts. Debug on the dev split, then run --freeze-prompts.")
    frozen = json.loads(lock_path.read_text())["prompts_sha256"]
    if frozen != current:
        raise SystemExit("Prompts have changed since they were frozen, so held-out results would not be valid.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--run-name")
    parser.add_argument("--backend", choices=["mlx", "mock", "bedrock"])
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=["controlled", "generated"])
    parser.add_argument("--split", choices=[DEV, TEST, "all"])
    parser.add_argument("--conditions", nargs="+", help=f"Default: the primary ones. Choices: {', '.join(CONDITIONS)}")
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--freeze-prompts", action="store_true", help="Record the prompt hash and exit")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    settings = config["experiment"]
    lock_path = REPO_ROOT / settings["prompt_lock"]
    current_prompts = prompts_sha256()

    if args.freeze_prompts:
        lock_path.write_text(json.dumps({"prompts_sha256": current_prompts}, indent=2) + "\n")
        print(f"Froze prompts at {current_prompts[:16]} in {lock_path}")
        return 0
    if not args.run_name:
        parser.error("--run-name is required")

    def pick(name: str) -> Any:
        value = getattr(args, name)
        return settings[name] if value is None else value

    backend_name, model_id, mode, split = pick("backend"), pick("model"), pick("mode"), pick("split")
    seed, reps, temperature = pick("seed"), pick("reps"), pick("temperature")
    limit_pairs = args.limit_pairs
    conditions = args.conditions or list(PRIMARY_CONDITIONS)
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        parser.error(f"Unknown conditions: {unknown}")

    # Mock output must never be mistaken for model behavior, even by its directory name.
    run_name = args.run_name
    if backend_name == "mock" and not run_name.startswith("mock_"):
        run_name = f"mock_{run_name}"
    if backend_name != "mock" and run_name.startswith("mock_"):
        parser.error("Run names starting with mock_ are reserved for the mock backend.")

    if backend_name != "mock" and split in (TEST, "all"):
        check_prompt_freeze(lock_path, current_prompts)

    scenarios, scenarios_sha = load_scenarios(REPO_ROOT / config["scenarios"]["output_dir"])
    if split != "all":
        scenarios = [s for s in scenarios if s.split == split]
    if limit_pairs is not None:
        keep = sorted({s.pair_id for s in scenarios})[:limit_pairs]
        scenarios = [s for s in scenarios if s.pair_id in keep]

    generation = GenerationSettings(
        temperature=temperature,
        upstream_max_tokens=settings["upstream_max_tokens"],
        review_max_tokens=settings["review_max_tokens"],
        decision_max_tokens=settings["decision_max_tokens"],
    )
    run_dir = REPO_ROOT / settings["runs_dir"] / run_name
    run_config = {
        "backend": backend_name,
        "is_mock": backend_name == "mock",
        "model": model_id if backend_name != "mock" else "MOCK-NOT-A-MODEL",
        "mode": mode,
        "split": split,
        "limit_pairs": limit_pairs,
        "conditions": conditions,
        "reps": reps,
        "seed": seed,
        "temperature": temperature,
        "upstream_max_tokens": generation.upstream_max_tokens,
        "review_max_tokens": generation.review_max_tokens,
        "decision_max_tokens": generation.decision_max_tokens,
        "scenarios_sha256": scenarios_sha,
        "policy_sha256": POLICY_SHA256,
        "prompts_sha256": current_prompts,
    }
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        differing = sorted(k for k in run_config if saved.get(k) != run_config[k])
        if differing:
            raise SystemExit(f"Run {run_name} exists with different settings ({differing}). Use a new --run-name.")

    records_path = run_dir / "records.jsonl"
    upstream_path = run_dir / "upstream.jsonl"
    done = {(r["scenario_id"], r["condition"], r["rep"]) for r in read_jsonl(records_path)}
    upstream_cache = {(r["scenario_id"], r["rep"]): r for r in read_jsonl(upstream_path)}
    jobs = build_jobs(scenarios, conditions, reps, seed)
    pending = [job for job in jobs if (job[1].scenario_id, job[2], job[0]) not in done]
    print(f"Run {run_name}: {len(jobs)} replays, {len(jobs) - len(pending)} already complete, {len(pending)} to run.")

    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        started = time.perf_counter()
        backend = make_backend(backend_name, model_id, offline=settings["offline"], run_dir=run_dir)
    except BackendUnavailable as error:
        print(f"Backend unavailable: {error}", file=sys.stderr)
        return 1
    print(f"Loaded {backend.model_id} (revision {backend.revision}) in {time.perf_counter() - started:.1f}s")

    if not config_path.exists():
        environment = {"python": platform.python_version(), "machine": platform.machine(), "versions": backend.versions, "revision": backend.revision}
        config_path.write_text(json.dumps({**run_config, "environment": environment}, indent=2, sort_keys=True) + "\n")

    prompts = load_prompts()

    def run_job(rep: int, scenario: Scenario, condition_key: str) -> dict[str, Any]:
        upstream_info = None
        if mode == "controlled":
            reports = controlled_reports(scenario.context.to_dict(), CONDITIONS[condition_key])
        else:
            # Generated once per scenario and repetition, then shared across conditions.
            cached = upstream_cache.get((scenario.scenario_id, rep))
            if cached is None:
                result = generate_upstream(scenario, prompts, generation, backend, stable_seed(seed, "upstream", scenario.scenario_id, rep))
                if getattr(backend, "deferred", False):
                    # Reports built from placeholders must never be stored or shown to Agent D.
                    raise PendingCall("upstream")
                cached = {"scenario_id": scenario.scenario_id, "rep": rep, "is_mock": backend.is_mock, "model": backend.model_id, **result}
                append_jsonl(upstream_path, cached)
                upstream_cache[(scenario.scenario_id, rep)] = cached
            reports = UpstreamReports.from_dict(cached["reports"])
            upstream_info = cached["grades"]

        replay_seed = stable_seed(seed, scenario.scenario_id, condition_key, rep)
        # No condition in this seed: every staged arm gets the same first decision.
        initial_seed = stable_seed(seed, "initial", scenario.scenario_id, rep)
        outcome = run_replay(scenario, CONDITIONS[condition_key], reports, prompts, generation, backend, replay_seed, initial_seed=initial_seed)
        return {
            "run_name": run_name,
            "backend": backend.name,
            "is_mock": backend.is_mock,
            "model": backend.model_id,
            "mode": mode,
            "report_source": mode,
            "condition": condition_key,
            "rep": rep,
            "scenario_id": scenario.scenario_id,
            "pair_id": scenario.pair_id,
            "split": scenario.split,
            "authorization_state": scenario.authorization_state,
            "manipulation": scenario.manipulation,
            "violated_requirement": scenario.violated_requirement,
            "ground_truth_decision": scenario.ground_truth.decision,
            "seed": replay_seed,
            "initial_seed": initial_seed if CONDITIONS[condition_key].staged else None,
            "temperature": temperature,
            "reports": reports.to_dict(),
            "upstream_grades": upstream_info,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **outcome,
        }

    consecutive_errors = 0
    waiting = 0
    for number, (rep, scenario, condition_key) in enumerate(pending, start=1):
        if hasattr(backend, "deferred"):
            backend.deferred = False
        try:
            record = run_job(rep, scenario, condition_key)
        except PendingCall:
            waiting += 1
            continue
        if getattr(backend, "deferred", False):
            # Relay backend: some reply in this replay was a placeholder, so it waits for the next pass.
            waiting += 1
            continue
        if PLACEHOLDER_PREFIX in json.dumps(record):
            raise SystemExit(f"A placeholder reached the record for {scenario.scenario_id} {condition_key}; refusing to write it.")
        append_jsonl(records_path, record)
        print(f"[{number}/{len(pending)}] {scenario.scenario_id} {condition_key} rep={rep} -> {record['outcome']}", flush=True)
        consecutive_errors = consecutive_errors + 1 if record["outcome"] == OUTCOME_MODEL_ERROR else 0
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(f"Stopping after {MAX_CONSECUTIVE_ERRORS} model errors in a row: {record['model_error']}", file=sys.stderr)
            return 1

    if hasattr(backend, "save_pending"):
        backend.save_pending()
        if waiting:
            print(f"\n{waiting} replays wait on {len(backend.pending)} queued Bedrock calls. Next: scripts/04_bedrock_relay.py script --run-name {run_name}")
        else:
            print("\nNo Bedrock calls pending: the run is complete.")
    print("\n" + format_outcome_table(read_jsonl(records_path)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
