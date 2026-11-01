#!/usr/bin/env python3
"""Measure the PCM speech API using an explicit, replayable workload (stdlib only).

Every request, including warm-up, rejection and transport failure, is recorded. TTFA
means client dispatch to the first complete PCM sample, not response headers. This
closed-loop harness measures localhost/HTTP serving, not native engine time or WER.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import platform
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

BYTES_PER_SECOND = 24000 * 2  # OpenAI PCM: mono signed 16-bit little-endian, 24 kHz


def load_workload(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("workload is empty")
    seen = set()
    measured = False
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each workload row must be an object")
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("request ids must be nonempty, unique strings")
        seen.add(row["id"])
        if not isinstance(row.get("input"), str) or not row["input"].strip():
            raise ValueError("every request needs input text")
        if not isinstance(row.get("voice"), str) or not row["voice"]:
            raise ValueError("every request needs a voice")
        phase = row.setdefault("phase", "measured")
        if phase not in ("warmup", "measured"):
            raise ValueError("phase must be warmup or measured")
        if phase == "warmup" and measured:
            raise ValueError("warm-up requests must precede all measured requests")
        measured |= phase == "measured"
    if not measured:
        raise ValueError("workload has no measured requests")
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def distribution(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": sum(values) / len(values) if values else None,
    }


def snapshot(base_url: str, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(base_url + "/v1/audio/cache", timeout=timeout) as response:
            return {"available": True, "value": json.load(response)}
    except (OSError, ValueError) as error:
        return {"available": False, "error": str(error)}


def request_one(base_url: str, row: dict, timeout: float, origin: float) -> dict:
    payload = json.dumps({"input": row["input"], "voice": row["voice"], "response_format": "pcm"}).encode()
    request = urllib.request.Request(
        base_url + "/v1/audio/speech", data=payload, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    result = {
        "id": row["id"], "phase": row["phase"], "voice": row["voice"],
        "input_sha256": hashlib.sha256(row["input"].encode()).hexdigest(),
        "start_s": start - origin, "status": None, "outcome": "error",
        "ttfa_ms": None, "pcm_bytes": 0, "audio_s": 0.0,
        "chunk_events": [],
    }
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result["status"] = response.status
            result["content_type"] = response.headers.get("Content-Type", "")
            if result["content_type"].split(";")[0].strip() != "audio/pcm":
                raise ValueError("expected audio/pcm, refusing to count non-PCM bytes as audio")
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                elapsed = time.perf_counter() - start
                result["pcm_bytes"] += len(chunk)
                digest.update(chunk)
                result["chunk_events"].append({"elapsed_s": elapsed, "bytes": len(chunk)})
                if result["ttfa_ms"] is None and result["pcm_bytes"] >= 2:
                    result["ttfa_ms"] = elapsed * 1000
            if result["pcm_bytes"] == 0 or result["pcm_bytes"] % 2:
                raise ValueError("empty or incomplete 16-bit PCM samples")
            result["outcome"] = "completed"
    except urllib.error.HTTPError as error:
        result["status"] = error.code
        result["outcome"] = "rejected" if error.code == 429 else "error"
        result["error"] = str(error)
        error.close()
    except (OSError, ValueError, EOFError, http.client.HTTPException) as error:
        result["error"] = str(error)
    end = time.perf_counter()
    result["end_s"] = end - origin
    result["total_ms"] = (end - start) * 1000
    result["audio_s"] = result["pcm_bytes"] / BYTES_PER_SECOND
    result["pcm_sha256"] = digest.hexdigest()
    return result


def summarize(rows: list[dict], wall_s: float, ttfa_slo_ms: float) -> dict:
    completed = [r for r in rows if r["outcome"] == "completed"]
    audio_s = sum(r["audio_s"] for r in completed)
    return {
        "requests": len(rows), "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "scenario_wall_s": wall_s, "completed_audio_s": audio_s,
        "audio_seconds_per_wall_second": audio_s / wall_s if wall_s else None,
        "completed_requests_per_second": len(completed) / wall_s if wall_s else None,
        "client_ttfa_ms_completed": distribution([r["ttfa_ms"] for r in completed]),
        "client_total_ms_completed": distribution([r["total_ms"] for r in completed]),
        "rtf_completed": distribution([r["total_ms"] / 1000 / r["audio_s"] for r in completed]),
        "ttfa_slo_ms": ttfa_slo_ms,
        "completed_within_ttfa_slo": sum(r["ttfa_ms"] <= ttfa_slo_ms for r in completed),
        "percentile_method": "nearest_rank",
        "quality_evaluated": False,
    }


def run(args: argparse.Namespace) -> dict:
    rows = load_workload(args.workload)
    args.output.mkdir(parents=True, exist_ok=False)  # never silently overwrite evidence
    raw_path = args.output / "requests.jsonl"
    origin = time.perf_counter()
    warmup = [r for r in rows if r["phase"] == "warmup"]
    measured = [r for r in rows if r["phase"] == "measured"]
    before = snapshot(args.base_url, args.timeout)
    with raw_path.open("w") as raw:
        # Serial warm-up gives a defined voice/cache order, even for concurrent trials.
        for row in warmup:
            result = request_one(args.base_url, row, args.timeout, origin)
            raw.write(json.dumps(result) + "\n")
            raw.flush()
            if result["outcome"] != "completed":
                raise RuntimeError("warm-up failed; raw record retained, no measured trial reported")
        after_warmup = snapshot(args.base_url, args.timeout)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            start = time.perf_counter()
            # At most concurrency requests are dispatched at once. This is closed-loop;
            # it intentionally does not claim a controlled open-loop arrival rate.
            pending = {}
            iterator = iter(measured)
            for _ in range(min(args.concurrency, len(measured))):
                row = next(iterator)
                pending[pool.submit(request_one, args.base_url, row, args.timeout, origin)] = row
            while pending:
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    result = future.result()
                    results.append(result)
                    raw.write(json.dumps(result) + "\n")
                    raw.flush()
                    row = next(iterator, None)
                    if row is not None:
                        pending[pool.submit(request_one, args.base_url, row, args.timeout, origin)] = row
            wall_s = time.perf_counter() - start
    report = {
        "schema_version": 1, "label": args.label, "server_revision": args.server_revision,
        "server_binary_sha256": hashlib.sha256(args.server_binary.read_bytes()).hexdigest() if args.server_binary else None,
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "client_platform": platform.platform(), "python": platform.python_version(),
        "base_url": args.base_url, "load_mode": "closed_loop", "concurrency": args.concurrency,
        "warmup_requests": len(warmup), "timeout_s": args.timeout,
        "cache_before": before, "cache_after_warmup": after_warmup,
        "cache_after": snapshot(args.base_url, args.timeout),
        "summary": summarize(results, wall_s, args.ttfa_slo_ms),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8880")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--server-revision", required=True, help="exact revision of the running server")
    parser.add_argument("--server-binary", type=Path, help="optional local binary to hash")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--ttfa-slo-ms", type=float, default=100)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if args.concurrency < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("concurrency and timeout must be positive")
    if not math.isfinite(args.ttfa_slo_ms) or args.ttfa_slo_ms <= 0:
        parser.error("TTFA SLO must be positive and finite")
    report = run(args)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["outcomes"].get("completed", 0) == report["summary"]["requests"] else 1


if __name__ == "__main__":
    sys.exit(main())
