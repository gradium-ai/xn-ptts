"""Offline checks for streaming timing, partial responses, and throughput accounting."""

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

spec = importlib.util.spec_from_file_location("bench_serving", Path(__file__).parents[1] / "bench-serving.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        mode = payload["input"]
        if mode == "busy":
            self.send_response(429)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if mode == "json" else "audio/pcm")
        if mode == "truncated":
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if mode == "stream":
            self.wfile.flush()
            time.sleep(0.08)  # headers arrive before the first playable sample
            self.wfile.write(b"\x01")
            self.wfile.flush()
            time.sleep(0.08)
            self.wfile.write(b"\x00\x02\x00")
        elif mode == "truncated":
            self.wfile.write(b"4\r\n\x01\x00")  # advertised chunk is incomplete
        elif mode == "odd":
            self.wfile.write(b"\x01")
        elif mode == "json":
            self.wfile.write(b"{}")
        self.wfile.flush()


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def request(self, text):
        return bench.request_one(
            self.url, {"id": text, "phase": "measured", "voice": "default", "input": text},
            2, time.perf_counter(),
        )

    def test_first_audio_requires_a_complete_sample_after_headers(self):
        row = self.request("stream")
        self.assertEqual(row["outcome"], "completed")
        self.assertEqual(row["pcm_bytes"], 4)
        self.assertGreaterEqual(row["ttfa_ms"], 130)
        self.assertAlmostEqual(row["audio_s"], 4 / 48000)

    def test_rejections_and_invalid_or_truncated_bodies_remain_failures(self):
        self.assertEqual(self.request("busy")["outcome"], "rejected")
        for mode in ["odd", "empty", "json", "truncated"]:
            row = self.request(mode)
            self.assertEqual(row["outcome"], "error", mode)
            self.assertIn("error", row)

    def test_aggregate_throughput_uses_wall_time_and_excludes_failed_partial_audio(self):
        completed = {"outcome": "completed", "audio_s": 10, "total_ms": 1000, "ttfa_ms": 20}
        failed = {"outcome": "error", "audio_s": 99, "total_ms": 100, "ttfa_ms": 1}
        summary = bench.summarize([completed, completed, failed], 2, 100)
        self.assertEqual(summary["audio_seconds_per_wall_second"], 10)
        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["client_ttfa_ms_completed"]["n"], 2)
        self.assertEqual(summary["completed_within_ttfa_slo"], 2)
        self.assertEqual(bench.distribution([])["p99"], None)

    def test_workload_rejects_duplicate_ids_and_late_warmup(self):
        row = {"id": "a", "voice": "default", "input": "Hello"}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.jsonl"
            for rows in [[row, row], [row, dict(row, id="b", phase="warmup")]]:
                path.write_text("\n".join(json.dumps(r) for r in rows))
                with self.assertRaises(ValueError):
                    bench.load_workload(path)


if __name__ == "__main__":
    unittest.main()
