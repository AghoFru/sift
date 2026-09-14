"""Check CLI and HTTP contracts before changes to the embedded application API."""

import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("SIFT_BIN", ROOT / "target/release/sift"))
MODEL = os.environ.get("SIFT_MODEL")
CORPUS = [
    {
        "id": "cat",
        "text": "A domestic cat sleeps on the windowsill.",
        "kind": "feline",
        "price": 2,
    },
    {
        "id": "kitten",
        "text": "A playful kitten chases string.",
        "kind": "feline",
        "price": 3,
    },
    {"id": "dog", "text": "A dog plays in the park.", "kind": "canine", "price": 4},
]


def command(*arguments):
    result = subprocess.run(
        [str(BINARY), *map(str, arguments)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout


def post(base, endpoint, body):
    req = Request(
        base + endpoint,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urlopen(req, timeout=60) as response:
        return json.load(response)


@contextmanager
def server(artifacts):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with (artifacts / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                str(BINARY),
                "serve",
                "--artifacts",
                str(artifacts),
                "--bind",
                f"127.0.0.1:{port}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("The test server exited before it became ready.")
                try:
                    with urlopen(base + "/readyz", timeout=0.2):
                        break
                except (URLError, TimeoutError):
                    time.sleep(0.05)
            else:
                raise RuntimeError("The test server exceeded the startup budget.")
            yield base
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


class ProductContract(unittest.TestCase):
    def setUp(self):
        if not MODEL:
            self.fail("Set SIFT_MODEL to a local model directory.")
        if not BINARY.is_file():
            self.fail(f"Build the Sift binary first: {BINARY}")
        work = ROOT / "target" / "product-contract"
        work.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=work)
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.corpus = self.work / "corpus.jsonl"
        self.corpus.write_text("".join(json.dumps(row) + "\n" for row in CORPUS))
        self.index = self.work / "docs.sift"

    def build(self, operation="build"):
        destination = "--out" if operation == "build" else "--index"
        return command(
            operation,
            "--input",
            self.corpus,
            destination,
            self.index,
            "--model",
            MODEL,
            "--stop-df",
            "1.0",
            "--positions",
            "--rank-fields",
            "price",
        )

    def test_cli_and_single_segment_http(self):
        self.build()
        result = command("search", self.index, "cat", "--semantic-weight", "0")
        self.assertIn("# 1 hits", result)
        self.assertIn("cat", result)
        with server(self.work) as base:
            query = {"index": "docs", "q": "cat", "blend_alpha": 0, "cache": False}
            response = post(base, "/search", {**query, "highlight": True})
            self.assertEqual([hit["doc_id"] for hit in response["hits"]], ["cat"])
            self.assertEqual(response["total"], 1)
            self.assertEqual(response["hits"][0]["payload"], CORPUS[0])
            self.assertIn("<mark>cat</mark>", response["hits"][0]["snippet_html"])
            self.assertIn("latency_us", response)
            phrase = post(base, "/search", {**query, "q": '"domestic cat"'})
            self.assertEqual([hit["doc_id"] for hit in phrase["hits"]], ["cat"])
            filtered = post(
                base,
                "/search",
                {**query, "filter": [{"field": "kind", "eq": "canine"}]},
            )
            self.assertEqual(filtered["hits"], [])
            page = post(base, "/search", {**query, "offset": 1})
            self.assertEqual(page["hits"], [])
            self.assertEqual(page["total"], 1)
            facets = post(base, "/search", {**query, "facets": ["price"]})
            # The existing facet API counts expanded matches independently of blend_alpha.
            self.assertEqual(
                facets["facets"]["price"],
                [
                    {"value": 2.0, "count": 1},
                    {"value": 3.0, "count": 1},
                ],
            )
            with self.assertRaises(HTTPError) as error:
                post(base, "/search", {**query, "blend_alpha": 2})
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
            for invalid in [
                {"index": "../escape", "docs": [{"id": "x", "text": "cat"}]},
                {"index": "docs", "docs": [{"id": True, "text": "cat"}]},
                {"index": "docs", "docs": [{"id": "a\nb", "text": "cat"}]},
            ]:
                with self.assertRaises(HTTPError) as invalid_error:
                    post(base, "/add", invalid)
                self.assertEqual(invalid_error.exception.code, 400)
                invalid_error.exception.close()

    def test_cli_updates_deletes_and_compaction(self):
        self.build("add")
        command("delete", "--index", self.index, "--id", "cat")
        response = command("search", self.index, "cat", "--semantic-weight", "0")
        self.assertIn("# 0 hits", response)
        command("compact", "--index", self.index)
        response = command("search", self.index, "cat", "--semantic-weight", "0")
        self.assertIn("# 0 hits", response)
        self.assertIn(
            "# 1 hits", command("search", self.index, "dog", "--semantic-weight", "0")
        )

    def test_http_writes_survive_reopen(self):
        self.build("add")
        with server(self.work) as base:
            added = post(
                base,
                "/add",
                {
                    "index": "docs",
                    "docs": [
                        {"id": "bird", "text": "A sparrow flies over a tree."},
                    ],
                },
            )
            self.assertEqual(added["affected"], 1)
            query = {"index": "docs", "q": "sparrow", "blend_alpha": 0, "cache": False}
            self.assertEqual(post(base, "/search", query)["hits"][0]["doc_id"], "bird")
            post(
                base,
                "/add",
                {
                    "index": "docs",
                    "upsert": True,
                    "docs": [
                        {"id": "bird", "text": "An eagle flies over a tree."},
                    ],
                },
            )
            self.assertEqual(post(base, "/search", query)["hits"], [])
            deleted = post(base, "/delete", {"index": "docs", "ids": ["dog"]})
            self.assertEqual(deleted["affected"], 1)
        with server(self.work) as base:
            query = {"index": "docs", "q": "dog", "blend_alpha": 0, "cache": False}
            self.assertEqual(post(base, "/search", query)["hits"], [])
            query["q"] = "eagle"
            self.assertEqual(post(base, "/search", query)["hits"][0]["doc_id"], "bird")


if __name__ == "__main__":
    unittest.main()
