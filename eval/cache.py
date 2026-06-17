"""Tiny on-disk cache for generations.

Generation is the only slow part, so we never want to redo it. Each sample is
one json line; on startup we read them back into a dict. Append-only, so a run
that dies partway through just resumes from whatever it already wrote.
"""
from __future__ import annotations
import json
from pathlib import Path

from src.sample import Sample


class SampleCache:
    def __init__(self, path):
        self.path = Path(path)
        self.mem = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a half-written trailing line (concurrent run)
                self.mem[r["key"]] = Sample(r["text"], r["gen_tokens"], r["truncated"])

    def get(self, key):
        return self.mem.get(key)

    def put(self, key, s: Sample):
        self.mem[key] = s
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "text": s.text,
                                "gen_tokens": s.gen_tokens,
                                "truncated": s.truncated}) + "\n")

    def __contains__(self, key):
        return key in self.mem
