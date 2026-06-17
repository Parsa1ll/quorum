"""The Sample record, kept dependency-free.

This lives apart from model.py (which imports mlx) so the offline path (the
cache and every scoring/analysis script) can read generations without pulling
in the Apple-Silicon GPU stack. Generation is the only step that needs mlx.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Sample:
    text: str
    gen_tokens: int
    truncated: bool          # True if we hit max_tokens before the model stopped
