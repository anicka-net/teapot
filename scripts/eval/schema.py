#!/usr/bin/env python3
"""Eval report schema — the stable contract (v1).

This file defines the eval report format that all test suites produce
and all consumers (SBOM, model cards, CI gates) read. Change this
interface only with a version bump.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SuiteResult:
    """Result from a single test suite."""
    name: str
    status: str  # "pass" | "fail" | "error" | "skip"
    passed: int
    total: int
    threshold: str = ""
    duration_seconds: float = 0.0
    details: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class EvalReport:
    """Top-level evaluation report."""
    version: str = "1"
    model: dict = field(default_factory=dict)
    timestamp: str = ""
    tier: str = "fast"
    duration_seconds: float = 0.0
    suites: list = field(default_factory=list)
    verdict: str = "error"
    notes: list = field(default_factory=list)

    def add_suite(self, result: SuiteResult):
        self.suites.append(asdict(result))

    def add_note(self, note: str):
        self.notes.append(note)

    def compute_verdict(self):
        """pass iff all non-skip suites pass."""
        active = [s for s in self.suites if s["status"] != "skip"]
        if not active:
            self.verdict = "error"
            self.add_note("No active suites ran")
        elif all(s["status"] == "pass" for s in active):
            self.verdict = "pass"
        elif any(s["status"] == "error" for s in active):
            self.verdict = "error"
        else:
            self.verdict = "fail"

    def to_dict(self) -> dict:
        return asdict(self)
