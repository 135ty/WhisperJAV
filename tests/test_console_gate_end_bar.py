"""Regression tests for ConsoleGate.end_bar pipe-mode final records.

Bug (user report 2026-07): when "Semantic scene detection complete:" printed,
the GUI's 3rd progress bar (main bar) never auto-finished. Root cause: in pipe
mode a bare ``end_bar()`` emitted NOTHING (``if fields or final_text`` guard),
and the semantic callback's final 1.0 frame could be dropped by the 0.5s pipe
rate limiter — so the GUI never received a terminal record.

These tests pin the fixed behavior without touching heavy imports
(console_gate only imports stdlib).
"""

import json

from whisperjav.utils.console_gate import ConsoleGate, PROGRESS_PREFIX


class _FakePipe:
    """Non-tty in-memory stream so the gate takes the pipe path."""

    def __init__(self):
        self.chunks = []

    def write(self, s):
        self.chunks.append(s)
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False

    def getvalue(self):
        return "".join(self.chunks)


def _make_gate():
    gate = ConsoleGate()
    gate.configure(enabled=True, verbose=False)
    return gate


def _records(gate, n):
    out = gate._out().getvalue()
    recs = []
    for line in out.splitlines():
        if line.startswith(PROGRESS_PREFIX):
            recs.append(json.loads(line[len(PROGRESS_PREFIX):]))
    assert len(recs) == n, f"expected {n} records, got {len(recs)}: {recs}"
    return recs


def test_bare_end_bar_emits_final_record_in_pipe_mode():
    """A bare end_bar() must emit a final main-channel record at pct=100."""
    gate = _make_gate()
    buf = _FakePipe()
    gate._direct_stream = buf
    try:
        gate.end_bar()
        (rec,) = _records(gate, 1)
        assert rec["final"] is True
        assert rec["channel"] == "main"
        assert rec["pct"] == 100.0
    finally:
        gate._direct_stream = None


def test_end_bar_with_fields_emits_them():
    gate = _make_gate()
    buf = _FakePipe()
    gate._direct_stream = buf
    try:
        gate.end_bar("Detecting scenes: complete", pct=100.0)
        (rec,) = _records(gate, 1)
        assert rec["final"] is True
        assert rec["detail"] == "Detecting scenes: complete"
        assert rec["pct"] == 100.0
    finally:
        gate._direct_stream = None


def test_end_bar_final_bypasses_rate_limit():
    """Even right after a regular emit, the final record must go out."""
    import time as _time

    gate = _make_gate()
    buf = _FakePipe()
    gate._direct_stream = buf
    try:
        gate.update_bar("working", pct=42.0)   # sets _last_pipe_emit = now
        assert gate._last_pipe_emit > 0.0
        _time.sleep(0.01)                      # well inside the 0.5s window
        gate.end_bar()                         # must NOT be rate-limited
        recs = _records(gate, 2)
        assert recs[-1]["final"] is True
        assert recs[-1]["pct"] == 100.0
    finally:
        gate._direct_stream = None


def test_end_bar_routes_channel_fields():
    """Channelled end_bar fields must reach the named GUI sub-bar."""
    gate = _make_gate()
    buf = _FakePipe()
    gate._direct_stream = buf
    try:
        gate.end_bar(pct=87.5, channel="asr")
        (rec,) = _records(gate, 1)
        assert rec["channel"] == "asr"
        assert rec["final"] is True
        assert rec["pct"] == 87.5
    finally:
        gate._direct_stream = None


def test_scope_fold_does_not_swallow_final_after_clear_scope():
    """clear_scope() BEFORE end_bar() so the scope fold cannot eat the final.

    This pins the qwen_pipeline Phase 3/4 ordering: end_bar-then-clear_scope
    emitted nothing in pipe mode (update_bar folded the record into the
    active scope and returned early).
    """
    gate = _make_gate()
    buf = _FakePipe()
    gate._direct_stream = buf
    try:
        gate.update_scope(3, 5, "Enhancing")
        # Fold active: a channel-less pipe record is swallowed into scope.
        gate.update_bar("inner frame", pct=10.0)
        recs = _records(gate, 1)  # only the scope record so far
        assert recs[0]["scene"] == 3
        gate.clear_scope()
        gate.end_bar("Enhancing: complete", pct=100.0)
        recs = _records(gate, 2)
        assert recs[-1]["final"] is True
        assert recs[-1]["pct"] == 100.0
    finally:
        gate._direct_stream = None

