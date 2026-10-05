#!/usr/bin/env python3
"""The GUI hides a transient sub-bar when the CLI retracts it.

Companion to ``tests/test_console_gate_end_bar.py`` (the CLI half). Owner
report 2026-10: in a multi-file run the "ASR Text Gen" bar stayed on screen,
frozen at its last frame, after that file's text generation had finished and
the run had moved on to the next file. The CLI now sends
``{"channel": "asr", "retract": true}`` when the phase ends; ``app.js`` must
hide that one row on receipt — and only that row, since "files" / "translate"
settle per file with ``final`` and must stay visible.

``app.js`` is shipped as a static asset (no bundler), so these assertions read
it the way the other frontend tests do. The ProgressManager / routing sources
are sliced out first, so an unrelated ``retract`` elsewhere cannot satisfy
them.
"""

import re
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parent.parent / "whisperjav" / "webview_gui" / "assets" / "app.js"


@pytest.fixture(scope="module")
def js():
    return APP_JS.read_text(encoding="utf-8")


def _slice(js, start_marker, end_marker):
    start = js.index(start_marker)
    end = js.index(end_marker, start)
    return js[start:end]


@pytest.fixture(scope="module")
def progress_manager(js):
    """The ProgressManager object literal, _auxBars through hideAuxBars."""
    return _slice(js, "const ProgressManager = {", "\n    reset() {")


@pytest.fixture(scope="module")
def routing(js):
    """_handleProgressRecord (where PROGRESS records are routed by channel)."""
    return _slice(js, "_handleProgressRecord(payload) {", "\n    // Reduce a CLI bar line")


def test_asr_route_forwards_a_retract_record(js, routing):
    """The retract record has no detail, so the old `if (p.detail)` gate
    dropped it and the row was never hidden."""
    assert "p.retract === true" in routing
    # Forwarded when there is a detail OR when it is a retract.
    assert re.search(r"if \(d \|\| p\.retract === true\)", routing), routing


def test_setaux_hides_the_bar_on_retract(progress_manager):
    """setAux must hide (and stop early) BEFORE it un-hides the row."""
    body = _slice(progress_manager, "setAux(channel, p) {", "\n    hideAux(")
    assert "p.retract === true" in body
    assert "this.hideAux(channel)" in body
    retract_at = body.index("p.retract === true")
    show_at = body.index("row.hidden = false")
    assert retract_at < show_at, "the retract check must precede row.hidden = false"


def test_hideaux_clears_a_single_row(progress_manager):
    body = _slice(progress_manager, "hideAux(channel) {", "\n    hideAuxBars() {")
    assert "row.hidden = true" in body
    assert "fill.style.width = '0%'" in body
    assert "value.textContent = '—'" in body


def test_hideauxbars_reuses_hideaux(progress_manager):
    body = progress_manager[progress_manager.index("hideAuxBars() {"):]
    assert "Object.keys(this._auxBars)" in body
    assert "this.hideAux(channel)" in body


def test_the_asr_row_is_still_an_aux_bar(progress_manager):
    """Guard against the retract fix detaching the row it is meant to hide."""
    assert re.search(r"asr:\s*\{[^}]*row: 'asrProgressRow'", progress_manager)
