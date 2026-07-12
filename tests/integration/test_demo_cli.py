"""`panoptes demo` end-to-end: the zero-download showcase must produce a
valid results.json (tracks + line crossings) and a non-empty annotated
video (ARCHITECTURE.md "Testing strategy": "Integration: `panoptes demo`
end-to-end produces TRACK_FINISHED + LINE_CROSSED events")."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from typer.testing import CliRunner

runner = CliRunner()

_DEMO_FRAMES = 150  # enough for cars to finish + the truck to dwell


def _combined_output(result) -> str:
    text = result.output
    with contextlib.suppress(ValueError, AttributeError):
        text += result.stderr
    return text


def test_demo_produces_results_and_annotated_video(tmp_path: Path) -> None:
    # ARCHITECTURE.md "CLI": demo = synthetic video + mock detector,
    # no downloads; works headless.
    from panoptes.cli import app

    out_dir = tmp_path / "demo-out"
    result = runner.invoke(
        app, ["demo", "--frames", str(_DEMO_FRAMES), "--output-dir", str(out_dir)]
    )
    assert result.exit_code == 0, _combined_output(result)

    results_path = out_dir / "results.json"
    assert results_path.exists(), "demo must write results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))

    assert len(results["tracks"]) >= 1
    assert results["summary"]["n_tracks"] >= 1

    types = {e["type"] for e in results["events"]}
    assert "track_finished" in types
    assert "line_crossed" in types

    annotated = out_dir / "annotated.mp4"
    assert annotated.exists(), "demo must write the annotated video"
    assert annotated.stat().st_size > 0
