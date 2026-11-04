import json

import pytest

from diffusionlab.visual import save


def test_report_escapes_embedded_script_and_roundtrips_json(tmp_path):
    payload = {"quote": "</script><script>unexpected()</script>", "score": 0.8}
    save(tmp_path, payload, "<Example>", "Synthetic", "<p>Report</p>")
    page = (tmp_path / "demo.html").read_text()
    assert page.count("</script>") == 1
    assert "&lt;Example&gt;" in page
    assert json.loads((tmp_path / "metrics.json").read_text()) == payload


def test_nonfinite_report_metrics_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        save(tmp_path, {"metric": float("nan")}, "Lab", "Synthetic", "")
