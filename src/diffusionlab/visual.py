"""Dependency-free, standalone HTML reports; structured data are embedded."""

import html
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def save(out, payload, title, description, body, script=""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "created": "2026-11-04",
        "synthetic": True,
    }
    for dependency in ["torch", "scikit-learn", "scipy"]:
        try:
            environment[dependency] = version(dependency)
        except PackageNotFoundError:
            pass
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    css = """body{margin:0;background:#0a1324;color:#e9f0ff;font:16px/1.6 system-ui}
    main{max-width:1080px;margin:48px auto;padding:0 24px}h1{font-size:44px;line-height:1.15}
    h2{font-size:22px}p{max-width:850px;color:#b6c8e3}a{color:#66e4cc}button,select,input{
    background:#182b48;color:#fff;border:1px solid #4c6c94;border-radius:6px;padding:10px}
    canvas,svg{background:#101f36;border-radius:12px;max-width:100%}.card{padding:22px;
    border:1px solid #29425f;border-radius:12px;margin:20px 0}table{border-collapse:collapse;
    width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #29425f}
    .tag{color:#66e4cc}code{overflow-wrap:anywhere}.muted{font-size:13px}label{margin-right:20px}
    @media(max-width:600px){h1{font-size:32px}main{padding:0 16px}}"""
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    page = (
        f'<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{css}</style><main>"
        '<div class="tag">SYNTHETIC ML LAB · MEASURED RUN</div>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p>"
        + body
        + '<p class="muted">New AI-assisted personal work · November 2026 · '
        "Synthetic data only. This report replays an executed experiment; "
        "see the README for assumptions and reproduction.</p></main>"
        "<script>const DATA=" + data + ";" + script + "</script></html>"
    )
    (out / "demo.html").write_text(page)


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def rows(records, keys):
    return "".join(
        "<tr>" + "".join(f"<td>{html.escape(display(r[k]))}</td>" for k in keys) + "</tr>"
        for r in records
    )


def table(records, keys):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(k)}</th>" for k in keys)
        + "</tr></thead><tbody>"
        + rows(records, keys)
        + "</tbody></table>"
    )


def scatter_svg(series, title, limit=3.0):
    colors = ["#66e4cc", "#ffb86c", "#bc9bff"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 440" '
        'role="img" aria-label="' + html.escape(title) + '">',
        '<rect width="720" height="440" fill="#101f36"/>',
        '<text x="30" y="30" fill="#e9f0ff" font-family="sans-serif" '
        'font-size="20">' + html.escape(title) + "</text>",
    ]
    for j, (name, points) in enumerate(series):
        parts.append(
            f'<text x="{30 + j * 220}" y="420" fill="{colors[j % 3]}" '
            f'font-family="sans-serif">{html.escape(name)}</text>'
        )
        for x, y in np.asarray(points):
            px = 360 + 300 * np.clip(x / limit, -1, 1)
            py = 225 - 165 * np.clip(y / limit, -1, 1)
            parts.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.2" fill="{colors[j % 3]}" opacity=".6"/>'
            )
    return "".join(parts) + "</svg>"
