"""Small SVG composition helpers."""

from __future__ import annotations

import copy
import math
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _SVG_NS)


@dataclass(frozen=True)
class _SvgSource:
    path: Path
    root: ET.Element
    min_x: float
    min_y: float
    width: float
    height: float


def _local_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[-1]
    return tag


def _parse_svg_number(value: str | None) -> float | None:
    if value is None:
        return None
    m = re.match(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", str(value))
    if not m:
        return None
    out = float(m.group(1))
    if not math.isfinite(out) or out <= 0.0:
        return None
    return out


def _read_svg_source(path: str | Path) -> _SvgSource:
    p = Path(path)
    root = ET.parse(p).getroot()
    if _local_name(root.tag) != "svg":
        raise ValueError(f"Expected SVG root in {p}")

    view_box = root.attrib.get("viewBox")
    if view_box:
        vals = [float(x) for x in re.split(r"[\s,]+", view_box.strip()) if x]
        if len(vals) != 4 or vals[2] <= 0.0 or vals[3] <= 0.0:
            raise ValueError(f"Invalid SVG viewBox in {p}: {view_box!r}")
        return _SvgSource(p, root, vals[0], vals[1], vals[2], vals[3])

    width = _parse_svg_number(root.attrib.get("width"))
    height = _parse_svg_number(root.attrib.get("height"))
    if width is None or height is None:
        raise ValueError(f"SVG must define viewBox or positive width/height: {p}")
    return _SvgSource(p, root, 0.0, 0.0, width, height)


def concatenate_svgs_horizontally(
    source_paths: list[str | Path],
    out_path: str | Path,
    *,
    spacing: float = 24.0,
) -> str:
    """Write one SVG with each source SVG as a left-to-right vector column."""
    if not source_paths:
        raise ValueError("source_paths must contain at least one SVG")
    if spacing < 0.0 or not math.isfinite(float(spacing)):
        raise ValueError("spacing must be a finite non-negative value")

    sources = [_read_svg_source(p) for p in source_paths]
    total_width = sum(src.width for src in sources) + float(spacing) * (len(sources) - 1)
    total_height = max(src.height for src in sources)

    out_root = ET.Element(
        f"{{{_SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": f"{total_width:g}",
            "height": f"{total_height:g}",
            "viewBox": f"0 0 {total_width:g} {total_height:g}",
        },
    )
    out_root.append(ET.Comment(" Columns, left-to-right: " + ", ".join(src.path.name for src in sources) + " "))

    x_offset = 0.0
    for src in sources:
        col = ET.SubElement(
            out_root,
            f"{{{_SVG_NS}}}svg",
            {
                "x": f"{x_offset:g}",
                "y": "0",
                "width": f"{src.width:g}",
                "height": f"{src.height:g}",
                "viewBox": f"{src.min_x:g} {src.min_y:g} {src.width:g} {src.height:g}",
            },
        )
        for child in list(src.root):
            col.append(copy.deepcopy(child))
        x_offset += src.width + float(spacing)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(out_root).write(out, encoding="utf-8", xml_declaration=True)
    return str(out)


def concatenate_svgs_horizontally_to_png(
    source_paths: list[str | Path],
    out_path: str | Path,
    *,
    spacing: float = 24.0,
    dpi: int = 300,
) -> str:
    """Compose SVG columns and rasterize the combined layout to PNG."""
    if not source_paths:
        raise ValueError("source_paths must contain at least one SVG")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    sources = [_read_svg_source(p) for p in source_paths]
    total_width = sum(src.width for src in sources) + float(spacing) * (len(sources) - 1)
    total_height = max(src.height for src in sources)
    pixel_width = max(1, int(round(total_width / 72.0 * float(dpi))))
    pixel_height = max(1, int(round(total_height / 72.0 * float(dpi))))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="svg_columns_") as td:
        tmp_svg = Path(td) / "combined.svg"
        concatenate_svgs_horizontally(source_paths, tmp_svg, spacing=spacing)
        subprocess.run(
            [
                "rsvg-convert",
                "-f",
                "png",
                "--dpi-x",
                str(int(dpi)),
                "--dpi-y",
                str(int(dpi)),
                "-w",
                str(pixel_width),
                "-h",
                str(pixel_height),
                "-o",
                str(out),
                str(tmp_svg),
            ],
            check=True,
        )
    return str(out)
