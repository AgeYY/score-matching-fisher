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


@dataclass(frozen=True)
class _SvgColumnLayout:
    source: _SvgSource
    x: float
    y: float
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


def svg_viewbox_size(path: str | Path) -> tuple[float, float]:
    """Return the effective SVG width and height in viewBox/user units."""
    src = _read_svg_source(path)
    return src.width, src.height


def _svg_layout(
    sources: list[_SvgSource],
    *,
    spacing: float,
    target_height: float | None,
    valign: str,
) -> tuple[list[_SvgColumnLayout], float, float]:
    if target_height is not None and (not math.isfinite(float(target_height)) or float(target_height) <= 0.0):
        raise ValueError("target_height must be a finite positive value")
    if valign not in {"top", "center", "bottom"}:
        raise ValueError("valign must be one of 'top', 'center', or 'bottom'")

    total_height = float(target_height) if target_height is not None else max(src.height for src in sources)
    columns: list[_SvgColumnLayout] = []
    x_offset = 0.0
    for src in sources:
        if target_height is None:
            col_width = src.width
            col_height = src.height
            y_offset = 0.0
        else:
            scale = float(target_height) / src.height
            col_width = src.width * scale
            col_height = float(target_height)
            if valign == "top":
                y_offset = 0.0
            elif valign == "bottom":
                y_offset = total_height - col_height
            else:
                y_offset = (total_height - col_height) / 2.0
        columns.append(_SvgColumnLayout(src, x_offset, y_offset, col_width, col_height))
        x_offset += col_width + float(spacing)
    total_width = x_offset - float(spacing)
    return columns, total_width, total_height


def _prefix_copied_svg_ids(root: ET.Element, prefix: str) -> None:
    id_map: dict[str, str] = {}
    for elem in root.iter():
        old_id = elem.attrib.get("id")
        if old_id:
            new_id = f"{prefix}{old_id}"
            elem.set("id", new_id)
            id_map[old_id] = new_id
    if not id_map:
        return

    url_ref = re.compile(r"url\(#([^)]+)\)")
    hash_ref = re.compile(r"^#(.+)$")
    for elem in root.iter():
        for key, value in list(elem.attrib.items()):
            value = url_ref.sub(lambda m: f"url(#{id_map.get(m.group(1), m.group(1))})", value)
            value = hash_ref.sub(lambda m: f"#{id_map.get(m.group(1), m.group(1))}", value)
            elem.set(key, value)


def concatenate_svgs_horizontally(
    source_paths: list[str | Path],
    out_path: str | Path,
    *,
    spacing: float = 24.0,
    target_height: float | None = None,
    valign: str = "center",
) -> str:
    """Write one SVG with each source SVG as a left-to-right vector column."""
    if not source_paths:
        raise ValueError("source_paths must contain at least one SVG")
    if spacing < 0.0 or not math.isfinite(float(spacing)):
        raise ValueError("spacing must be a finite non-negative value")

    sources = [_read_svg_source(p) for p in source_paths]
    columns, total_width, total_height = _svg_layout(
        sources,
        spacing=float(spacing),
        target_height=target_height,
        valign=valign,
    )

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

    for idx, col_layout in enumerate(columns):
        src = col_layout.source
        col = ET.SubElement(
            out_root,
            f"{{{_SVG_NS}}}svg",
            {
                "x": f"{col_layout.x:g}",
                "y": f"{col_layout.y:g}",
                "width": f"{col_layout.width:g}",
                "height": f"{col_layout.height:g}",
                "viewBox": f"{src.min_x:g} {src.min_y:g} {src.width:g} {src.height:g}",
            },
        )
        copied_children = [copy.deepcopy(child) for child in list(src.root)]
        tmp_root = ET.Element(f"{{{_SVG_NS}}}g")
        for child in copied_children:
            tmp_root.append(child)
        _prefix_copied_svg_ids(tmp_root, f"svg{idx}_")
        for child in copied_children:
            col.append(child)

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
    target_height: float | None = None,
    valign: str = "center",
) -> str:
    """Compose SVG columns and rasterize the combined layout to PNG."""
    if not source_paths:
        raise ValueError("source_paths must contain at least one SVG")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    sources = [_read_svg_source(p) for p in source_paths]
    _, total_width, total_height = _svg_layout(
        sources,
        spacing=float(spacing),
        target_height=target_height,
        valign=valign,
    )
    pixel_width = max(1, int(round(total_width / 72.0 * float(dpi))))
    pixel_height = max(1, int(round(total_height / 72.0 * float(dpi))))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="svg_columns_") as td:
        tmp_svg = Path(td) / "combined.svg"
        tmp_png = out.with_name(f".{out.name}.tmp")
        concatenate_svgs_horizontally(
            source_paths,
            tmp_svg,
            spacing=spacing,
            target_height=target_height,
            valign=valign,
        )
        try:
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
                    str(tmp_png),
                    str(tmp_svg),
                ],
                check=True,
            )
            _verify_png(tmp_png)
            tmp_png.replace(out)
        except Exception:
            if tmp_png.exists():
                tmp_png.unlink()
            if out.exists() and out.stat().st_size == 0:
                out.unlink()
            raise
    return str(out)


def _verify_png(path: Path) -> None:
    from PIL import Image

    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"PNG was not written or is empty: {path}")
    with Image.open(path) as im:
        if im.format != "PNG":
            raise ValueError(f"Expected PNG output, got {im.format!r}: {path}")
        im.verify()


def _resample_lanczos() -> int:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    return int(resampling.LANCZOS)


def concatenate_pngs_horizontally(
    source_paths: list[str | Path],
    out_path: str | Path,
    *,
    spacing: int = 100,
    target_height: int | None = None,
    valign: str = "center",
    background: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> str:
    """Write one valid PNG with source images as left-to-right columns."""
    if not source_paths:
        raise ValueError("source_paths must contain at least one PNG")
    if spacing < 0:
        raise ValueError("spacing must be non-negative")
    if target_height is not None and int(target_height) <= 0:
        raise ValueError("target_height must be positive")
    if valign not in {"top", "center", "bottom"}:
        raise ValueError("valign must be one of 'top', 'center', or 'bottom'")

    from PIL import Image

    images = []
    try:
        for path in source_paths:
            img = Image.open(path).convert("RGBA")
            if img.width <= 0 or img.height <= 0:
                raise ValueError(f"PNG has invalid dimensions: {path}")
            if target_height is not None and img.height != int(target_height):
                width = max(1, int(round(img.width * (int(target_height) / img.height))))
                img = img.resize((width, int(target_height)), resample=_resample_lanczos())
            images.append(img)

        total_height = int(target_height) if target_height is not None else max(img.height for img in images)
        total_width = sum(img.width for img in images) + int(spacing) * (len(images) - 1)
        canvas = Image.new("RGBA", (total_width, total_height), background)
        x_offset = 0
        for img in images:
            if valign == "top":
                y_offset = 0
            elif valign == "bottom":
                y_offset = total_height - img.height
            else:
                y_offset = (total_height - img.height) // 2
            canvas.alpha_composite(img, (x_offset, y_offset))
            x_offset += img.width + int(spacing)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_png = out.with_name(f".{out.name}.tmp")
        try:
            canvas.convert("RGB").save(tmp_png, format="PNG")
            _verify_png(tmp_png)
            tmp_png.replace(out)
        except Exception:
            if tmp_png.exists():
                tmp_png.unlink()
            if out.exists() and out.stat().st_size == 0:
                out.unlink()
            raise
        return str(out)
    finally:
        for img in images:
            img.close()
