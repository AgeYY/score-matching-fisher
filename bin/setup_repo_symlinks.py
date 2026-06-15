#!/usr/bin/env python3
"""Create repo-root journal/ and report/ links into the sibling notes clone.

These paths are gitignored in the coding repo. Notes and report sources are
versioned only in ``../score-matching-fisher-note/score-matching-fisher-note-repo/``.

On Windows, default Git checkout can leave plain text stubs instead of links.
Run after clone (or whenever links break):

    mamba run -n geo_diffusion python bin/setup_repo_symlinks.py
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RELATIVE_TARGETS = {
    "journal": Path("../score-matching-fisher-note/score-matching-fisher-note-repo/journal"),
    "report": Path("../score-matching-fisher-note/score-matching-fisher-note-repo/report"),
}


def _target_path(name: str) -> Path:
    target = RELATIVE_TARGETS[name]
    if target.is_absolute():
        return target.resolve()
    return (REPO_ROOT / target).resolve()


def _resolved_if_link(path: Path) -> Path | None:
    if not path.exists() and not path.is_symlink():
        return None
    try:
        return path.resolve()
    except OSError:
        return None


def _is_plain_symlink_stub(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return text.startswith("../") or text.startswith("./")


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    if sys.platform == "win32" and path.exists():
        try:
            return bool(path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except OSError:
            return False
    return False


def _remove_existing_link(link_path: Path) -> None:
    if link_path.is_file() and not link_path.is_symlink():
        link_path.unlink()
        return
    if link_path.is_symlink():
        link_path.unlink()
        return
    if link_path.is_dir() and _is_reparse_point(link_path):
        if sys.platform == "win32":
            subprocess.run(
                ["cmd", "/c", "rmdir", str(link_path)],
                check=True,
            )
            return
        link_path.unlink()
        return
    raise SystemExit(
        f"Refusing to remove {link_path}: it is a real directory, not a link. "
        "Move it aside manually if you really intend to replace it."
    )


def _create_dir_link(link_path: Path, target_path: Path) -> None:
    target_path = target_path.resolve()
    if not target_path.is_dir():
        raise SystemExit(
            f"Target missing for {link_path.name}/: {target_path}\n"
            "Clone the sibling notes repo first "
            "(../score-matching-fisher-note/score-matching-fisher-note-repo/)."
        )

    link_path.parent.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        try:
            os.symlink(target_path, link_path, target_is_directory=True)
            return
        except OSError:
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
                check=True,
            )
            return

    relative_target = os.path.relpath(target_path, link_path.parent)
    os.symlink(relative_target, link_path, target_is_directory=True)


def ensure_repo_symlinks() -> list[str]:
    created: list[str] = []
    for name in RELATIVE_TARGETS:
        link_path = REPO_ROOT / name
        target_path = _target_path(name)

        current = _resolved_if_link(link_path)
        if current == target_path:
            print(f"[ok] {name}/ -> {target_path}")
            continue

        if link_path.exists() or link_path.is_symlink():
            if _is_plain_symlink_stub(link_path):
                print(f"[fix] replacing plain-text stub {name}")
            _remove_existing_link(link_path)

        _create_dir_link(link_path, target_path)
        print(f"[link] {name}/ -> {target_path}")
        created.append(name)

    return created


def main() -> None:
    created = ensure_repo_symlinks()
    if created:
        print("\nDone. journal/ and report/ are gitignored in the coding repo.")
    else:
        print("\nAll repo symlinks already point at the notes tree.")


if __name__ == "__main__":
    main()
