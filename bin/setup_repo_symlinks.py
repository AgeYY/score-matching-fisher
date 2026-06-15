#!/usr/bin/env python3
"""Create repo-root journal/ and report/ links into the sibling notes clone.

Git stores these paths as symlinks (mode 120000). On Windows with
``core.symlinks=false`` (the default), checkout leaves plain text files
containing the relative target path. IDEs then open those files instead of
the notes tree. Run this script after clone (or whenever links break):

    mamba run -n geo_diffusion python bin/setup_repo_symlinks.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fallback when not run inside a git checkout (same targets as HEAD symlinks).
DEFAULT_RELATIVE_TARGETS = {
    "journal": Path("../score-matching-fisher-note/score-matching-fisher-note-repo/journal"),
    "report": Path("../score-matching-fisher-note/score-matching-fisher-note-repo/report"),
}


def _git_symlink_target(name: str) -> Path:
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{name}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return DEFAULT_RELATIVE_TARGETS[name]
    target = proc.stdout.strip()
    if not target:
        return DEFAULT_RELATIVE_TARGETS[name]
    target_path = Path(target)
    if target_path.is_absolute():
        return target_path
    return (REPO_ROOT / target_path).resolve()


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


def _remove_existing(link_path: Path) -> None:
    if link_path.is_symlink():
        link_path.unlink()
        return
    if link_path.is_file():
        link_path.unlink()
        return
    if link_path.is_dir():
        # Junction or real directory.
        if sys.platform == "win32":
            subprocess.run(
                ["cmd", "/c", "rmdir", str(link_path)],
                check=True,
            )
            return
        raise SystemExit(
            f"{link_path} is a real directory; move it aside and rerun this script."
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
        # Prefer a directory symlink (matches git); fall back to a junction.
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
    for name in DEFAULT_RELATIVE_TARGETS:
        link_path = REPO_ROOT / name
        target_path = _git_symlink_target(name)

        current = _resolved_if_link(link_path)
        if current == target_path.resolve():
            print(f"[ok] {name}/ -> {target_path}")
            continue

        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and current not in (None, target_path.resolve()):
                _remove_existing(link_path)
            elif _is_plain_symlink_stub(link_path) or link_path.is_file():
                print(f"[fix] replacing plain-text stub {name}")
                _remove_existing(link_path)
            elif link_path.is_symlink() or (
                sys.platform == "win32" and link_path.is_dir()
            ):
                _remove_existing(link_path)
            else:
                raise SystemExit(f"Cannot replace existing path: {link_path}")

        _create_dir_link(link_path, target_path)
        print(f"[link] {name}/ -> {target_path}")
        created.append(name)

    return created


def main() -> None:
    created = ensure_repo_symlinks()
    if created:
        print(
            "\nDone. On Windows with git core.symlinks=false, "
            "`git status` may list journal/report as deleted; "
            "do not commit that — the working links are intentional."
        )
    else:
        print("\nAll repo symlinks already point at the notes tree.")


if __name__ == "__main__":
    main()
