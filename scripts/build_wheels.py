# Copyright (c) 2026 Trillim. Proprietary and confidential. See LICENSE.
"""Build platform-specific wheels for all 6 supported platforms.

WARNING: This script is NOT for general use. It requires access to the private
DarkNet repository with pre-compiled binaries for all target platforms. Only
Trillim developers building distribution wheels should run this.

For each platform:
1. Cleans src/trillim/_bin/
2. Copies the correct binaries from DarkNet's executables/<platform>/
3. Builds a wheel with `uv build --wheel`
4. Retags the wheel with the correct platform tag
5. Moves the tagged wheel to dist/

Usage:
    uv run scripts/build_wheels.py                 # Build all 6 platforms
    uv run scripts/build_wheels.py linux-x86_64    # Build one platform
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DARKNET = ROOT.parent / "DarkNet"
BIN_DIR = ROOT / "src" / "trillim" / "_bin"
DIST_DIR = ROOT / "dist"

BINARIES = ["inference", "trillim-quantize"]

PLATFORMS: dict[str, dict] = {
    "linux-x86_64": {
        "src": DARKNET / "executables",
        "tag": "manylinux_2_17_x86_64.manylinux2014_x86_64",
        "exe_suffix": "",
    },
    "linux-arm64": {
        "src": DARKNET / "executables" / "linux-arm64",
        "tag": "manylinux_2_17_aarch64.manylinux2014_aarch64",
        "exe_suffix": "",
    },
    "macos-x86_64": {
        "src": DARKNET / "executables" / "macos-x86_64",
        "tag": "macosx_11_0_x86_64",
        "exe_suffix": "",
    },
    "macos-arm64": {
        "src": DARKNET / "executables" / "macos-arm64",
        "tag": "macosx_11_0_arm64",
        "exe_suffix": "",
    },
    "win-x86_64": {
        "src": DARKNET / "executables" / "win-x86_64",
        "tag": "win_amd64",
        "exe_suffix": ".exe",
    },
    "win-arm64": {
        "src": DARKNET / "executables" / "win-arm64",
        "tag": "win_arm64",
        "exe_suffix": ".exe",
    },
}


def clean_bin_dir() -> None:
    if BIN_DIR.exists():
        shutil.rmtree(BIN_DIR)
    BIN_DIR.mkdir(parents=True)


def copy_binaries(platform: str) -> None:
    info = PLATFORMS[platform]
    src_dir: Path = info["src"]
    suffix: str = info["exe_suffix"]

    for name in BINARIES:
        src = src_dir / (name + suffix)
        dst = BIN_DIR / (name + suffix)
        if not src.exists():
            print(f"  ERROR: {src} not found", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(src, dst)
        dst.chmod(0o755)

    # Remove .gitkeep so force-include doesn't duplicate it
    gitkeep = BIN_DIR / ".gitkeep"
    if gitkeep.exists():
        gitkeep.unlink()

    print(f"  Copied binaries from {src_dir}")


def build_wheel() -> Path:
    result = subprocess.run(
        ["uv", "build", "--wheel"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  uv build failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    wheels = list(DIST_DIR.glob("trillim-*.whl"))
    if not wheels:
        print("  ERROR: no wheel found in dist/", file=sys.stderr)
        sys.exit(1)

    # Return the most recently created wheel
    return max(wheels, key=lambda p: p.stat().st_mtime)


def retag_wheel(wheel_path: Path, platform_tag: str) -> Path:
    """Rename wheel file and patch WHEEL metadata inside the zip."""
    name = wheel_path.name
    # Replace py3-none-any with py3-none-<platform_tag>
    new_name = name.replace("py3-none-any", f"py3-none-{platform_tag}")
    new_path = DIST_DIR / new_name

    with zipfile.ZipFile(wheel_path, "r") as zin:
        with zipfile.ZipFile(new_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)

                # Patch the WHEEL metadata
                if item.filename.endswith(".dist-info/WHEEL"):
                    text = data.decode("utf-8")
                    text = text.replace(
                        "Tag: py3-none-any",
                        f"Tag: py3-none-{platform_tag}",
                    )
                    data = text.encode("utf-8")

                # Patch the RECORD filename references
                new_filename = item.filename
                if "py3-none-any" in item.filename:
                    new_filename = item.filename.replace(
                        "py3-none-any", f"py3-none-{platform_tag}"
                    )

                zout.writestr(
                    zipfile.ZipInfo(new_filename, date_time=item.date_time),
                    data,
                )

    # Remove the original any-tagged wheel
    wheel_path.unlink()

    print(f"  Tagged: {new_name}")
    return new_path


def build_platform(platform: str) -> Path:
    info = PLATFORMS[platform]
    print(f"\n[{platform}]")

    clean_bin_dir()
    copy_binaries(platform)
    wheel = build_wheel()
    tagged = retag_wheel(wheel, info["tag"])

    # Clean bin dir after build so binaries aren't left around
    clean_bin_dir()

    return tagged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build platform-specific wheels")
    parser.add_argument(
        "platforms",
        nargs="*",
        choices=list(PLATFORMS.keys()),
        help="Platforms to build (default: all)",
    )
    args = parser.parse_args()
    if not args.platforms:
        args.platforms = list(PLATFORMS.keys())

    DIST_DIR.mkdir(exist_ok=True)

    built: list[Path] = []
    for platform in args.platforms:
        built.append(build_platform(platform))

    print(f"\nBuilt {len(built)} wheel(s) in {DIST_DIR}/:")
    for w in built:
        print(f"  {w.name}")


if __name__ == "__main__":
    main()
