"""Verify flat repository files and reconstruct the local research workspace."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


ROOT = Path(__file__).resolve().parent


def normalized_digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def safe_destination(base, member):
    path = PurePosixPath(member)
    if not path.parts or path.is_absolute() or "\\" in member:
        raise ValueError(f"Invalid archive path: {member}")
    if any(part in ("..", ".") or ":" in part for part in path.parts):
        raise ValueError(f"Invalid archive path: {member}")
    base = base.resolve()
    destination = base.joinpath(*path.parts).resolve()
    if destination == base or base not in destination.parents:
        raise ValueError(f"Archive path leaves the workspace: {member}")
    return destination


def load_manifest():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported manifest format")
    return manifest


def verify_flat_files(manifest):
    for name, item in manifest["files"].items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"Manifest entry is not a top-level file: {name}")
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() == item["sha256"]:
            continue
        if item.get("sha256_lf") == normalized_digest(data):
            continue
        raise ValueError(f"Checksum mismatch: {name}. Use the complete original download.")


def validate_archive(archive, spec, base):
    expected = spec["members"]
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(name.casefold() for name in names)):
            raise ValueError(f"Duplicate archive names: {archive.name}")
        if set(names) != set(expected):
            raise ValueError(f"Unexpected archive contents: {archive.name}")
        for info in z.infolist():
            safe_destination(base, info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Unexpected archive entry: {info.filename}")
            if hashlib.sha256(z.read(info)).hexdigest() != expected[info.filename]:
                raise ValueError(f"Archive member checksum mismatch: {info.filename}")


def prepare_workspace(workspace="_workspace", check_only=False):
    manifest = load_manifest()
    verify_flat_files(manifest)
    base = safe_destination(ROOT, workspace)
    for name, spec in manifest["archives"].items():
        validate_archive(ROOT / name, spec, base)
    if check_only:
        print(f"Verified {len(manifest['files'])} top-level files and three asset archives.")
        return base

    created = preserved = 0
    for name, relative in manifest["code_paths"].items():
        destination = safe_destination(base, relative)
        source = ROOT / name
        if destination.exists():
            if normalized_digest(destination.read_bytes()) != normalized_digest(source.read_bytes()):
                raise ValueError(f"Local code differs: {destination}. Keep your changes and use a fresh workspace.")
            preserved += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        created += 1

    # Existing results may come from a new run and must not be replaced by the archive.
    for name in manifest["archives"]:
        with zipfile.ZipFile(ROOT / name) as z:
            for info in z.infolist():
                destination = safe_destination(base, info.filename)
                if destination.exists():
                    preserved += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                created += 1
    print(f"Prepared {base.name}: {created} files restored, {preserved} existing files preserved.")
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify files without creating a workspace")
    parser.add_argument("--workspace", default="_workspace", help="Local workspace path below this repository")
    args = parser.parse_args()
    prepare_workspace(args.workspace, check_only=args.check)


if __name__ == "__main__":
    main()
