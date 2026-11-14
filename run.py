"""Run an archived research program from the flat GitHub repository."""

import argparse
from pathlib import Path
import subprocess
import sys

from prepare import ROOT, load_manifest, prepare_workspace


ACTIONS = {
    "validate": ("validate_ellipsoid_enclosure.py", []),
    "block-check": ("validate_block_enclosure.py", []),
    "figures": ("build_v3_results.py", []),
    "sparse": ("run_sparse_enclosure_benchmark.py", []),
    "paired": ("run_sparse_enclosure_benchmark.py", ["--paired-relative"]),
    "network": ("run_elastic_network_benchmark.py", []),
    "face-check": ("run_compatible_face_benchmark.py", ["--validate-only"]),
    "legacy-figures": ("build_numerics.py", []),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[*ACTIONS, "script"])
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.action == "script":
        if not extra:
            parser.error("The script action needs the name of an included Python program")
        name, extra = extra[0], extra[1:]
        fixed = []
    else:
        name, fixed = ACTIONS[args.action]
    manifest = load_manifest()
    if name not in manifest["code_paths"]:
        parser.error("Choose a research program included in this repository")
    workspace = prepare_workspace()
    target = workspace / Path(manifest["code_paths"][name])
    return subprocess.run([sys.executable, str(target), *fixed, *extra], cwd=workspace).returncode


if __name__ == "__main__":
    raise SystemExit(main())
