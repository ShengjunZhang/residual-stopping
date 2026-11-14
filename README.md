# Residual-based stopping rules for equilibrium-constrained optimization

Companion code and data for the manuscript by Shengjun Zhang and Tingyi Liu,
prepared for Applied Mathematics and Computation.

Corresponding author: Tingyi Liu, Wuhan University, tingyiL@whu.edu.cn.

## Files

All repository files are stored at the top level. The original Python programs
are available as individual files. Three archives contain the larger assets:

- `input_data.zip`: six cached TNTP network and demand files.
- `recorded_results.zip`: the recorded CSV results and JSON metadata.
- `figures.zip`: the EPS and PDF figures supplied with the manuscript.

`manifest.json` records the original location and checksums of the files.
`prepare.py` reconstructs their working directories inside `_workspace/` on your
computer. That generated directory is ignored by Git; it is not part of the
flat repository layout.

## Quick Start

Use Python 3.10 or newer. From the downloaded repository directory:

```bash
python -m pip install -r requirements.txt
python prepare.py --check
python run.py validate
python run.py block-check
python run.py figures
```

`run.py` prepares the working directory automatically. The commands above run
independent enclosure checks and rebuild the main figures from recorded data.
They do not rerun the full optimization benchmarks. Rebuilt figures and validation
outputs are written under `_workspace/manuscript_jamc_v6/`.

The scientific programs are unchanged. Their historical directory name,
`manuscript_jamc_v6`, is retained inside the local workspace so that relative
imports and data paths continue to work.

## Benchmarks

```bash
python run.py sparse -- --etas
python run.py paired -- --etas
python run.py network
python run.py face-check
```

The sparse tests use 200, 1000, and 5000 lower variables. The elastic-demand
transportation models have 555, 1350, and 1634 candidate-path variables.
Each benchmark writes its new measurements inside `_workspace/`; the uploaded
archives retain the recorded measurements.

To run another included program:

```bash
python run.py script run_tntp_multi_network_benchmark.py
```

See `REPRODUCIBILITY_DETAILS.txt` for experimental settings and policy definitions.
Commands in that file refer to paths inside `_workspace/`; use that directory
as the working directory when following those original commands directly.

## Recorded Results

Preparation preserves results already present in `_workspace/`, including outputs
from a previous run. Use a fresh download to restore every archived measurement.
Changing the flat source files requires updating their manifest; preparation stops
if the source checksums no longer match. Text checksums allow line-ending changes
between Windows and Unix.

The manuscript's numerical results use the recorded benchmark environment.
New timings depend on the local machine and numerical libraries.

## Data Source and Rights

The cached transportation inputs come from the
[Transportation Networks for Research repository](https://github.com/bstabler/TransportationNetworks).
Their original headers and academic-research terms are retained. They require
source attribution and are not relicensed by this repository.

See `RIGHTS_AND_ATTRIBUTION.txt` for the distinction between original code,
generated results, and third-party inputs. No blanket open-source license is
applied to the combined material.

## Citation

Author and manuscript details are provided in `CITATION.cff` and
`repository_metadata.json`. No article DOI or repository DOI has been assigned
in these files.
