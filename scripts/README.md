# Scripts

One-off operational scripts: downloads, cohort inspection, figure regeneration, QC
montages.

Pipeline stages are **not** here — those are Hydra entrypoints under
`src/oasis_dl/cli/`, reachable through `make`. A script that starts composing configs
or training a model belongs in the package, where it can be imported and tested.
