# Experiment records

One directory per run, written by Hydra: `experiments/<experiment>/<timestamp>/`.

Each holds the resolved config, `provenance.json` (git commit, dirty flag, seed,
library versions, GPU, manifest path), logs, and metrics. Artifacts — checkpoints,
predictions — are gitignored; the record is what stays.

A result whose configuration cannot be recovered does not go in the report
(ADR-0008). A run made on a dirty checkout is fine while exploring and disqualifying
for a reported number, because the code that produced it no longer exists anywhere.
