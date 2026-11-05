# Working in this repository

Instructions for any coding agent working here — Claude Code, Cursor, Codex,
and anything else that reads `AGENTS.md`.

A research project. The deliverable is a defensible finding, not a high number, so the
controls below outrank convenience everywhere they conflict.

## Never fabricate a result

Tables in `docs/` carry `_tbd_` where a real measurement belongs. Leave them. A
plausible-looking number in a research document is worse than a blank, because a blank
announces itself and a number does not.

This covers cohort counts, metrics, dates in `docs/data/access-log.md`, and anything
quoting what a model "achieved". Numbers from the predecessor project are quotable and
already cited; anything about *this* project's models is unmeasured until a run
produces it. Results tables are generated from recorded runs, never typed.

## The three controls

Code that would weaken one of these is wrong even when it would work:

1. **Cognitive scores never reach inference.** MMSE may inform labels; no model reads
   one when predicting ([ADR-0002](docs/adr/0002-no-cognitive-scores-at-inference.md)).
   This extends to any feature that is a near-deterministic function of one.
2. **Partitions are assigned by subject, drawn once, never redrawn**
   ([ADR-0001](docs/adr/0001-subject-level-partitioning.md)). Re-drawing a split after
   seeing results is test-set fitting. Watch for the quiet version: normalization
   statistics or target scalers fit on the full cohort rather than on training only.
3. **The test partition is read once per arm**, after that arm's training and tuning
   are complete. Thresholds and every hyperparameter come from validation subjects.

## Data never enters git

Imaging volumes, derivatives, subject-identifying manifests, and model weights stay out
of version control — a licensing obligation under the OASIS Data Use Agreement, not
only repository hygiene. `.gitignore`, a pre-commit hook, and a CI job enforce it. When
something needs a path, read it from `OASIS_ROOT`.

## The seams are unimplemented on purpose

Every module under `src/oasis_dl/` is a typed interface whose body raises
`NotImplementedError`, with the contract in the docstring. Most are blocked on OASIS-3
access; the label rule is additionally blocked on
[ADR-0004](docs/adr/0004-label-definition-and-target-visit.md) reaching `accepted`.

Implement one when asked to implement it. Filling in a seam encountered in passing is
scope the user did not ask for.

## Use the project's words

[`GLOSSARY.md`](GLOSSARY.md) fixes the vocabulary — *subject*, *session*, *visit*,
*target visit*, *impaired*, *inference feature*, *training target*. Terms listed under
`_Avoid_` are there because they blur a distinction this project depends on. When a
term is missing or a conversation sharpens one, update the glossary in the same change.

## Where things live

| Looking for | Go to |
|---|---|
| What a word means | [`GLOSSARY.md`](GLOSSARY.md) |
| Why it is built this way | [`docs/adr/`](docs/adr/) |
| What will be run, and how it is judged | [`docs/protocol/`](docs/protocol/) |
| Dataset, access status, cohort rules | [`docs/data/`](docs/data/) |
| What the prior work already does | [`docs/survey/`](docs/survey/) |

Decisions go in `docs/adr/`, definitions in `GLOSSARY.md`. Keep them apart. Write a new
ADR when a choice is hard to reverse, would surprise a reader who only saw the code, and
came out of a real trade-off — all three, or it is just the obvious thing.

Amend `docs/protocol/` by appending a dated amendment rather than editing in place, so
the distance between what was planned and what was run stays visible.

## Commits

No trailers. No `Co-Authored-By`, no generated-with line, no attribution of any kind —
write the subject and body and stop.

## Why there is also a CLAUDE.md

It is one line, `@AGENTS.md`, and this file is the only source of truth.

Claude Code reads `AGENTS.md` directly, but **only when no `CLAUDE.md` exists** — so a
repo holding both, each with its own content, gives Claude Code one set of instructions
and everyone else's agent another, and the two drift. The import collapses that: every
agent reads this file.

It is an import rather than a symlink because git checks a committed symlink out as a
plain text file on Windows unless `core.symlinks` is set, which would leave that clone
with a one-line `CLAUDE.md` instead of these instructions.

Add Claude-Code-specific instructions under the import in `CLAUDE.md`. Everything that
applies to any agent belongs here.
