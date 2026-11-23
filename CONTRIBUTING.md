# Working on this repository

For the three of us. Coding agents read [`AGENTS.md`](AGENTS.md) instead — that file is
the source of truth for the rules, and this one is the short path through them.

## Setup, once

```bash
make setup                               # venv + editable install + git hooks
source .venv/bin/activate
cp .env.example .env                     # then edit it; .env is gitignored
make check                               # lint + typecheck + test, the gates CI runs
```

**`make setup` installs the git hooks, and that step is not optional.** The hooks run
`ruff --fix` before every commit. CI was red on `main` for seventeen days over a single
unsorted import block that the hooks would have fixed silently — the commit came from a
clone where nobody had run `make setup`. If `make check` passes locally, CI passes.

## Starting a ticket

```bash
git checkout main && git pull
git checkout -b <yourname>/<short-slug>   # e.g. mohit/cohort-manifest
```

One ticket per branch where you can manage it. Branch from `main`, not from someone
else's branch, unless your ticket is genuinely blocked on theirs — the issue says which.

Before you push:

```bash
make check
```

Then open a PR with the **other two as reviewers**, and put `Closes #<n>` in the body so
the issue closes on merge rather than by hand.

## Commits

Write a subject and a body and stop. **No trailers** — no `Co-Authored-By`, no
generated-with line, no attribution of any kind. This applies to commits made by your
coding agent too; `AGENTS.md` says so, but agents have to be told twice.

Say *why*, not *what*. The diff already shows what changed.

## The three controls

These outrank convenience everywhere they conflict. Code that weakens one is wrong even
when it works, and none of them fails loudly:

1. **Cognitive scores never reach inference.** MMSE may inform labels; no model reads
   one when predicting ([ADR-0002](docs/adr/0002-no-cognitive-scores-at-inference.md)).
   Extends to anything that is a near-deterministic function of one.
2. **Partitions are assigned by subject, drawn once, never redrawn**
   ([ADR-0001](docs/adr/0001-subject-level-partitioning.md)). Watch for the quiet
   version: normalization statistics or target scalers fit on the whole cohort instead
   of on training only.
3. **The test partition is read once per arm**, after that arm's training and tuning are
   done. Every threshold and hyperparameter comes from validation subjects.

If your change touches any of them, say so in the PR body. A reviewer cannot spot a leak
they were not told to look for.

## Two rules that are easy to break by accident

**Never fabricate a number.** Tables in `docs/` carry `_tbd_` where a measurement
belongs. Leave them. A plausible-looking number in a research document is worse than a
blank, because a blank announces itself. Results tables are generated from recorded
runs, never typed.

**Data never enters git.** Scans, derivatives, subject-identifying manifests and model
weights stay out — a licensing obligation under the OASIS Data Use Agreement, not just
hygiene. Read paths from `OASIS_ROOT`. A pre-commit hook and a CI job enforce it, but
they are a backstop, not a permission slip.

## Where to look things up

| Looking for | Go to |
|---|---|
| What a word means | [`GLOSSARY.md`](GLOSSARY.md) |
| Why it is built this way | [`docs/adr/`](docs/adr/) |
| What will be run, and how it is judged | [`docs/protocol/`](docs/protocol/) |
| Who owns what, and the gates | [`docs/protocol/implementation-plan.md`](docs/protocol/implementation-plan.md) |
| Dataset, access status, cohort rules | [`docs/data/`](docs/data/) |

Use the glossary's words in code and in docstrings. Terms under `_Avoid_` are listed
because they blur a distinction this project depends on — *scan* not *volume*, *visit*
not *timepoint*. If a term is missing, add it in the same change that needed it.

## Decisions

Write an ADR when a choice is **hard to reverse**, **would surprise a reader who only
saw the code**, and **came out of a real trade-off**. All three, or it is just the
obvious thing and needs no record.

Amend `docs/protocol/` by appending a dated amendment rather than editing in place, so
the distance between what was planned and what was run stays visible.

## If you are stuck

The seams under `src/oasis_dl/` are typed interfaces with the contract in the docstring
and an unimplemented body. Read the docstring before writing the body — most of them
record a trap someone already thought about, and several name the specific bug the
contract exists to prevent.
