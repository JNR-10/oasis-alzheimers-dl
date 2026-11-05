# Documentation

| Document | What it answers |
|---|---|
| [`../GLOSSARY.md`](../GLOSSARY.md) | What does this word mean here? |
| [`adr/`](adr/) | Why was it built this way? |
| [`data/`](data/) | What data, obtained how, under what terms? |
| [`protocol/`](protocol/) | What exactly will be run, and how is it judged? |
| [`protocol/implementation-plan.md`](protocol/implementation-plan.md) | Who builds what, by when, and where scope gets cut |
| [`survey/`](survey/) | What does the prior work already do? |
| [`audits/`](audits/) | What could go wrong, and what is the novelty claim? |
| [`proposal/`](proposal/) | What was submitted to the course? |

Decisions go in `adr/` and definitions go in `GLOSSARY.md`. Keep them apart: a
glossary that absorbs rationale stops being a glossary, and a decision record that
redefines terms splits the vocabulary.

`protocol/` is written **before** results exist and is not edited to match them.
Changes to it after the first training run are recorded as amendments with dates,
so the distance between what was planned and what was run stays visible.
