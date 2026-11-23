<!--
Keep this short. The checklist is here to catch the failures that pass silently,
not to make you tick boxes. Delete any section that does not apply.
-->

Closes #

## What this changes

<!-- What now works that did not before, from the point of view of someone using it.
     The diff already says what changed; say why it changed. -->

## Checks

- [ ] `make check` passes locally (lint, typecheck, tests)
- [ ] No scans, derivatives, manifests or model weights in the diff
- [ ] Any new or sharpened term is in `GLOSSARY.md`
- [ ] No numbers typed into `docs/` that a run did not produce

## Did this touch a control?

Tick anything the change comes near, so a reviewer knows where to look. None of these
fails loudly, which is why they are asked about rather than tested for.

- [ ] **Cognitive scores** — could any model read MMSE, or something that is nearly a
      function of it, at inference time? (ADR-0002)
- [ ] **Partitioning** — does anything split, re-split, or fit statistics across the
      partition boundary? Normalization and target scalers count. (ADR-0001)
- [ ] **The test partition** — is it read anywhere before that arm's tuning is done?
- [ ] **None of the above**

## Anything a reviewer should push back on

<!-- A decision you were unsure about, a shortcut you took, scope you added beyond the
     ticket. Saying it here is cheaper than it being found later. -->
