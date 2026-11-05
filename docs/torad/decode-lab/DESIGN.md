# Decode comparison surface

## Purpose

An operating screen for seeing the ideal four-GPU decode budget beside the matching slots for a future measurement. The default is one active session at 400,000 context. The user's direction is visual-first: see the known budget and the missing evidence at a glance, inspect for detail. The ledger is evidence, not the interface.

## Composition

Two equal-width GPU diagrams occupy the desk viewport (3840 x 2050). Each modeled GPU's 100-square time mosaic uses the resource encoding: blue local work, hatched orange transfer/wait, neutral unknown activity. Future GPUs have empty outlined squares rather than a fictional zero-time distribution. Logical PCIe paths connect the GPU row to its host; future paths are dashed. Link width does not encode throughput.

Both sides have the same four phase slots and 45 target-layer slots. The left shows model budgets; the right shows empty outlines and missing values. The future total is '--', never zero. Workload identity and the pending state stay visible. Numeric readouts remain subordinate to the graphics. Historical whole-capture timelines live only in raw evidence, not opposite a single ideal cycle.

## Interaction

GPUs, paths, categories, phases and layers open a contextual inspector. Exact bytes, definitions and full event tables are behind deliberate actions. Model settings are an attached utility panel; raw evidence is a page section opened on demand. On the wide desk the utility reserves its own column without covering GPUs. At narrower widths it opens in the page below the diagrams. Escape closes the utility and returns focus. Timeline windows support arrow-key navigation and focus tooltips.

## Material and adaptation

Cool gray ground, opaque chart surfaces, Source Sans 3 labels, IBM Plex Mono measurements. No mandatory brand framework. Light and dark themes preserve resource identity; hatching supplies a second encoding for transfer time. No entrance animation or simulated playback.

At laptop width the diagrams remain side by side. On phones a paired time-share comparison fits in the first viewport, followed by stacked diagrams with readable two-by-two GPU grids. Data tables scroll within their own containers, never the page.

## Evidence limits

Empty future slots mean not measured, never zero. Historical gray means missing exported activity, not proven idle. Ideal blue is a byte budget, not measured compute occupancy. Unbalanced ownership does not acquire invented per-rank timings: with one session only the busiest owner's full timing is modeled. The real launcher is adaptive; the displayed reference is a two-proposal cycle, not a prediction of every adaptive cycle. Kernel-driven PCIe bandwidth remains unmeasured. Imports stay in raw evidence and never silently fill the future slots. Automatic rendering of matched-cycle comparison output is not implemented.
