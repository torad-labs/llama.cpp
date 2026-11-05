# Decode comparison surface

## Purpose

An operating screen for seeing where four-GPU decode spends time, ideal and recorded side by side. The user's direction is visual-first: understand the difference at a glance, inspect for detail. The ledger is evidence, not the interface.

## Composition

Two equal-width GPU diagrams occupy the desk viewport (3840 x 2050). Each GPU's 100-square time mosaic uses the same resource encoding: blue local work, hatched orange transfer/wait, neutral unrecorded activity. Logical PCIe paths connect the GPU row to its host. Link width does not encode throughput.

The ideal has a four-phase path and 45 layer budget bars. The recorded side has four interval timelines. These are independent scopes, not aligned execution traces. Workload identity and the mismatch stay visible. Numeric readouts remain subordinate to the graphics.

## Interaction

GPUs, paths, categories, phases and layers open a contextual inspector. Exact bytes, definitions and full event tables are behind deliberate actions. Model settings are an attached utility panel; raw evidence is a page section opened on demand. On the wide desk the utility reserves its own column without covering GPUs. At narrower widths it opens in the page below the diagrams. Escape closes the utility and returns focus. Timeline windows support arrow-key navigation and focus tooltips.

## Material and adaptation

Cool gray ground, opaque chart surfaces, Source Sans 3 labels, IBM Plex Mono measurements. No mandatory brand framework. Light and dark themes preserve resource identity; hatching supplies a second encoding for transfer time. No entrance animation or simulated playback.

At laptop width the diagrams remain side by side. On phones a paired time-share comparison fits in the first viewport, followed by stacked diagrams with readable two-by-two GPU grids. Data tables scroll within their own containers, never the page.

## Evidence limits

Gray is missing exported activity, not proven idle. Ideal blue is a byte budget, not measured compute occupancy. Unbalanced ownership does not acquire invented per-rank timings. Kernel-driven PCIe bandwidth remains unmeasured. The data's scopes are part of the visual design, not footnotes to erase.
