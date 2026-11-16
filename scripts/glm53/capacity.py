#!/usr/bin/env python3
"""Estimate MTP-2 DP capacity from two measured boot allocations; never a serving qualification."""
import argparse
import json
import math
from pathlib import Path

MIB = 1024 * 1024
SLOT_CELLS = 442368
MLA_LAYERS = 12  # Eleven target layers and the MTP layer.
LATENT_ROW_BYTES = 512 // 32 * 34
INDEXER_ROW_BYTES = 384 // 32 * 34


def estimate(lower_slots, lower, upper_slots, upper, *, host_kv=False, host_budget_mib=None,
             reserve_mib=4096, peak_extra_mib=2048, max_slots=256, devices=4, slot_cells=SLOT_CELLS):
    """Fit fixed + owner_slots * incremental allocation, retaining observed compute/metadata costs."""
    if devices not in (2, 4) or slot_cells < 256:
        raise ValueError("Devices must be 2 or 4 and a DP slot holds at least 256 cells")
    if len(lower) != devices or len(upper) != devices or lower_slots <= 0 or upper_slots <= lower_slots:
        raise ValueError(f"Need two increasing {devices}-card DP boot measurements")
    if lower_slots % devices or upper_slots % devices:
        raise ValueError(f"Calibration slot counts must divide evenly over {devices} cards")
    if any(not math.isfinite(n) or n < 0 for n in (reserve_mib, peak_extra_mib)) or not 1 <= max_slots <= 256:
        raise ValueError("Memory margins must be finite and nonnegative and max-slots 1..256")
    if host_kv and (host_budget_mib is None or not math.isfinite(host_budget_mib) or host_budget_mib <= 0):
        raise ValueError("Host KV requires a positive available host-memory budget, after other allocations")
    for snapshot in (lower, upper):
        if {r['device'] for r in snapshot} != set(range(devices)):
            raise ValueError(f"Each snapshot must identify all {devices} devices exactly once")
        if any(not math.isfinite(r[k]) or r[k] < 0 for r in snapshot for k in ('used_mib', 'free_mib')):
            raise ValueError("Memory measurements must be finite and nonnegative")
    lower = sorted(lower, key=lambda r: r['device'])
    upper = sorted(upper, key=lambda r: r['device'])
    latent_mib = slot_cells * MLA_LAYERS * LATENT_ROW_BYTES / MIB
    cache_mib = slot_cells * MLA_LAYERS * (LATENT_ROW_BYTES + INDEXER_ROW_BYTES) / MIB
    # Target and draft each keep the main graph plus up to three 32-row graph slots (llama-context.h).
    # Every graph reuses its scratch across successive attention layers.
    staging_mib = 2 * ((512 + 3 * 32) * 2080 * (LATENT_ROW_BYTES + 4)) / MIB if host_kv else 0
    cards = []
    for a, b in zip(lower, upper):
        total = a['used_mib'] + a['free_mib']
        if abs(total - b['used_mib'] - b['free_mib']) > 1:
            raise ValueError("Device memory capacity changed between calibration runs")
        incremental = (b['used_mib'] - a['used_mib']) / ((upper_slots - lower_slots) / devices)
        if incremental < cache_mib:
            raise ValueError("Observed allocation increase is smaller than the MTP-2 Q8 cache geometry")
        fixed = a['used_mib'] - (lower_slots / devices) * incremental
        if fixed < 0:
            raise ValueError("Calibration implies a negative fixed allocation")
        cards.append({'device': a['device'], 'total_mib': total, 'fixed_mib': fixed,
                      'incremental_mib': incremental - (latent_mib if host_kv else 0)})
    fits = []
    for slots in range(1, max_slots + 1):
        owners = [(slots + devices - 1 - d) // devices for d in range(devices)]
        free = [c['total_mib'] - c['fixed_mib'] - n * c['incremental_mib'] - staging_mib - peak_extra_mib
                for c, n in zip(cards, owners)]
        host_mib = slots * latent_mib if host_kv else 0
        if min(free) >= reserve_mib and (not host_kv or host_mib <= host_budget_mib):
            fits.append({'slots': slots, 'slots_per_card': owners, 'free_mib_per_card': free, 'host_kv_mib': host_mib})
    return {'estimated_max': fits[-1] if fits else None, 'mtp_depth': 2, 'devices': devices, 'slot_cells': slot_cells,
            'cache_mib_per_slot': cache_mib, 'host_latent_mib_per_slot': latent_mib if host_kv else 0,
            'staging_mib_per_card': staging_mib, 'reserve_mib': reserve_mib, 'peak_extra_mib': peak_extra_mib,
            'calibration': cards, 'quality_qualified': False, 'capacity_qualified': False,
            'scope': 'Linear memory estimate for the same pack, Q8 caches/state, MTP 2 and ubatch 512. Capture, allocator and workload changes require a new peak measurement.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lower-memory', type=Path, required=True)
    p.add_argument('--lower-slots', type=int, required=True)
    p.add_argument('--upper-memory', type=Path, required=True)
    p.add_argument('--upper-slots', type=int, required=True)
    p.add_argument('--host-kv', action='store_true')
    p.add_argument('--host-budget-mib', type=float)
    p.add_argument('--reserve-mib', type=float, default=4096)
    p.add_argument('--peak-extra-mib', type=float, default=2048)
    p.add_argument('--max-slots', type=int, default=256)
    p.add_argument('--devices', type=int, choices=(2, 4), default=4)
    p.add_argument('--slot-cells', type=int, default=SLOT_CELLS)
    a = p.parse_args()
    result = estimate(a.lower_slots, json.loads(a.lower_memory.read_text()),
                      a.upper_slots, json.loads(a.upper_memory.read_text()), host_kv=a.host_kv,
                      host_budget_mib=a.host_budget_mib, reserve_mib=a.reserve_mib,
                      peak_extra_mib=a.peak_extra_mib, max_slots=a.max_slots,
                      devices=a.devices, slot_cells=a.slot_cells)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
