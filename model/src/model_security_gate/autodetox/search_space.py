from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence


def bounded_grid(space: Mapping[str, Sequence[Any]], *, max_points: int = 16) -> list[dict[str, Any]]:
    """Deterministic small-grid generator used by AutoDetox recipes."""

    keys = list(space.keys())
    rows = [dict(zip(keys, values)) for values in product(*(space[k] for k in keys))]
    return rows[: max(1, int(max_points))]


def last_mile_weight_soup_space() -> list[dict[str, Any]]:
    return [{"alpha": a} for a in [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06]]


def targeted_negative_space() -> list[dict[str, Any]]:
    return bounded_grid(
        {
            "lr": [8e-6, 4e-6, 2e-6],
            "epochs": [1, 2, 3],
            "target_absent_weight": [0.5, 1.0],
        },
        max_points=12,
    )


def geometry_frequency_space() -> list[dict[str, Any]]:
    return bounded_grid(
        {
            "lr": [1e-6, 5e-7],
            "epochs": [1, 2],
            "geometry_weight": [0.5, 1.0],
            "frequency_weight": [0.5, 1.0],
        },
        max_points=12,
    )
