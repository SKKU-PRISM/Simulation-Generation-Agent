"""Deterministic task taxonomy helpers mirrored from docs/task_taxonomy.md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PRIMARY_LABELS = {
    "prehensile": "파지 (Prehensile)",
    "placement": "배치 (Placement)",
    "non_prehensile": "비파지 (Non-prehensile)",
    "articulated": "관절체 (Articulated)",
    "insertion_assembly": "삽입/조립 (Insertion/Assembly)",
    "deformable": "변형체 (Deformable)",
    "rotational": "회전 (Rotational)",
    "contact_rich": "접촉 (Contact-rich)",
}


@dataclass(frozen=True)
class TaskClassification:
    """Primary task classification used in reports."""

    key: str
    label: str
    detail: str


def classify_task_yaml(task_path: str | Path) -> TaskClassification:
    """Classify a task YAML path using the current taxonomy rules."""

    path = Path(task_path)
    stem = path.stem.lower()
    parts = path.parts

    try:
        tasks_idx = parts.index("tasks")
        category = parts[tasks_idx + 2].lower()
    except (ValueError, IndexError):
        category = ""

    if category == "reach":
        return TaskClassification(
            key="non_prehensile",
            label=PRIMARY_LABELS["non_prehensile"],
            detail="reach",
        )

    if category == "cabinet" and stem.endswith("_cabinet"):
        return TaskClassification(
            key="articulated",
            label=PRIMARY_LABELS["articulated"],
            detail="cabinet",
        )

    if category == "assembly" and stem.endswith("_lift_peg_upright"):
        return TaskClassification(
            key="rotational",
            label=PRIMARY_LABELS["rotational"],
            detail="lift_peg_upright",
        )

    if category == "assembly" and stem.endswith("_assembling_kits"):
        return TaskClassification(
            key="insertion_assembly",
            label=PRIMARY_LABELS["insertion_assembly"],
            detail="assembling_kits",
        )

    if category == "assembly" and stem.endswith("_peg_insertion_side"):
        return TaskClassification(
            key="insertion_assembly",
            label=PRIMARY_LABELS["insertion_assembly"],
            detail="peg_insertion_side",
        )

    if category == "assembly" and stem.endswith("_plug_charger"):
        return TaskClassification(
            key="insertion_assembly",
            label=PRIMARY_LABELS["insertion_assembly"],
            detail="plug_charger",
        )

    if category == "peg_insert":
        return TaskClassification(
            key="insertion_assembly",
            label=PRIMARY_LABELS["insertion_assembly"],
            detail="peg_insert",
        )

    if category == "pick_place":
        return TaskClassification(
            key="placement",
            label=PRIMARY_LABELS["placement"],
            detail="pick_place",
        )

    if category == "sort":
        return TaskClassification(
            key="placement",
            label=PRIMARY_LABELS["placement"],
            detail="sort",
        )

    if category == "stack":
        return TaskClassification(
            key="placement",
            label=PRIMARY_LABELS["placement"],
            detail="stack",
        )

    if category == "cabinet":
        return TaskClassification(
            key="placement",
            label=PRIMARY_LABELS["placement"],
            detail="cabinet_object_transfer",
        )

    if category == "lift":
        return TaskClassification(
            key="prehensile",
            label=PRIMARY_LABELS["prehensile"],
            detail="lift",
        )

    return TaskClassification(
        key="prehensile",
        label=PRIMARY_LABELS["prehensile"],
        detail=category or "unknown",
    )
