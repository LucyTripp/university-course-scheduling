from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import argparse
from typing import Any, Literal

import pandas as pd
from ortools.sat.python import cp_model


AssignmentKey = tuple[int, int, str]
ResourceSlot = tuple[int, int, int]
ObjectiveMode = Literal["room_efficiency", "student_conflicts"]


def false_mask(series: pd.Series) -> pd.Series:
    """Return True where a CSV-loaded column represents False."""
    return series.astype(str).str.strip().str.lower().eq("false")


def true_mask(series: pd.Series) -> pd.Series:
    """Return True where a CSV-loaded column represents True."""
    return series.astype(str).str.strip().str.lower().eq("true")


def optional_int(value: Any) -> int | None:
    """Convert a pandas/CSV value to int, preserving missing values as None."""
    if pd.isna(value):
        return None
    return int(value)


def parse_time_key(time_key: str) -> tuple[str, int, int]:
    """Parse 'days_mask|start_slot|length_slots'."""
    days_mask, start_slot, length_slots = str(time_key).split("|")
    return days_mask.zfill(7), int(start_slot), int(length_slots)


def occupied_slots(time_key: str) -> list[tuple[int, int]]:
    """Expand one meeting pattern into (day_index, five-minute slot) pairs."""
    days_mask, start_slot, length_slots = parse_time_key(time_key)
    result: list[tuple[int, int]] = []

    for day_index, meets in enumerate(days_mask):
        if meets == "1":
            for slot in range(start_slot, start_slot + length_slots):
                result.append((day_index, slot))

    return result


def meeting_patterns_overlap(time_a: str, time_b: str) -> bool:
    """Return True when two weekly meeting patterns overlap in day and time."""
    days_a, start_a, length_a = parse_time_key(time_a)
    days_b, start_b, length_b = parse_time_key(time_b)

    same_day = any(
        flag_a == "1" and flag_b == "1"
        for flag_a, flag_b in zip(days_a, days_b)
    )
    if not same_day:
        return False

    end_a = start_a + length_a
    end_b = start_b + length_b
    return start_a < end_b and start_b < end_a


def date_masks_overlap(mask_a: str | None, mask_b: str | None) -> bool:
    """Return True when two classes are active on at least one common term date.

    UniTime uses a binary class-level dates mask in addition to the weekly day mask.
    Missing masks are treated conservatively as overlapping.
    """
    if not mask_a or not mask_b:
        return True

    a = str(mask_a).strip()
    b = str(mask_b).strip()
    if not a or not b or a.lower() == "nan" or b.lower() == "nan":
        return True

    if len(a) != len(b):
        # All classes in one Purdue instance should use the same term calendar.
        # If malformed input violates that assumption, do not silently allow a clash.
        return True

    return any(x == "1" and y == "1" for x, y in zip(a, b))


def load_processed_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the CSV files written by data_extraction_purdue.py."""
    files = {
        "classes": "classes.csv",
        "rooms": "rooms.csv",
        "class_rooms": "class_room_options.csv",
        "class_times": "class_time_options.csv",
        "class_instructors": "class_instructors.csv",
        "group_constraints": "group_constraints.csv",
        "constraint_classes": "constraint_classes.csv",
        "class_pair_conflicts": "class_pair_conflicts.csv",
    }

    tables: dict[str, pd.DataFrame] = {}
    missing: list[Path] = []

    for table_name, filename in files.items():
        path = data_dir / filename
        if path.exists():
            tables[table_name] = pd.read_csv(path)
        else:
            missing.append(path)

    if missing:
        missing_text = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing processed data files:\n  "
            + missing_text
            + "\nRun data_extraction_purdue.py first."
        )

    return tables


def build_lookup_tables(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Convert the processed DataFrames into convenient Python lookup tables."""
    classes_df = tables["classes"]
    rooms_df = tables["rooms"]
    class_rooms_df = tables["class_rooms"]
    class_times_df = tables["class_times"]
    class_instructors_df = tables["class_instructors"]
    group_constraints_df = tables["group_constraints"]
    constraint_classes_df = tables["constraint_classes"]

    schedulable_df = classes_df.loc[false_mask(classes_df["committed"])].copy()
    committed_df = classes_df.loc[~false_mask(classes_df["committed"])].copy()

    classes = sorted(
        schedulable_df["class_id"].dropna().astype(int).unique().tolist()
    )
    committed_classes = sorted(
        committed_df["class_id"].dropna().astype(int).unique().tolist()
    )

    class_dates: dict[int, str | None] = {}
    for class_value, dates_value in zip(
        classes_df["class_id"].tolist(),
        classes_df["dates_mask"].tolist(),
    ):
        class_id = optional_int(class_value)
        if class_id is None:
            continue
        if pd.isna(dates_value):
            class_dates[class_id] = None
        else:
            class_dates[class_id] = str(dates_value).strip()

    room_is_constrained: dict[int, bool] = {}
    has_constraint_column = "constraint_room" in rooms_df.columns
    for _, row in rooms_df.iterrows():
        room_id = optional_int(row.get("room_id"))
        if room_id is None:
            continue
        if not has_constraint_column or pd.isna(row.get("constraint_room")):
            room_is_constrained[room_id] = True
        else:
            room_is_constrained[room_id] = (
                str(row.get("constraint_room")).strip().lower() == "true"
            )

    class_limit: dict[int, int | None] = {}
    for class_id, limit in zip(classes_df["class_id"], classes_df["class_limit"]):
        if pd.notna(class_id):
            class_limit[int(class_id)] = optional_int(limit)

    room_capacity: dict[int, int | None] = {}
    for room_id, capacity in zip(rooms_df["room_id"], rooms_df["capacity"]):
        if pd.notna(room_id):
            room_capacity[int(room_id)] = optional_int(capacity)

    # Build these lookups row-by-row instead of iterating over groupby keys.
    # This is a little more verbose, but it keeps Pylance from treating the
    # groupby key as an arbitrary Hashable and flagging int(class_id).
    room_option_sets: dict[int, set[int]] = defaultdict(set)
    valid_room_rows = class_rooms_df.dropna(subset=["class_id", "room_id"])
    for class_value, room_value in zip(
        valid_room_rows["class_id"].tolist(),
        valid_room_rows["room_id"].tolist(),
    ):
        class_id = optional_int(class_value)
        room_id = optional_int(room_value)
        if class_id is not None and room_id is not None:
            room_option_sets[class_id].add(room_id)

    room_options: dict[int, list[int]] = {
        class_id: sorted(room_ids)
        for class_id, room_ids in room_option_sets.items()
    }

    time_option_sets: dict[int, set[str]] = defaultdict(set)
    valid_time_rows = class_times_df.dropna(subset=["class_id", "time_key"])
    for class_value, time_value in zip(
        valid_time_rows["class_id"].tolist(),
        valid_time_rows["time_key"].tolist(),
    ):
        class_id = optional_int(class_value)
        if class_id is not None and pd.notna(time_value):
            time_option_sets[class_id].add(str(time_value))

    time_options: dict[int, list[str]] = {
        class_id: sorted(time_keys)
        for class_id, time_keys in time_option_sets.items()
    }

    instructor_sets: dict[int, set[int]] = defaultdict(set)
    valid_instructor_rows = class_instructors_df.dropna(
        subset=["class_id", "instructor_id"]
    )
    for class_value, instructor_value in zip(
        valid_instructor_rows["class_id"].tolist(),
        valid_instructor_rows["instructor_id"].tolist(),
    ):
        class_id = optional_int(class_value)
        instructor_id = optional_int(instructor_value)
        if class_id is not None and instructor_id is not None:
            instructor_sets[class_id].add(instructor_id)

    instructors_by_class: dict[int, list[int]] = {
        class_id: sorted(instructor_ids)
        for class_id, instructor_ids in instructor_sets.items()
    }


    # Classes in required CAN_SHARE_ROOM / MEET_WITH constraints are allowed
    # to occupy the same room at the same time (subject to room capacity).
    share_room_pairs: set[tuple[int, int]] = set()
    if not group_constraints_df.empty and not constraint_classes_df.empty:
        required_mask = (
            true_mask(group_constraints_df["required"])
            if "required" in group_constraints_df.columns
            else group_constraints_df["preference"].astype(str).eq("R")
        )
        share_constraints = group_constraints_df.loc[
            required_mask
            & group_constraints_df["constraint_type"]
            .astype(str)
            .str.upper()
            .isin(["CAN_SHARE_ROOM", "MEET_WITH", "MEET_WITH_WEEKS"])
        ]

        for constraint_key in share_constraints["constraint_key"].dropna().tolist():
            key_int = optional_int(constraint_key)
            if key_int is None:
                continue
            member_rows = constraint_classes_df.loc[
                constraint_classes_df["constraint_key"] == constraint_key,
                "class_id",
            ]
            member_ids = sorted(
                {
                    int(class_id)
                    for class_id in member_rows.dropna().tolist()
                }
            )
            for class_a, class_b in combinations(member_ids, 2):
                share_room_pairs.add((class_a, class_b))

    # Human-readable details for the final schedule.csv file.
    time_details: dict[str, dict[str, object]] = {}
    detail_columns = [
        "days_mask",
        "days",
        "start_slot",
        "length_slots",
        "start_time",
        "end_time",
    ]

    for _, row in class_times_df.dropna(subset=["time_key"]).iterrows():
        key = str(row["time_key"])
        if key not in time_details:
            time_details[key] = {
                column: row[column] if column in row.index else None
                for column in detail_columns
            }

    return {
        "classes": classes,
        "committed_classes": committed_classes,
        "class_dates": class_dates,
        "room_is_constrained": room_is_constrained,
        "class_limit": class_limit,
        "room_capacity": room_capacity,
        "room_options": room_options,
        "time_options": time_options,
        "instructors_by_class": instructors_by_class,
        "share_room_pairs": share_room_pairs,
        "time_details": time_details,
    }


def build_fixed_occupancy(
    tables: dict[str, pd.DataFrame],
    committed_classes: list[int],
    instructors_by_class: dict[int, list[int]],
    room_is_constrained: dict[int, bool],
) -> tuple[dict[ResourceSlot, set[int]], dict[ResourceSlot, set[int]]]:
    """Index committed Purdue placements by resource slot and class id."""
    class_rooms_df = tables["class_rooms"]
    class_times_df = tables["class_times"]
    committed_set = set(committed_classes)

    solution_rooms = class_rooms_df.loc[
        class_rooms_df["class_id"].isin(committed_set)
        & true_mask(class_rooms_df["is_solution"])
    ]
    solution_times = class_times_df.loc[
        class_times_df["class_id"].isin(committed_set)
        & true_mask(class_times_df["is_solution"])
    ]

    rooms_by_class: dict[int, list[int]] = defaultdict(list)
    for class_id, room_id in zip(
        solution_rooms["class_id"], solution_rooms["room_id"]
    ):
        if pd.notna(class_id) and pd.notna(room_id):
            rooms_by_class[int(class_id)].append(int(room_id))

    times_by_class: dict[int, list[str]] = defaultdict(list)
    for class_id, time_key in zip(
        solution_times["class_id"], solution_times["time_key"]
    ):
        if pd.notna(class_id) and pd.notna(time_key):
            times_by_class[int(class_id)].append(str(time_key))

    fixed_room_slots: dict[ResourceSlot, set[int]] = defaultdict(set)
    fixed_instructor_slots: dict[ResourceSlot, set[int]] = defaultdict(set)

    for class_id in committed_classes:
        for time_key in times_by_class.get(class_id, []):
            slots = occupied_slots(time_key)

            for room_id in rooms_by_class.get(class_id, []):
                if not room_is_constrained.get(room_id, True):
                    continue
                for day_index, slot in slots:
                    fixed_room_slots[(room_id, day_index, slot)].add(class_id)

            for instructor_id in instructors_by_class.get(class_id, []):
                for day_index, slot in slots:
                    fixed_instructor_slots[(instructor_id, day_index, slot)].add(
                        class_id
                    )

    return dict(fixed_room_slots), dict(fixed_instructor_slots)


def conflicts_with_committed(
    class_id: int,
    room_id: int,
    time_key: str,
    instructors_by_class: dict[int, list[int]],
    class_dates: dict[int, str | None],
    room_is_constrained: dict[int, bool],
    fixed_room_slots: dict[ResourceSlot, set[int]],
    fixed_instructor_slots: dict[ResourceSlot, set[int]],
    share_room_pairs: set[tuple[int, int]],
    class_limit: dict[int, int | None],
    room_capacity: dict[int, int | None],
) -> bool:
    """Return True if a candidate conflicts with a committed placement.

    Required CAN_SHARE_ROOM / MEET_WITH pairs may overlap in a room when the
    combined class limits fit the room.
    """
    candidate_dates = class_dates.get(class_id)
    slots = occupied_slots(time_key)

    if room_is_constrained.get(room_id, True):
        for day_index, slot in slots:
            overlapping_committed = [
                committed_class
                for committed_class in fixed_room_slots.get(
                    (room_id, day_index, slot), set()
                )
                if date_masks_overlap(
                    candidate_dates, class_dates.get(committed_class)
                )
            ]

            if overlapping_committed:
                # Every overlapping committed class must explicitly be allowed
                # to share the room with the candidate.
                if not all(
                    tuple(sorted((class_id, committed_class))) in share_room_pairs
                    for committed_class in overlapping_committed
                ):
                    return True

                capacity = room_capacity.get(room_id)
                candidate_size = class_limit.get(class_id)
                committed_sizes = [
                    class_limit.get(committed_class)
                    for committed_class in overlapping_committed
                ]
                if (
                    capacity is None
                    or candidate_size is None
                    or any(size is None for size in committed_sizes)
                    or candidate_size + sum(
                        int(size) for size in committed_sizes if size is not None
                    ) > capacity
                ):
                    return True

    for instructor_id in instructors_by_class.get(class_id, []):
        for day_index, slot in slots:
            for committed_class in fixed_instructor_slots.get(
                (instructor_id, day_index, slot), set()
            ):
                if date_masks_overlap(
                    candidate_dates, class_dates.get(committed_class)
                ):
                    return True

    return False

def add_date_aware_overlap_constraints(
    model: cp_model.CpModel,
    slot_entries: dict[
        ResourceSlot,
        list[tuple[int, AssignmentKey, cp_model.IntVar]],
    ],
    class_dates: dict[int, str | None],
    share_room_pairs: set[tuple[int, int]] | None = None,
) -> int:
    """Add pairwise no-overlap constraints for a resource.

    When share_room_pairs is supplied, those class pairs are exempt from the
    ordinary room exclusivity rule. Capacity is enforced separately.
    """
    seen_pairs: set[frozenset[AssignmentKey]] = set()
    count = 0
    share_room_pairs = share_room_pairs or set()

    for entries in slot_entries.values():
        if len(entries) < 2:
            continue

        for (class_a, key_a, var_a), (class_b, key_b, var_b) in combinations(
            entries, 2
        ):
            if class_a == class_b:
                continue

            pair = frozenset((key_a, key_b))
            if pair in seen_pairs:
                continue

            if not date_masks_overlap(
                class_dates.get(class_a), class_dates.get(class_b)
            ):
                continue

            if tuple(sorted((class_a, class_b))) in share_room_pairs:
                continue

            model.add_at_most_one([var_a, var_b])
            seen_pairs.add(pair)
            count += 1

    return count


def add_room_capacity_constraints(
    model: cp_model.CpModel,
    room_slot_vars: dict[
        ResourceSlot,
        list[tuple[int, AssignmentKey, cp_model.IntVar]],
    ],
    class_limit: dict[int, int | None],
    room_capacity: dict[int, int | None],
) -> int:
    """Ensure simultaneous room-sharing classes fit in the room.

    Non-sharing class pairs are already blocked by pairwise room constraints.
    This weighted constraint handles CAN_SHARE_ROOM / MEET_WITH groups, including
    cases where more than two classes share a room simultaneously.
    """
    count = 0
    for (room_id, _day_index, _slot), entries in room_slot_vars.items():
        capacity = room_capacity.get(room_id)
        if capacity is None or not entries:
            continue

        terms = []
        for class_id, _key, var in entries:
            size = class_limit.get(class_id)
            # If a class limit is unknown, conservatively treat it as using the
            # full room so it cannot share with another class.
            weight = capacity if size is None else int(size)
            terms.append(weight * var)

        model.add(sum(terms) <= capacity)
        count += 1

    return count

def validate_reference_solution(
    tables: dict[str, pd.DataFrame],
    lookups: dict[str, object],
) -> dict[str, int]:
    """Check Purdue's supplied solution against the hard rules used by this model.

    This is a debugging oracle: if the published/reference placement violates one
    of our simplified rules, that rule is probably being interpreted too strictly.
    """
    class_rooms_df = tables["class_rooms"]
    class_times_df = tables["class_times"]

    classes = lookups["classes"]
    committed_classes = lookups["committed_classes"]
    class_dates = lookups["class_dates"]
    room_is_constrained = lookups["room_is_constrained"]
    share_room_pairs = lookups["share_room_pairs"]
    room_capacity = lookups["room_capacity"]
    class_limit = lookups["class_limit"]
    instructors_by_class = lookups["instructors_by_class"]

    assert isinstance(classes, list)
    assert isinstance(committed_classes, list)
    assert isinstance(class_dates, dict)
    assert isinstance(room_is_constrained, dict)
    assert isinstance(share_room_pairs, set)
    assert isinstance(room_capacity, dict)
    assert isinstance(class_limit, dict)
    assert isinstance(instructors_by_class, dict)

    all_classes = set(classes) | set(committed_classes)

    solution_rooms_df = class_rooms_df.loc[
        class_rooms_df["class_id"].isin(all_classes)
        & true_mask(class_rooms_df["is_solution"])
    ]
    solution_times_df = class_times_df.loc[
        class_times_df["class_id"].isin(all_classes)
        & true_mask(class_times_df["is_solution"])
    ]

    rooms_by_class: dict[int, list[int]] = defaultdict(list)
    for class_value, room_value in zip(
        solution_rooms_df["class_id"].tolist(),
        solution_rooms_df["room_id"].tolist(),
    ):
        class_id = optional_int(class_value)
        room_id = optional_int(room_value)
        if class_id is not None and room_id is not None:
            rooms_by_class[class_id].append(room_id)

    times_by_class: dict[int, list[str]] = defaultdict(list)
    for class_value, time_value in zip(
        solution_times_df["class_id"].tolist(),
        solution_times_df["time_key"].tolist(),
    ):
        class_id = optional_int(class_value)
        if class_id is not None and pd.notna(time_value):
            times_by_class[class_id].append(str(time_value))

    # Count reference placements that would have been incorrectly rejected by
    # the old capacity >= class_limit rule.
    capacity_mismatches = 0
    for class_id, room_ids in rooms_by_class.items():
        limit = class_limit.get(class_id)
        if limit is None:
            continue
        for room_id in room_ids:
            capacity = room_capacity.get(room_id)
            if capacity is not None and capacity < limit:
                capacity_mismatches += 1

    missing_solution_time = sum(
        1 for class_id in classes if not times_by_class.get(class_id)
    )
    missing_solution_room = sum(
        1 for class_id in classes if not rooms_by_class.get(class_id)
    )

    room_slot_entries: dict[
        ResourceSlot, list[tuple[int, int, str]]
    ] = defaultdict(list)
    instructor_slot_entries: dict[
        ResourceSlot, list[tuple[int, int, str]]
    ] = defaultdict(list)

    for class_id in all_classes:
        for time_key in times_by_class.get(class_id, []):
            slots = occupied_slots(time_key)
            for room_id in rooms_by_class.get(class_id, []):
                if room_is_constrained.get(room_id, True):
                    for day_index, slot in slots:
                        room_slot_entries[(room_id, day_index, slot)].append(
                            (class_id, room_id, time_key)
                        )
            for instructor_id in instructors_by_class.get(class_id, []):
                for day_index, slot in slots:
                    instructor_slot_entries[(instructor_id, day_index, slot)].append(
                        (class_id, instructor_id, time_key)
                    )

    def count_reference_conflicts(
        slot_entries: dict[ResourceSlot, list[tuple[int, int, str]]]
    ) -> int:
        seen: set[tuple[int, int, int, int, str, str]] = set()
        conflicts = 0
        for entries in slot_entries.values():
            if len(entries) < 2:
                continue
            for a, b in combinations(entries, 2):
                class_a, resource_a, time_a = a
                class_b, resource_b, time_b = b
                if class_a == class_b:
                    continue
                key = (
                    min(class_a, class_b),
                    max(class_a, class_b),
                    resource_a,
                    resource_b,
                    min(time_a, time_b),
                    max(time_a, time_b),
                )
                if key in seen:
                    continue
                if date_masks_overlap(
                    class_dates.get(class_a), class_dates.get(class_b)
                ):
                    seen.add(key)
                    conflicts += 1
        return conflicts

    return {
        "reference_capacity_mismatches": capacity_mismatches,
        "reference_room_conflicts": count_reference_conflicts(room_slot_entries),
        "reference_instructor_conflicts": count_reference_conflicts(
            instructor_slot_entries
        ),
        "schedulable_missing_solution_time": missing_solution_time,
        "schedulable_missing_solution_room": missing_solution_room,
    }


def print_reference_diagnostics(diagnostics: dict[str, int]) -> None:
    print("\n=== Purdue Reference Solution Check ===")
    print(
        "Reference rooms with capacity < class_limit: "
        f"{diagnostics['reference_capacity_mismatches']:,}"
    )
    print(
        "Reference room conflicts under our overlap rule: "
        f"{diagnostics['reference_room_conflicts']:,}"
    )
    print(
        "Reference instructor conflicts under our overlap rule: "
        f"{diagnostics['reference_instructor_conflicts']:,}"
    )
    print(
        "Schedulable classes missing a reference time: "
        f"{diagnostics['schedulable_missing_solution_time']:,}"
    )
    print(
        "Schedulable classes missing a reference room: "
        f"{diagnostics['schedulable_missing_solution_room']:,}"
    )
    if (
        diagnostics["schedulable_missing_solution_time"] > 0
        and diagnostics["schedulable_missing_solution_room"] > 0
    ):
        print(
            "Note: the benchmark input file may not embed a solution for the "
            "schedulable classes; UniTime distributes solution files separately."
        )


def get_committed_solution_times(
    tables: dict[str, pd.DataFrame],
    committed_classes: list[int],
) -> dict[int, list[str]]:
    """Return the fixed time placements supplied for committed classes."""
    class_times_df = tables["class_times"]
    committed_set = set(committed_classes)
    solution_rows = class_times_df.loc[
        class_times_df["class_id"].isin(committed_set)
        & true_mask(class_times_df["is_solution"])
    ]

    times_by_class: dict[int, list[str]] = defaultdict(list)
    for class_value, time_value in zip(
        solution_rows["class_id"].tolist(),
        solution_rows["time_key"].tolist(),
    ):
        class_id = optional_int(class_value)
        if class_id is not None and pd.notna(time_value):
            times_by_class[class_id].append(str(time_value))

    return dict(times_by_class)


def add_student_conflict_objective(
    model: cp_model.CpModel,
    tables: dict[str, pd.DataFrame],
    classes: list[int],
    committed_classes: list[int],
    class_dates: dict[int, str | None],
    time_choice_vars: dict[tuple[int, str], cp_model.IntVar],
) -> dict[str, int]:
    """Minimize weighted student conflicts between class pairs.

    `class_pair_conflicts.csv` contains the number of students shared by each
    pair of classes. A pair incurs that weight when the chosen meeting patterns
    overlap on at least one common active date.

    Pairs of two schedulable classes receive one conflict Boolean. Pairs with
    one committed class use that class's fixed Purdue time directly.
    """
    pair_df = tables["class_pair_conflicts"]
    schedulable_set = set(classes)
    committed_set = set(committed_classes)
    relevant_classes = schedulable_set | committed_set
    committed_times = get_committed_solution_times(tables, committed_classes)

    times_by_schedulable_class: dict[int, list[str]] = defaultdict(list)
    for class_id, time_key in time_choice_vars:
        times_by_schedulable_class[class_id].append(time_key)

    objective_terms: list[cp_model.LinearExpr] = []
    pair_records = 0
    conflict_variables = 0
    conflict_link_constraints = 0
    fixed_time_penalty_terms = 0

    if pair_df.empty:
        model.minimize(0)
        return {
            "student_pair_records": 0,
            "student_conflict_variables": 0,
            "student_conflict_link_constraints": 0,
            "student_fixed_penalty_terms": 0,
        }

    for _, row in pair_df.iterrows():
        class_a = optional_int(row.get("class_id_a"))
        class_b = optional_int(row.get("class_id_b"))
        shared_students = optional_int(row.get("shared_students"))

        if (
            class_a is None
            or class_b is None
            or shared_students is None
            or shared_students <= 0
            or class_a not in relevant_classes
            or class_b not in relevant_classes
        ):
            continue

        # Classes active in disjoint portions of the term cannot conflict.
        if not date_masks_overlap(
            class_dates.get(class_a), class_dates.get(class_b)
        ):
            continue

        a_sched = class_a in schedulable_set
        b_sched = class_b in schedulable_set

        # At least one class must be a decision class; committed/committed
        # conflicts are constants and cannot be improved by this optimizer.
        if not a_sched and not b_sched:
            continue

        pair_records += 1

        if a_sched and b_sched:
            overlapping_choices: list[tuple[cp_model.IntVar, cp_model.IntVar]] = []
            for time_a in times_by_schedulable_class.get(class_a, []):
                var_a = time_choice_vars[(class_a, time_a)]
                for time_b in times_by_schedulable_class.get(class_b, []):
                    if meeting_patterns_overlap(time_a, time_b):
                        var_b = time_choice_vars[(class_b, time_b)]
                        overlapping_choices.append((var_a, var_b))

            if not overlapping_choices:
                continue

            conflict = model.new_bool_var(
                f"student_conflict_c{class_a}_c{class_b}"
            )
            conflict_variables += 1

            # If the selected times are an overlapping pair, conflict must be 1.
            # Positive objective weight then drives it to 0 whenever possible.
            for var_a, var_b in overlapping_choices:
                model.add(conflict >= var_a + var_b - 1)
                conflict_link_constraints += 1

            objective_terms.append(shared_students * conflict)
            continue

        sched_class = class_a if a_sched else class_b
        fixed_class = class_b if a_sched else class_a
        fixed_times = committed_times.get(fixed_class, [])
        if not fixed_times:
            # No fixed placement is available for the committed class, so this
            # pair cannot be scored reliably from the benchmark input.
            continue

        for sched_time in times_by_schedulable_class.get(sched_class, []):
            if any(
                meeting_patterns_overlap(sched_time, fixed_time)
                for fixed_time in fixed_times
            ):
                objective_terms.append(
                    shared_students * time_choice_vars[(sched_class, sched_time)]
                )
                fixed_time_penalty_terms += 1

    model.minimize(sum(objective_terms) if objective_terms else 0)

    return {
        "student_pair_records": pair_records,
        "student_conflict_variables": conflict_variables,
        "student_conflict_link_constraints": conflict_link_constraints,
        "student_fixed_penalty_terms": fixed_time_penalty_terms,
    }


def calculate_student_conflicts(
    schedule: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    lookups: dict[str, object],
) -> tuple[int, int, pd.DataFrame]:
    """Evaluate weighted student conflicts in a solved schedule."""
    pair_df = tables["class_pair_conflicts"]
    class_dates = lookups["class_dates"]
    committed_classes = lookups["committed_classes"]
    assert isinstance(class_dates, dict)
    assert isinstance(committed_classes, list)

    selected_times: dict[int, list[str]] = defaultdict(list)
    for class_value, time_value in zip(
        schedule.get("class_id", pd.Series(dtype=int)).tolist(),
        schedule.get("time_key", pd.Series(dtype=str)).tolist(),
    ):
        class_id = optional_int(class_value)
        if class_id is not None and pd.notna(time_value):
            selected_times[class_id].append(str(time_value))

    committed_set = set(committed_classes)
    committed_times = get_committed_solution_times(tables, committed_classes)
    for class_id, time_keys in committed_times.items():
        selected_times[class_id].extend(time_keys)

    weighted_conflicts = 0
    conflicting_pairs = 0
    rows: list[dict[str, object]] = []

    for _, row in pair_df.iterrows():
        class_a = optional_int(row.get("class_id_a"))
        class_b = optional_int(row.get("class_id_b"))
        shared_students = optional_int(row.get("shared_students"))
        if (
            class_a is None
            or class_b is None
            or shared_students is None
            or shared_students <= 0
        ):
            continue
        if class_a in committed_set and class_b in committed_set:
            continue

        times_a = selected_times.get(class_a, [])
        times_b = selected_times.get(class_b, [])
        if not times_a or not times_b:
            continue
        if not date_masks_overlap(
            class_dates.get(class_a), class_dates.get(class_b)
        ):
            continue

        overlap = any(
            meeting_patterns_overlap(time_a, time_b)
            for time_a in times_a
            for time_b in times_b
        )
        if not overlap:
            continue

        weighted_conflicts += shared_students
        conflicting_pairs += 1
        rows.append(
            {
                "class_id_a": class_a,
                "class_id_b": class_b,
                "shared_students": shared_students,
                "time_a": ";".join(times_a),
                "time_b": ";".join(times_b),
            }
        )

    conflict_df = pd.DataFrame(rows)
    if not conflict_df.empty:
        conflict_df = conflict_df.sort_values(
            "shared_students", ascending=False
        ).reset_index(drop=True)

    return weighted_conflicts, conflicting_pairs, conflict_df



def add_room_efficiency_objective(
    model: cp_model.CpModel,
    assignment_vars: dict[AssignmentKey, cp_model.IntVar],
    class_limit: dict[int, int | None],
    room_capacity: dict[int, int | None],
) -> dict[str, int]:
    """Minimize unused seats without considering student schedule conflicts.

    This is the comparison baseline. Because the objective depends only on the
    selected room, time choices are driven only by feasibility.
    """
    objective_terms: list[cp_model.LinearExpr] = []
    nonzero_terms = 0

    for (class_id, room_id, _time_key), var in assignment_vars.items():
        limit = class_limit.get(class_id)
        capacity = room_capacity.get(room_id)
        if limit is None or capacity is None:
            continue
        unused_seats = max(int(capacity) - int(limit), 0)
        if unused_seats > 0:
            objective_terms.append(unused_seats * var)
            nonzero_terms += 1

    model.minimize(sum(objective_terms) if objective_terms else 0)
    return {"room_waste_objective_terms": nonzero_terms}


def calculate_room_metrics(schedule: pd.DataFrame) -> tuple[int, float]:
    """Return total unused seats and mean class-level room utilization."""
    if schedule.empty:
        return 0, float("nan")

    valid = schedule.loc[
        schedule["class_limit"].notna()
        & schedule["room_capacity"].notna()
        & (schedule["room_capacity"] > 0)
    ].copy()
    if valid.empty:
        return 0, float("nan")

    unused = ((valid["room_capacity"] - valid["class_limit"]).clip(lower=0).sum())
    utilization = (valid["class_limit"] / valid["room_capacity"]).mean()
    return int(round(float(unused))), float(utilization)

def build_model(
    tables: dict[str, pd.DataFrame],
    objective_mode: ObjectiveMode,
) -> tuple[
    cp_model.CpModel,
    dict[AssignmentKey, cp_model.IntVar],
    dict[str, object],
    dict[str, int],
]:
    """Build the first full Purdue CP-SAT scheduling model."""
    lookups = build_lookup_tables(tables)

    classes = lookups["classes"]
    class_limit = lookups["class_limit"]
    room_capacity = lookups["room_capacity"]
    room_options = lookups["room_options"]
    time_options = lookups["time_options"]
    instructors_by_class = lookups["instructors_by_class"]
    committed_classes = lookups["committed_classes"]
    class_dates = lookups["class_dates"]
    room_is_constrained = lookups["room_is_constrained"]
    share_room_pairs = lookups["share_room_pairs"]

    # Narrow the object-typed lookup dictionary back to the concrete types used here.
    assert isinstance(classes, list)
    assert isinstance(class_limit, dict)
    assert isinstance(room_capacity, dict)
    assert isinstance(room_options, dict)
    assert isinstance(time_options, dict)
    assert isinstance(instructors_by_class, dict)
    assert isinstance(committed_classes, list)
    assert isinstance(class_dates, dict)
    assert isinstance(room_is_constrained, dict)
    assert isinstance(share_room_pairs, set)

    fixed_room_slots, fixed_instructor_slots = build_fixed_occupancy(
        tables,
        committed_classes,
        instructors_by_class,
        room_is_constrained,
    )

    model = cp_model.CpModel()
    assignment_vars: dict[AssignmentKey, cp_model.IntVar] = {}

    assignments_by_class: dict[int, list[cp_model.IntVar]] = defaultdict(list)
    assignments_by_class_time: dict[tuple[int, str], list[cp_model.IntVar]] = defaultdict(list)
    room_slot_vars: dict[
        ResourceSlot, list[tuple[int, AssignmentKey, cp_model.IntVar]]
    ] = defaultdict(list)
    instructor_slot_vars: dict[
        ResourceSlot, list[tuple[int, AssignmentKey, cp_model.IntVar]]
    ] = defaultdict(list)
    no_feasible_assignments: list[int] = []

    for class_id in classes:
        allowed_rooms = room_options.get(class_id, [])
        allowed_times = time_options.get(class_id, [])
        limit = class_limit.get(class_id)

        for room_id in allowed_rooms:
            capacity = room_capacity.get(room_id)

            # Do not independently reject a room because capacity < class_limit.
            # UniTime's class <room> list already contains only valid room options,
            # accounting for roomToLimitRatio and multi-room requirements.

            for time_key in allowed_times:
                if conflicts_with_committed(
                    class_id,
                    room_id,
                    time_key,
                    instructors_by_class,
                    class_dates,
                    room_is_constrained,
                    fixed_room_slots,
                    fixed_instructor_slots,
                    share_room_pairs,
                    class_limit,
                    room_capacity,
                ):
                    continue

                safe_time_name = str(time_key).replace("|", "_")
                var = model.new_bool_var(
                    f"x_c{class_id}_r{room_id}_t{safe_time_name}"
                )

                assignment_key = (class_id, room_id, str(time_key))
                assignment_vars[assignment_key] = var
                assignments_by_class[class_id].append(var)
                assignments_by_class_time[(class_id, str(time_key))].append(var)

                for day_index, slot in occupied_slots(str(time_key)):
                    if room_is_constrained.get(room_id, True):
                        room_slot_vars[(room_id, day_index, slot)].append(
                            (class_id, assignment_key, var)
                        )

                    for instructor_id in instructors_by_class.get(class_id, []):
                        instructor_slot_vars[
                            (instructor_id, day_index, slot)
                        ].append((class_id, assignment_key, var))

        if not assignments_by_class[class_id]:
            no_feasible_assignments.append(class_id)

    if no_feasible_assignments:
        examples = []
        for class_id in no_feasible_assignments[:20]:
            examples.append(
                f"class {class_id}: "
                f"{len(room_options.get(class_id, []))} room options, "
                f"{len(time_options.get(class_id, []))} time options, "
                f"class_limit={class_limit.get(class_id)}"
            )

        extra = "\n  ..." if len(no_feasible_assignments) > 20 else ""
        raise ValueError(
            "Some schedulable classes have no feasible room/time assignment:\n  "
            + "\n  ".join(examples)
            + extra
        )

    # 1) Every schedulable class gets exactly one room-time assignment.
    for class_id in classes:
        model.add_exactly_one(assignments_by_class[class_id])

    # Student-conflict optimization needs one Boolean per class-time choice.
    # The room-efficiency baseline does not need these extra variables.
    time_choice_vars: dict[tuple[int, str], cp_model.IntVar] = {}
    if objective_mode == "student_conflicts":
        for (class_id, time_key), room_vars in assignments_by_class_time.items():
            safe_time_name = str(time_key).replace("|", "_")
            time_var = model.new_bool_var(f"y_c{class_id}_t{safe_time_name}")
            model.add(time_var == sum(room_vars))
            time_choice_vars[(class_id, time_key)] = time_var

    # 2) Prevent room conflicts only when classes overlap in weekly time AND
    #    are active on at least one common date in the term. Rooms marked
    #    constraint=false in UniTime are intentionally allowed to overlap.
    room_overlap_constraints = add_date_aware_overlap_constraints(
        model,
        room_slot_vars,
        class_dates,
        share_room_pairs=share_room_pairs,
    )
    room_capacity_constraints = add_room_capacity_constraints(
        model, room_slot_vars, class_limit, room_capacity
    )

    # 3) All instructors listed on a Purdue class are required for that class.
    #    Apply the same date-aware overlap rule to instructor occupancy.
    instructor_overlap_constraints = add_date_aware_overlap_constraints(
        model, instructor_slot_vars, class_dates
    )

    # 4) Solve the same hard-constraint model under two different objectives.
    if objective_mode == "student_conflicts":
        objective_diagnostics = add_student_conflict_objective(
            model, tables, classes, committed_classes, class_dates, time_choice_vars
        )
    else:
        objective_diagnostics = add_room_efficiency_objective(
            model, assignment_vars, class_limit, room_capacity
        )

    diagnostics = {
        "assignment_variables": len(assignment_vars),
        "room_overlap_constraints": room_overlap_constraints,
        "room_capacity_constraints": room_capacity_constraints,
        "instructor_overlap_constraints": instructor_overlap_constraints,
        "share_room_pairs": len(share_room_pairs),
        "fixed_room_slots": len(fixed_room_slots),
        "fixed_instructor_slots": len(fixed_instructor_slots),
        "time_choice_variables": len(time_choice_vars),
        **objective_diagnostics,
    }

    return model, assignment_vars, lookups, diagnostics


def extract_schedule(
    solver: cp_model.CpSolver,
    assignment_vars: dict[AssignmentKey, cp_model.IntVar],
    lookups: dict[str, object],
) -> pd.DataFrame:
    """Convert the selected assignment variables into a readable DataFrame."""
    class_limit = lookups["class_limit"]
    room_capacity = lookups["room_capacity"]
    instructors_by_class = lookups["instructors_by_class"]
    time_details = lookups["time_details"]
    class_dates = lookups["class_dates"]

    assert isinstance(class_limit, dict)
    assert isinstance(room_capacity, dict)
    assert isinstance(instructors_by_class, dict)
    assert isinstance(time_details, dict)
    assert isinstance(class_dates, dict)

    rows: list[dict[str, object]] = []

    for (class_id, room_id, time_key), var in assignment_vars.items():
        if not solver.boolean_value(var):
            continue

        detail = time_details.get(time_key, {})
        limit = class_limit.get(class_id)
        capacity = room_capacity.get(room_id)

        utilization: float | None = None
        if limit is not None and capacity not in (None, 0):
            utilization = float(limit) / float(capacity)

        rows.append(
            {
                "class_id": class_id,
                "instructor_ids": ";".join(
                    str(instructor_id)
                    for instructor_id in instructors_by_class.get(class_id, [])
                ),
                "room_id": room_id,
                "days": detail.get("days"),
                "start_time": detail.get("start_time"),
                "end_time": detail.get("end_time"),
                "time_key": time_key,
                "dates_mask": class_dates.get(class_id),
                "class_limit": limit,
                "room_capacity": capacity,
                "room_utilization": utilization,
            }
        )

    schedule = pd.DataFrame(rows)

    if not schedule.empty:
        schedule = schedule.sort_values(
            by=["days", "start_time", "room_id", "class_id"],
            na_position="last",
        ).reset_index(drop=True)

    return schedule


def print_summary(
    label: str,
    objective_mode: ObjectiveMode,
    status: cp_model.CpSolverStatus,
    solver: cp_model.CpSolver,
    schedule: pd.DataFrame,
    lookups: dict[str, object],
    diagnostics: dict[str, int],
) -> None:
    classes = lookups["classes"]
    assert isinstance(classes, list)

    print(f"\n=== {label} ===")
    print(f"Solver status: {solver.status_name(status)}")
    print(f"Schedulable classes: {len(classes):,}")
    print(f"Assignment variables: {diagnostics['assignment_variables']:,}")
    if diagnostics.get("time_choice_variables", 0):
        print(f"Time-choice variables: {diagnostics['time_choice_variables']:,}")
    print(f"Room overlap constraints: {diagnostics['room_overlap_constraints']:,}")
    print(f"Room capacity constraints: {diagnostics['room_capacity_constraints']:,}")
    print(f"Room-sharing class pairs: {diagnostics['share_room_pairs']:,}")
    print(f"Instructor overlap constraints: {diagnostics['instructor_overlap_constraints']:,}")
    print(f"Committed room-slots blocked: {diagnostics['fixed_room_slots']:,}")
    print(f"Committed instructor-slots blocked: {diagnostics['fixed_instructor_slots']:,}")

    if objective_mode == "student_conflicts":
        print(f"Student-pair records modeled: {diagnostics['student_pair_records']:,}")
        print(f"Student-conflict variables: {diagnostics['student_conflict_variables']:,}")
        print(f"Student-conflict link constraints: {diagnostics['student_conflict_link_constraints']:,}")
        print(f"Student fixed-time penalty terms: {diagnostics['student_fixed_penalty_terms']:,}")
    else:
        print(f"Room-waste objective terms: {diagnostics.get('room_waste_objective_terms', 0):,}")

    print(f"Solve time: {solver.wall_time:.2f} seconds")
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Classes scheduled: {len(schedule):,}")
        objective_name = "weighted student conflicts" if objective_mode == "student_conflicts" else "unused seats"
        print(f"Objective ({objective_name}): {solver.objective_value:,.0f}")
        print(f"Best objective bound: {solver.best_objective_bound:,.0f}")



def solve_one_model(
    tables: dict[str, pd.DataFrame],
    objective_mode: ObjectiveMode,
    label: str,
    schedule_output_path: Path,
    conflict_output_path: Path,
    max_time_seconds: float,
    workers: int,
) -> dict[str, Any] | None:
    """Solve one objective using the shared hard-constraint formulation."""
    model, assignment_vars, lookups, diagnostics = build_model(tables, objective_mode)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = 42
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print_summary(label, objective_mode, status, solver, pd.DataFrame(), lookups, diagnostics)
        if status == cp_model.INFEASIBLE:
            print("The model is infeasible under the current constraints.")
        else:
            print("No feasible schedule was found within the solve limit.")
        return None

    schedule = extract_schedule(solver, assignment_vars, lookups)
    weighted_conflicts, conflicting_pairs, conflict_df = calculate_student_conflicts(schedule, tables, lookups)
    total_unused_seats, average_room_utilization = calculate_room_metrics(schedule)

    schedule_output_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(schedule_output_path, index=False)
    conflict_df.to_csv(conflict_output_path, index=False)

    print_summary(label, objective_mode, status, solver, schedule, lookups, diagnostics)
    print(f"Verified weighted student conflicts: {weighted_conflicts:,}")
    print(f"Conflicting class pairs: {conflicting_pairs:,}")
    print(f"Total unused seats: {total_unused_seats:,}")
    print(f"Average room utilization: {average_room_utilization:.1%}")
    print(f"Schedule written to: {schedule_output_path}")
    print(f"Conflict report written to: {conflict_output_path}")

    return {
        "model": label,
        "objective_mode": objective_mode,
        "status": solver.status_name(status),
        "classes_scheduled": len(schedule),
        "weighted_student_conflicts": weighted_conflicts,
        "conflicting_class_pairs": conflicting_pairs,
        "total_unused_seats": total_unused_seats,
        "average_room_utilization_pct": average_room_utilization * 100.0,
        "solve_time_seconds": solver.wall_time,
        "objective_value": solver.objective_value,
        "best_objective_bound": solver.best_objective_bound,
    }


def run_comparison(
    data_dir: Path,
    output_dir: Path,
    max_time_seconds: float = 120.0,
    workers: int = 8,
) -> pd.DataFrame | None:
    """Run the room-efficiency baseline and student-focused model back to back."""
    tables = load_processed_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning room-efficiency baseline...")
    baseline = solve_one_model(
        tables, "room_efficiency", "Room-Efficiency Baseline",
        output_dir / "baseline_room_efficiency_schedule.csv",
        output_dir / "baseline_student_conflicts.csv",
        max_time_seconds, workers,
    )

    print("\nRunning student-conflict optimization...")
    student = solve_one_model(
        tables, "student_conflicts", "Student-Conflict Optimized",
        output_dir / "student_optimized_schedule.csv",
        output_dir / "student_optimized_conflicts.csv",
        max_time_seconds, workers,
    )

    if baseline is None or student is None:
        print("\nComparison could not be completed because one model did not solve.")
        return None

    comparison = pd.DataFrame([baseline, student])
    baseline_conflicts = int(baseline["weighted_student_conflicts"])
    student_conflicts = int(student["weighted_student_conflicts"])
    reduction_pct = ((baseline_conflicts - student_conflicts) / baseline_conflicts * 100.0) if baseline_conflicts > 0 else 0.0
    utilization_change = float(student["average_room_utilization_pct"]) - float(baseline["average_room_utilization_pct"])

    comparison["student_conflict_reduction_vs_baseline_pct"] = [0.0, reduction_pct]
    comparison["room_utilization_change_vs_baseline_pp"] = [0.0, utilization_change]
    comparison_path = output_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    display_columns = [
        "model", "status", "classes_scheduled", "weighted_student_conflicts",
        "conflicting_class_pairs", "total_unused_seats",
        "average_room_utilization_pct", "solve_time_seconds",
    ]
    pretty = comparison[display_columns].copy().rename(columns={
        "model": "Model", "status": "Status", "classes_scheduled": "Classes",
        "weighted_student_conflicts": "Weighted conflicts",
        "conflicting_class_pairs": "Conflict pairs", "total_unused_seats": "Unused seats",
        "average_room_utilization_pct": "Room util. %", "solve_time_seconds": "Solve sec.",
    })
    pretty["Room util. %"] = pretty["Room util. %"].map(lambda x: f"{x:.1f}")
    pretty["Solve sec."] = pretty["Solve sec."].map(lambda x: f"{x:.2f}")

    print("\n=== Model Comparison ===")
    print(pretty.to_string(index=False))
    print(f"\nStudent-conflict reduction versus the room-efficiency baseline: {reduction_pct:.1f}%")
    print(f"Change in average room utilization: {utilization_change:+.1f} percentage points")
    print(f"Comparison written to: {comparison_path}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a room-efficiency baseline with a student-conflict optimized Purdue course schedule."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data") / "processed",
        help="Directory containing the processed Purdue CSV files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Directory for schedules, conflict reports, and comparison CSV.",
    )
    parser.add_argument(
        "--max-time", type=float, default=120.0,
        help="Maximum CP-SAT solve time per model in seconds.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of CP-SAT search workers.",
    )
    args = parser.parse_args()
    run_comparison(args.data_dir, args.output_dir, args.max_time, args.workers)


if __name__ == "__main__":
    main()
