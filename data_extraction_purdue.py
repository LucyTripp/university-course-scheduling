from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse

import pandas as pd


DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _as_int(value):
    """Convert an XML attribute to int when present; otherwise return None."""
    return None if value is None else int(value)


def _as_float(value):
    """Convert an XML attribute to float when present; otherwise return None."""
    return None if value is None else float(value)


def _as_bool(value):
    """Convert 'true'/'false' XML text to bool when present; otherwise return None."""
    if value is None:
        return None
    return value.lower() == "true"


def _slot_to_clock(slot: int, slot_minutes: int = 5) -> str:
    """Convert a UniTime slot number to HH:MM."""
    total_minutes = slot * slot_minutes
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def _days_to_names(days_mask: str) -> str:
    """Convert a seven-character day mask like 1010100 to 'Mon Wed Fri'."""
    return " ".join(
        day for day, flag in zip(DAY_NAMES, days_mask)
        if flag == "1"
    )


def load_purdue_timetabling_xml(xml_path: str | Path) -> dict[str, pd.DataFrame]:
    """
    Parse a UniTime/Purdue course timetabling XML file into normalized pandas DataFrames.

    The output is intentionally normalized rather than forcing the hierarchical XML into
    one giant table. This makes it easier to build optimization sets, parameters, and
    constraints from the data.

    Returns a dictionary containing:
        metadata
        rooms
        classes
        class_instructors
        class_room_options
        class_time_options
        group_constraints
        constraint_classes
        students
        student_offerings
        student_classes
        student_prohibited_classes
        class_pair_conflicts
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    slot_minutes = 24 * 60 // int(root.attrib.get("slotsPerDay", 288))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata_df = pd.DataFrame([root.attrib])

    # ------------------------------------------------------------------
    # Rooms
    # ------------------------------------------------------------------
    room_rows = []
    for room in root.findall("rooms/room"):
        location = room.get("location")
        location_x = location_y = None
        if location and "," in location:
            x, y = location.split(",", 1)
            location_x = _as_int(x)
            location_y = _as_int(y)

        room_rows.append(
            {
                "room_id": _as_int(room.get("id")),
                "capacity": _as_int(room.get("capacity")),
                "constraint_room": _as_bool(room.get("constraint")),
                "location_x": location_x,
                "location_y": location_y,
                "has_sharing_rules": room.find("sharing") is not None,
            }
        )

    rooms_df = pd.DataFrame(room_rows)

    # ------------------------------------------------------------------
    # Classes + their instructor/room/time options
    # ------------------------------------------------------------------
    class_rows = []
    instructor_rows = []
    room_option_rows = []
    time_option_rows = []

    classes_element = root.find("classes")
    for class_element in root.findall("classes/class"):
        class_id = _as_int(class_element.get("id"))

        class_rows.append(
            {
                "class_id": class_id,
                "offering_id": _as_int(class_element.get("offering")),
                "config_id": _as_int(class_element.get("config")),
                "subpart_id": _as_int(class_element.get("subpart")),
                "parent_class_id": _as_int(class_element.get("parent")),
                "class_limit": _as_int(class_element.get("classLimit")),
                "scheduler_id": _as_int(class_element.get("scheduler")),
                "committed": _as_bool(class_element.get("committed")),
                "dates_mask": class_element.get("dates"),
                "room_to_limit_ratio": _as_float(class_element.get("roomToLimitRatio")),
            }
        )

        for instructor in class_element.findall("instructor"):
            instructor_rows.append(
                {
                    "class_id": class_id,
                    "instructor_id": _as_int(instructor.get("id")),
                    "is_solution": _as_bool(instructor.get("solution")) or False,
                }
            )

        for option_number, room in enumerate(class_element.findall("room"), start=1):
            room_option_rows.append(
                {
                    "class_id": class_id,
                    "room_option_number": option_number,
                    "room_id": _as_int(room.get("id")),
                    "room_preference": _as_float(room.get("pref")),
                    "is_solution": _as_bool(room.get("solution")) or False,
                }
            )

        for option_number, time in enumerate(class_element.findall("time"), start=1):
            days_mask = time.get("days")
            start_slot = _as_int(time.get("start"))
            length_slots = _as_int(time.get("length"))
            duration_minutes = None if length_slots is None else length_slots * slot_minutes
            end_slot = None if start_slot is None or length_slots is None else start_slot + length_slots

            # time_key is global: identical meeting patterns receive the same key
            time_key = f"{days_mask}|{start_slot}|{length_slots}"

            time_option_rows.append(
                {
                    "class_id": class_id,
                    "time_option_number": option_number,
                    "time_key": time_key,
                    "days_mask": days_mask,
                    "days": _days_to_names(days_mask) if days_mask else None,
                    "start_slot": start_slot,
                    "start_time": _slot_to_clock(start_slot, slot_minutes) if start_slot is not None else None,
                    "length_slots": length_slots,
                    "duration_minutes": duration_minutes,
                    "end_time": _slot_to_clock(end_slot, slot_minutes) if end_slot is not None else None,
                    "time_preference": _as_float(time.get("pref")),
                    "is_solution": _as_bool(time.get("solution")) or False,
                }
            )

    classes_df = pd.DataFrame(class_rows)
    class_instructors_df = pd.DataFrame(instructor_rows)
    class_room_options_df = pd.DataFrame(room_option_rows)
    class_time_options_df = pd.DataFrame(time_option_rows)

    # ------------------------------------------------------------------
    # Group constraints
    # ------------------------------------------------------------------
    group_constraint_rows = []
    constraint_class_rows = []

    for row_number, constraint in enumerate(root.findall("groupConstraints/constraint"), start=1):
        # row_number is necessary because the XML can reuse the same constraint id.
        constraint_key = row_number
        group_constraint_rows.append(
            {
                "constraint_key": constraint_key,
                "constraint_id": _as_int(constraint.get("id")),
                "constraint_type": constraint.get("type"),
                "preference": constraint.get("pref"),
                "preference_numeric": (
                    _as_float(constraint.get("pref"))
                    if constraint.get("pref") not in (None, "R")
                    else None
                ),
                "required": constraint.get("pref") == "R",
            }
        )

        for class_element in constraint.findall("class"):
            constraint_class_rows.append(
                {
                    "constraint_key": constraint_key,
                    "class_id": _as_int(class_element.get("id")),
                }
            )

    group_constraints_df = pd.DataFrame(group_constraint_rows)
    constraint_classes_df = pd.DataFrame(constraint_class_rows)

    # ------------------------------------------------------------------
    # Students
    # ------------------------------------------------------------------
    student_rows = []
    student_offering_rows = []
    student_class_rows = []
    student_prohibited_rows = []

    for student in root.findall("students/student"):
        student_id = _as_int(student.get("id"))
        student_rows.append({"student_id": student_id})

        for offering in student.findall("offering"):
            student_offering_rows.append(
                {
                    "student_id": student_id,
                    "offering_id": _as_int(offering.get("id")),
                    "weight": _as_float(offering.get("weight")),
                }
            )

        for class_element in student.findall("class"):
            student_class_rows.append(
                {
                    "student_id": student_id,
                    "class_id": _as_int(class_element.get("id")),
                }
            )

        for class_element in student.findall("prohibited-class"):
            student_prohibited_rows.append(
                {
                    "student_id": student_id,
                    "class_id": _as_int(class_element.get("id")),
                }
            )

    students_df = pd.DataFrame(student_rows)
    student_offerings_df = pd.DataFrame(student_offering_rows)
    student_classes_df = pd.DataFrame(student_class_rows)
    student_prohibited_classes_df = pd.DataFrame(student_prohibited_rows)

    # ------------------------------------------------------------------
    # Derived table: how many students create a conflict between each pair
    # of classes. This is useful for a student-conflict objective.
    # ------------------------------------------------------------------
    pair_counts = defaultdict(int)
    classes_by_student = student_classes_df.groupby("student_id")["class_id"].apply(list)

    for class_ids in classes_by_student:
        unique_class_ids = sorted(set(class_ids))
        for class_a, class_b in combinations(unique_class_ids, 2):
            pair_counts[(class_a, class_b)] += 1

    class_pair_conflicts_df = pd.DataFrame(
        [
            {
                "class_id_a": class_a,
                "class_id_b": class_b,
                "shared_students": count,
            }
            for (class_a, class_b), count in pair_counts.items()
        ]
    )

    if not class_pair_conflicts_df.empty:
        class_pair_conflicts_df = class_pair_conflicts_df.sort_values(
            "shared_students", ascending=False
        ).reset_index(drop=True)

    return {
        "metadata": metadata_df,
        "rooms": rooms_df,
        "classes": classes_df,
        "class_instructors": class_instructors_df,
        "class_room_options": class_room_options_df,
        "class_time_options": class_time_options_df,
        "group_constraints": group_constraints_df,
        "constraint_classes": constraint_classes_df,
        "students": students_df,
        "student_offerings": student_offerings_df,
        "student_classes": student_classes_df,
        "student_prohibited_classes": student_prohibited_classes_df,
        "class_pair_conflicts": class_pair_conflicts_df,
    }


def build_optimizer_inputs(tables: dict[str, pd.DataFrame]) -> dict:
    """
    Build convenient Python objects for the OR-Tools model.

    This adapter does NOT invent missing instructors. Purdue classes with a known
    instructor get that instructor as their allowed/fixed instructor set. Classes
    without instructor data receive an empty list so the optimization model can
    decide how it wants to handle unknown instructors.

    Only committed=False classes are treated as schedulable classes.
    """
    classes_df = tables["classes"]
    rooms_df = tables["rooms"]
    instructors_df = tables["class_instructors"]
    room_options_df = tables["class_room_options"]
    time_options_df = tables["class_time_options"]

    schedulable = classes_df.loc[classes_df["committed"] == False].copy()  # noqa: E712
    schedulable_ids = set(schedulable["class_id"])

    classes = sorted(schedulable_ids)
    rooms = sorted(rooms_df["room_id"].dropna().astype(int).unique().tolist())

    instructor_options = (
        instructors_df[instructors_df["class_id"].isin(schedulable_ids)]
        .groupby("class_id")["instructor_id"]
        .apply(lambda s: sorted(set(s.dropna().astype(int))))
        .to_dict()
    )
    instructor_options = {c: instructor_options.get(c, []) for c in classes}

    all_instructors = sorted(
        {
            instructor
            for options in instructor_options.values()
            for instructor in options
        }
    )

    room_options = (
        room_options_df[room_options_df["class_id"].isin(schedulable_ids)]
        .groupby("class_id")["room_id"]
        .apply(lambda s: sorted(set(s.dropna().astype(int))))
        .to_dict()
    )
    room_options = {c: room_options.get(c, []) for c in classes}

    time_options = (
        time_options_df[time_options_df["class_id"].isin(schedulable_ids)]
        .groupby("class_id")["time_key"]
        .apply(lambda s: sorted(set(s.dropna())))
        .to_dict()
    )
    time_options = {c: time_options.get(c, []) for c in classes}

    all_times = sorted(
        {
            time_key
            for options in time_options.values()
            for time_key in options
        }
    )

    class_limit = (
        schedulable.set_index("class_id")["class_limit"]
        .dropna()
        .astype(int)
        .to_dict()
    )

    room_capacity = (
        rooms_df.set_index("room_id")["capacity"]
        .dropna()
        .astype(int)
        .to_dict()
    )

    # Explicit columns avoid ambiguous itertuples() attribute types in Pylance.
    room_preference = {}
    valid_rooms = room_options_df.loc[
        room_options_df["class_id"].isin(schedulable_ids)
    ].dropna(subset=["class_id", "room_id", "room_preference"])
    for class_id, room_id, preference in zip(
        valid_rooms["class_id"].tolist(),
        valid_rooms["room_id"].tolist(),
        valid_rooms["room_preference"].tolist(),
    ):
        room_preference[(int(class_id), int(room_id))] = float(preference)

    time_preference = {}
    valid_times = time_options_df.loc[
        time_options_df["class_id"].isin(schedulable_ids)
    ].dropna(subset=["class_id", "time_key", "time_preference"])
    for class_id, time_key, preference in zip(
        valid_times["class_id"].tolist(),
        valid_times["time_key"].tolist(),
        valid_times["time_preference"].tolist(),
    ):
        time_preference[(int(class_id), str(time_key))] = float(preference)

    return {
        "classes": classes,
        "instructors": all_instructors,
        "rooms": rooms,
        "times": all_times,
        "class_limit": class_limit,
        "room_capacity": room_capacity,
        "instructor_options": instructor_options,
        "room_options": room_options,
        "time_options": time_options,
        "room_preference": room_preference,
        "time_preference": time_preference,
    }


def export_tables_to_csv(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Write every DataFrame to a CSV so it can be inspected in Excel/VS Code."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, dataframe in tables.items():
        dataframe.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Purdue/UniTime XML into normalized CSV tables.")
    parser.add_argument("--xml", type=Path, default=Path("data/pu-fal07-ms.xml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    xml_file = args.xml

    tables = load_purdue_timetabling_xml(xml_file)
    build_optimizer_inputs(tables)

    print("Loaded Purdue timetabling data")
    print(f"Total classes: {len(tables['classes']):,}")
    print(
        "Schedulable classes:",
        int((tables["classes"]["committed"] == False).sum()),  # noqa: E712
    )
    print(f"Rooms: {len(tables['rooms']):,}")
    print(f"Known instructor-class links: {len(tables['class_instructors']):,}")
    print(f"Room options: {len(tables['class_room_options']):,}")
    print(f"Time options: {len(tables['class_time_options']):,}")
    print(f"Students: {len(tables['students']):,}")
    print(f"Class-pair conflict records: {len(tables['class_pair_conflicts']):,}")

    export_tables_to_csv(tables, args.output_dir)


if __name__ == "__main__":
    main()



