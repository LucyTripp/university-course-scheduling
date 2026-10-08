import pandas as pd
import pytest
from ortools.sat.python import cp_model
from full_scheduler import (
    meeting_patterns_overlap, date_masks_overlap,
    add_date_aware_overlap_constraints, add_room_capacity_constraints,
    add_student_conflict_objective,
)


@pytest.mark.parametrize('a,b,expected', [
    ('1000000|100|12', '1000000|111|12', True),
    ('1000000|100|12', '1000000|112|12', False),
    ('1000000|100|12', '0100000|100|12', False),
])
def test_weekly_overlap(a, b, expected):
    assert meeting_patterns_overlap(a, b) is expected


def test_date_overlap():
    assert not date_masks_overlap('1000', '0010')
    assert date_masks_overlap('1100', '0100')
    assert date_masks_overlap(None, '0010')


@pytest.mark.parametrize('sharing,total,feasible', [(False, 20, False), (True, 20, True), (True, 31, False)])
def test_room_sharing_and_capacity(sharing, total, feasible):
    model = cp_model.CpModel()
    a, b = model.new_bool_var('a'), model.new_bool_var('b')
    entries = {(7, 0, 100): [(1, (1, 7, '1000000|100|1'), a), (2, (2, 7, '1000000|100|1'), b)]}
    add_date_aware_overlap_constraints(model, entries, {1: '1', 2: '1'}, {(1, 2)} if sharing else set())
    add_room_capacity_constraints(model, entries, {1: 10, 2: total - 10}, {7: 30})
    model.add(a == 1)
    model.add(b == 1)
    status = cp_model.CpSolver().solve(model)
    assert (status == cp_model.OPTIMAL) is feasible


def test_student_objective_avoids_shared_student_conflict():
    model = cp_model.CpModel()
    t = '1000000|100|12'
    u = '1000000|112|12'
    a, b, c = [model.new_bool_var(n) for n in ['a', 'b', 'c']]
    model.add(a == 1)
    model.add_exactly_one([b, c])
    tables = {
        'class_pair_conflicts': pd.DataFrame([{'class_id_a': 1, 'class_id_b': 2, 'shared_students': 5}]),
        'class_times': pd.DataFrame(columns=['class_id', 'is_solution', 'time_key']),
    }
    add_student_conflict_objective(model, tables, [1, 2], [], {1: '1', 2: '1'}, {(1, t): a, (2, t): b, (2, u): c})
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert solver.value(c) == 1
    assert solver.objective_value == 0
