import pandas as pd
from data_extraction_purdue import build_optimizer_inputs, load_purdue_timetabling_xml


def test_duplicate_enrollments_count_once_and_extract_preferences(tmp_path):
    xml = '''<timetable slotsPerDay="288"><rooms><room id="1" capacity="30"/></rooms>
    <classes><class id="1" committed="false" classLimit="10"><instructor id="1"/>
    <room id="1" pref="2"/><time days="1000000" start="100" length="12" pref="3"/></class>
    <class id="2" committed="false" classLimit="10"><room id="1" pref="0"/>
    <time days="0100000" start="100" length="12" pref="0"/></class></classes>
    <students><student id="1"><class id="1"/><class id="1"/><class id="2"/></student>
    <student id="2"><class id="1"/><class id="2"/></student></students></timetable>'''
    path = tmp_path / 'tiny.xml'
    path.write_text(xml)
    tables = load_purdue_timetabling_xml(path)
    assert tables['class_pair_conflicts'].to_dict('records') == [{'class_id_a': 1, 'class_id_b': 2, 'shared_students': 2}]
    adapter = build_optimizer_inputs(tables)
    assert adapter['room_preference'][(1, 1)] == 2.0
    assert adapter['time_preference'][(1, '1000000|100|12')] == 3.0
