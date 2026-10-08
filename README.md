# University Course Scheduling Project

A Python course-timetabling project using **Google OR-Tools CP-SAT** to compare classroom efficiency with student scheduling conflicts. Each of 525 schedulable class sections receives one allowed room and weekly meeting pattern while respecting the resource constraints implemented by the model.

I interviewed a BYU Mathematics Department associate chair to understand scheduling priorities, including student experience and classroom use. This portfolio model uses Purdue benchmark data; it is not a deployed BYU scheduling system.

## Verified results

The following run used the supplied processed CSV dataset on October 8, 2026, Python 3.12, pandas 2.2.3, OR-Tools 9.15.6755, eight solver workers, and a 120-second solver limit per model.

| Metric | Room-efficiency baseline | Student-conflict model |
| --- | ---: | ---: |
| Classes scheduled | 525 | 525 |
| Weighted student conflicts | 2,071 | 152 |
| Conflicting class pairs | 1,016 | 128 |
| Average room utilization | 85.5% | 83.9% |
| Total unused seats | 3,239 | 3,796 |
| Solver status | OPTIMAL | OPTIMAL |
| Solver wall time | 8.00 s | 9.87 s |

This run reduced weighted conflicts **92.7%**, with a **1.6 percentage-point** decrease in average room utilization. Full precision, objective bounds, schedules, and conflict reports are in [`results/`](results/). Solver times exclude Python model construction and can vary with hardware. Optimality applies to this formulation and objective, not to all UniTime requirements.

The room-efficiency objective has tied optimal schedules with different student-conflict scores. Earlier runs reported 2,023 baseline conflicts; a particular baseline conflict count or reduction percentage is not guaranteed on every rerun. Both models use the same hard constraints, but different objectives.

## Dataset

The source is the **Purdue Fall 2007 Mathematics & Statistics** instance from the **Purdue C8-2007 Data Sets** section of the [UniTime benchmark page](https://www.unitime.org/uct_datasets.php).

**Use this exact download:** [Purdue C8-2007 Fall Mathematics & Statistics ZIP](https://www.unitime.org/data/josh/pu-fal07-ms.zip). Its XML creation timestamp is `Thu Feb 14 22:17:36 EST 2008`. The same page also lists an older download with the same filename under “Fall 2007 Data Sets”; that file differs and will not reproduce this project.

The matching XML produces 1,284 total class records (525 schedulable and 759 committed), 14,331 students, and 38,646 unique class-pair records. Of those pairs, 22,660 involve at least one schedulable class. All 13 extracted tables were checked against the supplied CSVs and matched in content; ties in the class-pair weight sort can change row order without changing the data.

The project package includes the matching XML and local processed CSVs. Git excludes these third-party inputs; readers can download the source above and regenerate the tables. No claim is made that the project's code license covers benchmark data.

## Setup and run

Python 3.12 is the tested version. Run commands from the project root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

With the supplied processed CSVs in `data/processed/`, run the comparison:

```powershell
python full_scheduler.py --data-dir data/processed --output-dir outputs --max-time 120 --workers 8
```

To regenerate the processed CSVs from the matching XML:

```powershell
python data_extraction_purdue.py --xml data/pu-fal07-ms.xml --output-dir data/processed
python full_scheduler.py --data-dir data/processed --output-dir outputs --max-time 120 --workers 8
```

The extractor supports arbitrary input/output paths and exports all normalized tables. Row ordering and tied baseline schedules may vary; see the results discussion above.

## Data and model

The XML stores rooms, classes, allowed room/time options, known instructors, group constraints, student records, and committed placements in a nested structure. The extractor converts it to separate CSV tables. It counts each student's unique class pair once and aggregates those pairs into a `shared_students` weight.

The scheduler loads eight tables: classes, rooms, class room options, class time options, class instructors, group constraints, constraint classes, and class-pair conflicts. It creates Boolean variables for allowed class-room-time assignments and selects exactly one per schedulable class.

Implemented resource rules include:

- Room overlap restrictions for rooms marked `constraint=true`.
- Known instructor overlap restrictions; listed instructors are fixed requirements, not decision variables.
- Blocking against committed placements with available solution flags.
- Class date masks when checking pairwise overlap.
- Room-sharing exceptions from required `CAN_SHARE_ROOM`, `MEET_WITH`, and `MEET_WITH_WEEKS` groups, with a capacity constraint on schedulable room occupancy.

The baseline minimizes unused seats, computed from class limits and room capacities. The student objective minimizes the sum of shared-student weights for overlapping class pairs involving at least one schedulable class. Conflicts between two committed classes are excluded because this model cannot change them. Conflicts with committed classes lacking fixed time data cannot be reliably scored and are omitted.

A weighted conflict counts a student-class-pair overlap. It is not a count of unique affected students: one student can contribute multiple conflicting pairs. Room utilization is the unweighted average of class limit divided by assigned room capacity, not observed attendance or utilization over time.

## Scope and limitations

This is a simplified portfolio formulation, not a complete UniTime implementation. General group constraints, back-to-back travel distances, room-sharing availability calendars, preference penalties, multi-room placements, student section reassignment, and missing instructor availability are not fully modeled. Recognizing a `MEET_WITH` group as a room-sharing exception does not force its members to meet together.

Room-capacity sums operate on weekly resource slots and can be overly restrictive for classes meeting on disjoint term dates. Capacity accounting for committed sharing placements is checked candidate by candidate, rather than as one complete cumulative constraint including all selected and committed sections. These limitations matter when applying the model to other instances.

Allowed room lists come from UniTime, but this simplified single-room formulation does not fully implement `roomToLimitRatio` or multi-room requirements. Do not infer that every source requirement is satisfied from the solver's OPTIMAL status.

## Validation

Nine automated tests cover weekly overlap and touching endpoints, disjoint meeting days, date masks, permitted room sharing, capacity limits, student-conflict avoidance, and XML extraction with duplicate enrollment records and preferences. GitHub Actions runs them on pushes and pull requests.

The supplied CSV comparison was also checked for class coverage, allowed placements, modeled room and instructor overlaps, room capacity, committed-placement blocking, and recalculated conflict scores. These checks validate the implemented rules, subject to the limitations above.

## Files

| Path | Purpose |
| --- | --- |
| `data_extraction_purdue.py` | XML parser, normalized CSV export, optimizer input adapter |
| `full_scheduler.py` | Shared hard constraints, both objectives, comparison CLI |
| `requirements.txt` | Tested runtime dependencies |
| `requirements-dev.txt` | Runtime dependencies plus pytest |
| `tests/` | Small synthetic correctness tests |
| `.github/workflows/tests.yml` | Automated test workflow |
| `results/` | Verified CSV comparison, schedules, conflict reports, solver log |
| `data/README.md` | Dataset attribution and local input instructions |
| `docs/PUBLISHING.md` | Instructions for creating the GitHub repository |

## AI assistance disclosure

I used ChatGPT to assist with code development, debugging, and documentation for this project. My work included interviewing a university associate chair about scheduling priorities and interpreting the optimization results. AI assistance contributed to implementation and revisions of the data-processing and scheduling code.

## Attribution and license

Benchmark source: [UniTime University Course Timetabling Benchmark Datasets](https://www.unitime.org/uct_datasets.php), including the Purdue Fall 2007 Mathematics & Statistics instances. Format reference: [UniTime XML format v2.4](https://www.unitime.org/uct_dataformat_v24.php). The interview provides project context; BYU and Purdue have not endorsed this implementation.

No open-source code license has been selected yet. The benchmark data are third-party material and are not covered by any future code license for this repository.
