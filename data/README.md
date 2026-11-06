# Data

The analyses read one Excel file, `data/ER_A.xlsx` (another location can be given with the environment
variable `EYERIDERS_DATA`). It is the de-identified export of the Eye-Riders A game: one row per completed
level (254 rows = 128 children x 2 levels, minus 2 children who did not complete Level 2). It was created
from the original export with `tools/anonymize_data.py`, which replaced the platform identifier of the
child with a code and removed the session identifier and the date and time of play.

| Column | Content |
|---|---|
| `child_id` | child code, `C001`-`C128` |
| `scheduleId` | order group: 21 = Level 1 played first, 22 = Level 2 played first |
| `levelProg` | 1 = first session, 2 = second session |
| `trial` | attempt number for that level (a level could be replayed if not completed; the completed attempt is kept) |
| `score`, `maxScore`, `prop_correct`, `isCompleted`, `outOfContextInputCount` | game summary fields, not used |
| `collisionData` | JSON dictionary, one entry per obstacle (key = position in the level, from 0) |

Each entry of `collisionData` has:

| Field | Content |
|---|---|
| `obstacleData.mode` | `NORMAL` (not used) |
| `obstacleData.type` | `JUMP` or `SLIDE` |
| `obstacleData.distanceFromPrevious` | 25 m (0 for the first obstacle) |
| `reaction` | `CORRECT`, `WRONG` (action error) or `NO_REACTION` (omission: no response within 500 ms of the cue) |
| `reactionTime` | response time in seconds relative to the auditory cue; negative = before the cue |

`code/data_prep.py` turns this file into one row per obstacle (45,416 rows) and documents the derived
variables (level, session, minute, level x minute cell, effect codes).

Level 1 = 16 m/s, 240 s, 154 obstacles; Level 2 = 17 m/s, 300 s, 204 obstacles; one obstacle every 25 m;
one fixed pseudo-random sequence per level, the same for every child.

License: CC BY-NC-ND 4.0, as the rest of the repository (see `LICENSE`).
