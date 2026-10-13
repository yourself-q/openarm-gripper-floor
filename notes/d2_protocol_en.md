# D2: measure your OpenArm 2.0 gripper's friction floor in ten minutes

**Purpose.** In public teleoperation datasets, two groups of OpenArm 2.0 datasets recorded at the same gains, most likely two grippers (the dataset names say so; unconfirmed), differ several-fold in gripper friction. Closing at 40-100 deg/s the logged torque is 0.61-0.92 Nm on unit A by dataset (0.61-0.85 by upload batch; 0.70 Nm in the reference dataset) and 0.20 Nm on unit B (one session). The still-push band (p90) is 0.545 vs 0.136 Nm, and the within-stroke scatter is 0.066 vs 0.019 Nm. A grasp that needs less motor torque than this floor does not show up in the torque log, so the floor has to be measured on each unit. The card you get is one session's value: on unit A it stepped by 17 % between upload batches of a single dataset, with the jaw angles and speeds matched. To call it the unit's value, run this once more on another day. This script measures yours with nothing in the gripper and writes one CSV.

## Requirements
- An OpenArm 2.0 **follower** arm. The gripper is the DM4310 on CAN ids 0x08/0x18. You need a Linux host with the CAN interface LeRobot uses (e.g. `can0`).
- LeRobot at commit `24f15b92b556cb861ca65e989462e23505a701bd`, installed with the `openarms` extra. That commit needs Python >= 3.12. Run `tools/openarm_floor_logger.py` from this repository in the same environment.
  - The script uses two private LeRobot names, `bus._process_response` and `bus._last_known_states`. It exits before anything moves if they are missing.
- The arm must have been calibrated once with LeRobot. Calibration sets gripper 0 deg = closed and open is negative.
  - The script connects with `calibrate=False`, so it never re-zeroes anything.
- **Settings.** The script sets these itself:
  - `use_velocity_and_torque=True`. LeRobot's default is False, and with False there is no torque in the observation.
  - Gripper **kp 16, kd 0.3**, sent with `custom_kp` / `custom_kd`. LeRobot's default gripper kp is 25.
  - If you also record datasets, set the same three values there.
- **Why kp has to be stated.** LeRobot sends the gripper a position target with zero velocity target and zero feed-forward. The motor therefore applies τ = kp (q_cmd − q) − kd v, and at rest the logged torque is just kp times the lag. kp decides three things:
  - **How far the jaw lags its command** to overcome a given friction: 0.2 Nm means 0.72 deg at kp 16 and about 0.46 deg at kp 25.
  - **Whether the jaw stick-slips** on the gear train's position ripple. It does when the ripple's stiffness exceeds kp.
  - **How hard a held object is squeezed** for a given position error.

  A and B were recorded at kp 16: the effective kp is 16.01 Nm/rad in both datasets. A card measured at another kp is not comparable. The motor receives 12-bit gains, so 16 arrives as 15.995 and 0.3 as 0.299. The CSV header records both the requested and the received gains.

## Setup
1. **Power up and note the thermal state.** *Cold* means powered up less than 10 min ago and not used. *Running* means it was used just before. You pass this as `--thermal`.
2. **Put the arm where it is safe if every motor goes limp.** Rest it on the table, or let it hang with the gripper free. When the run ends, or on any abort, the script disconnects, and LeRobot then disables torque on **all** motors.
   - During the run, the 7 arm joints are commanded to stay where they were at the start, at LeRobot's default gains (`--arm hold`).
   - `--arm ignore` talks to the gripper motor only.
3. **Clear the fingers.** Nothing may be between or near them, because the jaw travels its full range twice. No camera is needed.
4. **Rehearse first.** `python tools/openarm_floor_logger.py --dry-run` prints the timeline without connecting.

## The one command
```
python tools/openarm_floor_logger.py --port can0 --side right --id <your follower id> --thermal cold \
    --unit-note "serial or purchase date, firmware, kit or self-built, hours of use, anything replaced or greased" \
    --out d2_<unit>.csv
```
`--side` is `left` or `right`, as in your LeRobot config. The script replaces LeRobot's gripper limits with its own soft limits.

## What happens (default 10-minute budget; times for a 45-deg range)
| min | phase | what the gripper does |
|---|---|---|
| before 0:00 | probe | Prints the first reading (q, v, τ, temperatures, status), then opens 3 deg and closes back at 2 deg/s. **Answer `y` if the fingers opened.** Any other answer stops the run; send the printed numbers. |
| 0:00-1:30 | P0 range | At kp 10, moves at 1 deg/s toward closed until it stalls on the stop (2.0 Nm, 1.5 s still), then does the same toward open. The working range is 2 deg inside each stop, and it must be 30-75 deg. |
| 1:30-4:55 | P1 strokes | Unloaded triangle strokes: 1 deg/s ×2 round trips in the centre, then 5, 20 and 50 deg/s ×3 each over the whole range. |
| 4:55-6:55 | P2 creep | Four 5-deg creeps at 0.2 deg/s (close, open, close, open) at mid-range. Each starts from rest after a 3 s dwell. |
| 6:55-8:00 | P3 hold | The command stays fixed for 30 s at mid-range, then 30 s near the open end. |
| 8:00-9:57 | P5 repeat | The 1 deg/s strokes again, to show whether friction changed as the motor warmed. |
| 10:00 | end | Returns to the open end and disables torque. Prints the achieved loop rate and the time per phase. |

If your range is larger, the 1 deg/s span shrinks so the run still fits 10 min.

**Optional P4: lever calibration with a bottle** (about 2 min per finger, plus handling). Weigh a 0.5 L water bottle on a kitchen scale. After the main run:
- Run `... --bottle 0.512 --finger 1 --append d2_<unit>.csv`, then the same with `--finger 2`.
- Before each run, rest the arm with the wrist turned so that this finger closes straight **up**.
- The script makes 3 round trips of ±6 deg at 2 deg/s, then asks you to lay the bottle on that finger's pad. The bottle's full weight must be on the pad and it must touch nothing else. It makes 3 more round trips, then asks you to remove the bottle.

The difference gives the finger's effective lever and whether one or both fingers load the motor. This is the conversion from torque to pad force.

**Optional P6: dither on the free jaw** (adds about 8 min to the main run). Add `--phases P0,P1,P2,P3,P5,P6 --budget-s 1200` to the one command. The P2 creep (0.2 deg/s over 5 deg) is repeated with a sine of 0, 0.5, 1, 2, 3 and again 0 deg at 2 Hz, plus 1 deg at 5 Hz, added to the command, closing then opening. Nothing in the gripper. The card then lists, per amplitude, the cycle-mean torque offset and its scatter, and from which amplitude the encoder shows the jaw reversing every cycle. The question is at which amplitude a position-mode dither takes the friction floor and its scatter out of the torque the jaw delivers, and whether it does at all.

**Optional P7: press on a kitchen scale** (about 8 min per finger, plus a video to transcribe). After the main run:
- Run `... --scale --finger 1 --append d2_<unit>.csv`, then the same with `--finger 2`.
- Before each run, rest the arm with the wrist turned so that this finger closes straight **down**. Zero the scale and film its display for the whole run.
- When asked, put the scale under the finger: platform level, pad 1-3 cm above it. The script finds the scale at 1 deg/s (kp 10, stall at 1 Nm), backs the command off 2 deg, sends three 5-deg command steps (sync marks: the first jump of the reading in the video is the first of the CSV times the script prints), then for each amplitude presses on the scale at 0.2 deg/s until the cycle-mean logged torque reaches 1.2 Nm and backs out, and ends with three more steps.
- Transcribe the display at 2-5 readings per second into a CSV with the columns `t_s` (CSV time, aligned with the marks) and `kg`, and run `floor/d2_analyze.py d2_<unit>.csv --scale-csv readings.csv --lever-mm R` with the lever from P4 (60 if unknown: the card's `centre` value then shows the error).

The card gives, per amplitude, the friction still hidden between the commanded torque and the pad (half the width of the press/release loop) and the encoder's own dither amplitude while the pad is blocked, which grows once the dither exceeds the friction upstream of the encoder. The press torque is capped at 2 Nm, plus up to 0.84 Nm of dither ripple at 3 deg; the 3 Nm torque abort stays on. Expect the scale to read up to about 3.5 kg at a 6 cm lever. The first press may start with the scale already loaded: the friction traps the wind-up after the back-off, and the video shows by how much.

## Safety
- **Torque abort:** the run stops if |τ| > 3 Nm for 3 frames (`--tau-abort`). In P0 the lag is capped, so the stop torque stays near 2.3 Nm.
- **Speed cap:** 50 deg/s, which cannot be raised.
- **Soft limits:** 1 deg inside the measured stops. The run aborts if the jaw leaves the measured range by 3 deg.
- **Other aborts:**
  - no gripper feedback frame for 0.1 s
  - the motor's status nibble changes
  - T_MOS or T_rotor above 70 °C
  - the jaw does not follow its command for 2 s
  - P0 finds no stop within 80 deg
- **On abort:** the jaw is held where it is for 0.3 s, then torque goes off on all motors. The CSV keeps everything up to the abort.
- **Ctrl-C once:** the jaw goes back to the open end at 10 deg/s, then torque goes off. **Ctrl-C twice:** torque goes off now.
- Keep your hands away from the fingers except when P4 asks for the bottle.

## What to send back
- **The CSV.** The P4 and P7 runs are appended to it.
  - Its header holds the requested and received kp/kd, the installed LeRobot commit, your unit note and the thermal state.
  - Each row holds q, v, τ, the raw 16/12/12-bit integers, the status nibble, and T_MOS / T_rotor.
- **The unit note**, also in the header. Add anything you remember afterwards.
- **Cold or running.** Say so in your message if you are unsure what to pass.
- **The last lines the script printed** (loop rate, time per phase, any abort message).

## What you get
Your unit card from `floor/d2_analyze.py`, next to units A and B:
- the scale check and the torque law actually obeyed (kp, kd, which command the reply belongs to)
- kinetic friction per speed and per 5-deg bin in both directions
- static-minus-kinetic margin τ_s − τ_k from the creep starts
- within-stroke scatter s, defined as in the dataset cards
- the position ripple's period in encoder steps
- the noise at rest
- the start-vs-end friction and temperatures
- with P4: the lever per finger and the coupling verdict, i.e. the pad force F = τ / (r1 + r2)
- with P6: the cycle-mean offset and scatter per dither amplitude, and the amplitude from which the jaw reverses
- with P7: the friction left between commanded torque and pad per amplitude, and the encoder's dither amplitude while blocked

The A/B comparison rows use the dataset definitions, with two cautions:
- **Still-push** depends on how the jaw was driven (teleoperated holds vs the script's turn-arounds).
- **The 40-100 deg/s row** is a reading of the logged torque, whose timing bias differs between a 30 Hz teleop loop and this 200 Hz loop.

Quote the card's kinetic friction instead.

## FAQ
- **My range is not about 45 deg.** All angles follow your measured stops. A range outside 30-75 deg stops the run, which usually means something is between the fingers or the gripper is a different build. Check, then rerun. If your gripper really has another range, pass `--range-check LO HI`.
- **The torque abort tripped.** Send the CSV and the printed line (phase, angle, torque). During the fast strokes, a trip means friction near 3 Nm, about four times unit A, and that is itself a result. Do not raise `--tau-abort` above 4 Nm. To repeat only part of the run, use `--phases P1,P2,P3,P5`. The range is then taken from `--assume-range`, so put your stops there.
- **Can it run on the leader arm?** No. The leader's gripper motor is the operator's handle, and its torque is off for manual control. Its friction is what the hand feels, not the floor of the jaw that grasps. The script drives the follower class only.

## Sources
- **huggingface/lerobot @ `24f15b92b556cb861ca65e989462e23505a701bd`**, paths under `src/lerobot/`:
  - `robots/openarm_follower/config_openarm_follower.py`:
    - :31, :42: gripper limits (−65, 0) deg
    - :67: torque off on disconnect
    - :71: `use_velocity_and_torque` defaults to False
    - :102-105: default gripper kp 25.0, kd 0.3
    - :107-120: limits when `side` is unset
  - `robots/openarm_follower/openarm_follower.py`:
    - :56-58: degrees
    - :73-86: `side` selects the limits
    - :130-147: connect, calibration only when asked
    - :188-198: gripper zero = closed
    - :266-340: `send_action`, `custom_kp` / `custom_kd`, clipping; MIT command (kp, kd, q, v = 0, τ_ff = 0) at :338
    - :345-352: disconnect
  - `motors/damiao/damiao.py`:
    - :446-447: 12-bit gain encoding
    - :491-527: the MIT reply updates the state
    - :553-581: decode with temperatures, `_process_response` at :569
    - :654-701: `sync_read_all_states`
  - `motors/damiao/tables.py`:
    - :100: DM4310 limits 12.5 rad, 30 rad/s, 10 Nm
    - :196-197: kp/kd ranges 500 and 5
  - `pyproject.toml:32`: Python >= 3.12.
  - The leader: `teleoperators/openarm_leader/config_openarm_leader.py:61`, `manual_control` = torque off.
- **enactic/openarm_can**, tree `f340d4b808fb177e1f297af54eb55fd51c6c7c10`:
  - `src/openarm/damiao_motor/dm_motor_control.cpp:100-106`: feedback frame, status nibble = byte 0 >> 4, q 16 bit, v 12 bit, τ 12 bit, T_MOS byte 6, T_rotor byte 7.
  - `include/openarm/damiao_motor/dm_motor_constants.hpp:152`: DM4310 pMax 12.5, vMax 30, tMax 10.
- **Datasets** (Hugging Face, LeRobot v3, 30 fps, gripper kp 16): `gtgando/openarm-pick-place-100ep-v2` (unit A) and `gtgando/openarm-pick-place-newarm` (unit B). The cards were made with `floor/unit_floor.py`, `floor/within_stroke.py` and `floor/d2_analyze.py`. The session spread of unit A (0.61-0.92 Nm) comes from all 17 gtgando datasets, split by upload batch: `floor/f8_sessions.py`, `notes/f8_sessions_ja.md`, table in `notes/f8_table.txt`.
- **Hao, L., Pagani, R., Beschi, M., Legnani, G.** (2021). Dynamic and Friction Parameters of an Industrial Robot: Identification, Comparison and Repetitiveness Analysis. *Robotics* 10(1), 49. doi:10.3390/robotics10010049. Joint friction drifts during operation as friction heats the joint, following a double-exponential law in time. Tested on two robots of one model, on different days. This is why the run ends with the P5 repeat and asks for cold or running.
