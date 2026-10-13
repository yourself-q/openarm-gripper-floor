# The force floor of a sensorless gripper, measured from public teleop data (OpenArm 2.0)

Version of 2026-10-13 (second pass, after a second reader's audit). It replaces the draft of 2026-10-12; every number
that changed, including inside this note, is listed in §10. Everything below is measured on public LeRobot v3 datasets recorded with OpenArm 2.0 follower arms, unless marked
*computed* (derived under stated assumptions) or *from documentation*. No physical arm was used. One person, no
funding, the data of other people. Figure: `floor_fig.png` (panels a–e).

## 0. What this note claims, strongest first

1. **The torque channel in these datasets is real and correctly scaled.** A still MIT-mode joint with zero feed-forward
   obeys τ = kp·(q_cmd − q). Regressing the logged torque on the command error over still frames returns the configured
   gains to three digits (arm joints 239.5–240.0 / 23.88 / 30.78 / 24.90 against 240 / 24 / 31 / 25; gripper 16.01,
   R² 1.000, every dataset, every upload batch). (§2)
2. **Nobody in this pipeline reads force.** The teleop stack reads each motor as {q, dq, 0}, the LeRobot driver sends
   (kp, kd, q, 0, 0), and the ROS 2 hardware layer fills the gripper's velocity and torque with 0 ("Unimplemented").
   The jaws stop on the red cube at −19.6° (unit A) / −19.3° (unit B) with command 0, and the hold torque is kp × error:
   5.01–5.31 Nm across the five pick-place sets that hold the cube (three on A, two on B). (§3)
3. **The official MuJoCo model has no Coulomb floor** (`frictionloss="0"`, `damping="0.6"` on the finger joints), and the
   friction yaml shipped with the teleop stack (≈ 0.10 Nm) is within 2× of unit B and 6–9× below unit A. (§5)
4. **Two groups of datasets with the same gains and the same hold torque differ several-fold in gripper friction.** The
   dataset names say they are two arms ("newarm", "operator"); the author has not confirmed it, so "unit A" and "unit B"
   are groups by measured level. Closing freely at 40–100 °/s the median torque is 0.61–0.92 Nm per dataset on unit A
   (six datasets with ≥ 30 frames; 0.61–0.85 Nm per upload batch) and 0.203 Nm on unit B (one session); by session the
   ratio is 3.0–4.2×. On strictly still, unloaded frames the band (p50 / p90 of |τ|) is 0.29 / 0.54 Nm on A and
   0.05 / 0.14 Nm on B. The scatter inside one closing stroke is 0.066 vs 0.019 Nm (3.5×; 1.6× against A's quietest
   session). (§4.1–4.3)
5. **After the pads land on a rigid object, the motor encoder keeps moving but the pads do not.** On 32 grasps of a wooden
   cube seen by the wrist camera, the encoder advances +1.02° while torque rises +2.49 Nm, and the finger edges move in the
   image by 0.11 of what the camera's own calibration predicts (median per edge; IQR 0.04–0.52). About 90 % of the
   post-contact encoder travel is compliance between motor and pad (≈ 140 Nm/rad, *computed*). (§6.1–6.2)
6. **Unit A's level is a session value, not a constant of the unit.** It steps by 17 % between upload batches of a single
   dataset with the jaw angles and speeds matched (0.779 / 0.769 → 0.637 Nm; 0.789 → 0.652 Nm; bootstrap intervals
   disjoint). Four batches uploaded the same day span max/min 1.14; inside one batch the halves of 100ep-v2 agree to
   2–7 %, but two other sets move −16 % and +15 % inside one batch. The "drift" an earlier draft reported inside one
   dataset was this step. (§4.3)
7. **Most of the rise of unit A's opening-side resistance toward the open end is direction-independent, with a zero near
   −28°.** Decomposing steady free motion within each session, the part that does not follow the direction of motion is
   +0.07–0.13 Nm near the open end (pushing toward closed) and −0.12 to −0.15 Nm in the middle (pushing toward open;
   1–2 sessions per cell there), and it carries 60–75 % of the rise toward the open end (share post hoc; intervals
   [0.50, 0.82]). At the open end itself the direction-following part is still the larger (0.71 against 0.13 Nm).
   Whether the direction-independent part is a conservative force or a direction-asymmetric friction cannot be decided
   from torque–angle data. (§4.4)
8. **A fake strawberry of unknown material differs from the cube, but the pre-registered estimator did not decide it.**
   On the fake strawberry (another user's data; torque columns zero, command equal to state) the pre-registered
   landing-window ratio is undecided (0.39 and 0.64 in the two uploads; it moves 0.39 → 1.26 with a quarter-frame timing
   change). Post-hoc windows give 0.55–0.70, falling to 0.46–0.59 with a different calibration, against 0.05–0.10 for the
   cube under the same windows. (§6.3)

**What this note does not show.** It does not show the cause of the session step (break-in, reassembly, temperature, a
third unit and operator habit are not separated). It does not show that unit A and unit B are two physical arms (the
grouping is by measured level; the dataset names agree; the author has not confirmed). It does not decide spring versus
friction on the opening side. It does not establish that the pads push into the fake strawberry (0.55–0.70 is post hoc
and calibration-dependent), nor what the prop is made of. It does not show how repeatable a unit's level is within a day
beyond one dataset (§4.3). It gives no force in newtons without an assumed lever arm (§7). Nothing here was run on an
arm: the ten-minute per-unit protocol (§8) exists as a script verified only in dry runs.

## 1. Hardware, data, decoding

- OpenArm 2.0 follower (Enactic). Gripper motor: Damiao DM-J4310-2EC (rated 3 Nm, peak 7 Nm, *from documentation*),
  MIT mode with position gain kp, damping kd, zero feed-forward torque. Gripper command 0 = closed, negative = open
  (degrees in the LeRobot `openarm_follower` convention).
- Datasets (LeRobot v3, `robot_type: openarm_follower`, `observation.state` = position / velocity / torque for 8 joints
  = 24 columns; gripper = columns 21–23; `action[7]` = gripper command), all by one author (gtgando), all with gripper
  kp 16: `openarm-pick-place-50ep`, `-practice`, `-100ep-v2`, `-blue-20ep`, `-orange-20ep`, `-yellow-20ep`,
  `openarm-cube-organization-v2`, `-cube-organization-full-clear-v2` (uploaded 2026-03-04..14, "unit A" below),
  `openarm-pick-place-newarm` (2026-03-18) and `-operator-50ep` (2026-03-23) ("unit B"). Of the author's 19 datasets
  (as listed on 2026-10-12), these ten are readable kp-16 sets; four have a truncated parquet footer and cannot be read
  (`-clean`, whose stats match a subset of 50ep, and three February sets); three February sets were recorded with other
  gains (kp_eff 10.73 / 3.58 / 1.44) and are not compared; two were not downloaded (`openarm_table_pickplace_v3`, 0
  episodes, and `cube_pickplace_v3`). §2–3 and §4.1 use the first draft's sets (100ep-v2, 50ep, blue-20ep for unit A;
  newarm, operator-50ep for unit B); §4.3–4.4 use all ten readable kp-16 sets.
- Unit A / unit B is my grouping by the measured friction level (and originally by joint range; §4.3 shows the open end
  is set by the operator, −40.9 to −55.7° across unit A sessions, so range does not identify a unit). The names
  "newarm" and "operator" suggest a second arm; the author has not confirmed it. There is no serial number in the data.
- Second source for §6.3 only: `xhaka3456/openarm_pick_strawberry_and_place_on_white_plate` (v1, 20 episodes) and
  `…plate2` (v2, 40 episodes), left wrist camera, gripper state only (the command column equals the state in every frame;
  velocity and torque columns are all zero).
- Decoding scale (*from the Damiao MIT protocol*): torque LSB 20/4095 = 0.00488 Nm, position LSB 0.0219°, velocity
  LSB 0.84 °/s. The kp regression of §2 is the check that these scales are applied as stated.

## 2. Effective gains (measured)

Still frames (|v| < 1 °/s, adjacent-frame position difference < 0.03°; §4.1 uses a stricter 7-frame window), τ against
(q_cmd − q):

| joint | kp_eff (Nm/rad) | configured | R² |
|---|---|---|---|
| 1–4 | 239.5–240.0 | 240 | > 0.999 |
| 5 / 6 / 7 | 23.88 / 30.78 / 24.90 | 24 / 31 / 25 | > 0.999 |
| gripper | 16.01 | user value, all gtgando sets | 1.000 |

The gripper value is 16.01 in every readable kp-16 dataset and in every upload batch separately, so the torque scale did
not change between the sessions compared in §4.3.

## 3. The hold (measured)

With command 0 (closed) the jaws stop on the red cube at −19.6° (unit A) / −19.3° (unit B). The hold torque is then just
kp × error: 5.31 / 5.28 / 5.30 Nm on unit A (100ep-v2 / 50ep / blue-20ep; command error 19.1°, 16.01 × 0.333 rad =
5.33 Nm) and 5.17 / 5.01 Nm on unit B (newarm / operator-50ep). The hold force is whatever the position error times kp
produces, and nothing in the stack reads it back: `openarm_teleop` reads each motor as {q, dq, 0}, LeRobot's
`openarm_follower` sends (kp, kd, q, 0, 0), and `openarm_ros2`'s hardware layer fills the gripper's velocity and torque
with 0 ("Unimplemented: Velocity and torque mapping").

## 4. The friction floor (measured)

### 4.1 Still frames: the band

A still motor under nonzero torque is held by static friction (or by a stop or an object; those frames are excluded by
requiring the jaw to be ≥ 6° more open than that episode's hold angle). Strictly still: position range < 0.03° over a
7-frame window and |v| < 1 °/s.

| | unit A (100ep-v2) | unit B |
|---|---|---|
| pushing closed, still, free: p50 / p90 of \|τ\| | 0.29 / 0.54 Nm (n 155) | 0.05 / 0.14 Nm (newarm, n 238); 0.06 / 0.11 (operator-50ep, n 1,684) |
| pulling open, still: p50 / p90 | 0.55 → 1.08 / 0.57 → 1.64 Nm, growing from the −27..−22° bin to the −52..−47° bin (n 12,004) | 0.04–0.13 / 0.10–0.24 Nm, flat over angle |
| dwell of a still-under-torque state | up to 1.13 s (opening side; 1.30 s in blue-20ep); 0.30 s (closing side) | — |
| joint range in the data | −55.7 .. 0.3° (100ep-v2) | −45.1 .. 0.5° (newarm) |

### 4.2 Steady motion: the kinetic level

Free motion (no object, |a| < 150 °/s², |τ| ≤ 1.5 Nm). Over 21–290 °/s the torque is flat, so Coulomb friction
dominates and no viscous term is visible (unit A, 100ep-v2):

| motor angle | closing | opening |
|---|---|---|
| −22..−12° | (few frames) | 0.44 Nm |
| −32..−22° | 0.68 | 0.63 |
| −42..−32° | 0.64 | 0.88 |
| −52..−42° | (few frames) | 0.97 |

Closing is flat at 0.64–0.68 Nm (static ≈ kinetic on this side). Opening grows toward the open end; §4.4 decomposes it.
Unit B: ≈ 0.15–0.19 Nm both ways on the first draft's coarse 15 °/s band (20–30 frames); 0.203 Nm in the 40–100 °/s
band used below. The 15–40 °/s band is not usable as a friction level: in every dataset 15–72 % of its frames are
moving with an object in the jaws (|τ| > 1.5 Nm); in the 40–100 °/s band it is 0–3 %.

### 4.3 Across sessions: unit or session? (`f8_sessions.py`)

Definition: the kinetic closing level is the median |τ| while closing freely at 40–100 °/s. Sessions are approximated by
Hugging Face upload batches (one "Upload folder" commit each; the commit time is an upper bound on the recording time).
Intervals are 95 % episode bootstrap (2000 resamples). Full table: `notes/f8_table.txt`.

- **Unit A, by dataset** (six with ≥ 30 frames): 0.608–0.916 Nm, max/min 1.51. **By upload batch** (nine with ≥ 30
  frames): 0.608–0.852 Nm, max/min 1.40. Matching the jaw-angle mix changes neither (1.51 / 1.44); the slope of |τ|
  against speed inside the band is −0.13..+0.14 Nm per 100 °/s (one batch, 50ep batch 2, gives −1.75 on 35 frames), so
  the spread is not a speed effect.
- **The spread is a step at batch boundaries, not a drift.** 100ep-v2: the two batches uploaded 03-06 give
  0.779 [0.745, 0.803] (n 69 frames / 31 episodes) and 0.769 [0.735, 0.808] (83 / 40); the batch uploaded 03-10 gives
  0.637 [0.628, 0.647] (112 / 43), −17 to −18 %. blue-20ep at the same boundary: 0.789 [0.728, 0.840] (32 / 14) →
  0.652 [0.630, 0.698] (42 / 13), −17 %. Inside a batch of 100ep-v2 the first and second halves agree to +2.0 / +6.5 /
  −0.5 %, and odd and even episodes to 0.5–5.7 %; but 50ep batch 2 moves 0.930! → 0.779! between its halves (−16 %, fewer
  than 30 frames each) and cube-organization-v2 0.558 → 0.641 (+15 % angle-matched, +4 % without), so agreement inside
  a batch is not general. Four sessions uploaded on 03-06 span max/min 1.14 (±6.5 %).
- **Other card entries step at the same boundary**: the still pulling-open torque, angle-matched, 0.873 / 0.856 →
  0.545 Nm in 100ep-v2 (−36 to −38 %) and 1.017 → 0.759 in blue (−25 %). Without angle matching the step is hidden,
  because the open end differs between batches (−45.7 / −55.7 / −51.7°) and the opening side depends on angle (§4.1).
- **Unit B**: one session with ≥ 30 frames, newarm 0.203 [0.188, 0.208] (69 / 27). operator-50ep gives 0.164!
  [0.144, 0.173] on 23 frames, below the threshold (0.81× newarm, intervals disjoint); its still-frame entries are
  0.8–1.3× newarm's (pushing closed p50 / p90 1.22 / 0.81, pulling open 0.80 / 1.27). Whether B's level also steps
  between sessions cannot be read from this data.
- **The unit difference survives the session spread**: A's lowest session 0.608 against B's 0.203 is 3.0×; A's highest
  0.852 is 4.2×. Read "A is 3–4× B" as the ratio of a unit whose level moves ±20 % between upload batches to a unit with
  one readable session (whose second session, 23 frames, sits 19 % lower).
- **Within-stroke scatter** (DOF-corrected pooled std of the closing-stroke residuals, `within_stroke.py`): 0.066 Nm on A
  (100ep-v2) vs 0.0187 Nm on B (newarm), 3.5×. Across A's sessions it is 0.030–0.082 Nm, so against B's one value the
  ratio is 1.6–4.4× (*computed* from the two ranges).

What this changes: a per-unit friction card (§8) is one session's value; to call it the unit's value it has to be measured
again on another day. The cause of the step (break-in, reassembly, temperature, a different unit behind the same dataset
name, operator habit) is not separated by this data.

### 4.4 The opening side: a direction-independent part with a zero near −28° (`f9_direction.py`)

Pre-registered question: is unit A's heavier opening side (0.44 → 0.97 Nm steady, 0.55 → 1.08 still) a conservative force
(spring, soft end stop, cable), a position-dependent friction, or both? Decomposition inside each session, on steady free
motion (|v| 40–100 °/s, |a| < 150 °/s², |τ| ≤ 1.5 Nm, more than 3° from the session's open end and from 0): per angle
bin, τ_open and τ_close are the median |τ| per direction; f = (τ_open + τ_close)/2 follows the direction of motion;
s = (τ_open − τ_close)/2 does not (positive = pushing toward closed). Cells need ≥ 5 frames per direction; intervals are
95 % episode bootstrap; "!" marks cells under 30 frames; k = number of sessions with the cell. Full table:
`notes/f9_table.txt`.

- **The pre-registered 5° grid is thin.** Unit A's twelve sessions hold only 431 opening and 820 closing frames in the
  band; 15 cells in 5 sessions; the key bin −47..−42° is empty. The pre-registered open-end prediction cannot be tested
  in its original form.
- **Direction-symmetric friction alone is rejected: s ≠ 0.** Pooled over sessions: +0.106! [0.081, 0.138] at −42..−37° (k 2),
  +0.068 [0.042, 0.103] at −37..−32° (k 3), −0.147! [−0.164, −0.104] at −22..−17° (k 2).
- **s changes sign near −28°, inside single sessions** (not predicted): cube-organization-v2 +0.125! (−42..−37°) →
  +0.063 (−32..−27°, 41 / 48 frames) → −0.062! (−27..−22°) → −0.143! (−17..−12°); 100ep-v2 batch 1 +0.027! → +0.011! →
  −0.074! → −0.173! (−22..−17°). Linear zero crossings −27.0° and −28.9°; pooled −27.6° (*computed*). Read as a
  conservative force it would push toward −28° from both sides, not the shape of a soft end stop.
- **Share of the rise** (post hoc, same thresholds; within-session least squares of cell values, slope per 10° toward
  the open end, median over 8 sessions on a 10° grid): s +0.144 [0.086, 0.162] (8 of 8 positive), f +0.047 [0.032, 0.092],
  τ_open +0.175 [0.128, 0.225], τ_close −0.073 [−0.087, −0.030]. s / (s + f) = 0.75 [0.56, 0.79]; on the 5° grid
  (3 sessions) 0.62 [0.50, 0.82]. So 60–75 % of the opening-side rise is direction-independent and the closing side is
  lighter near the open end by 0.07 Nm per 10°.
- **The pre-registered magnitudes miss.** In the 10° bin [−47, −37) (k 4): s 0.126 [0.098, 0.161], f 0.711 [0.675,
  0.745], τ_close 0.585 [0.556, 0.619]. My prediction (s 0.15–0.25, f 0.75–0.85, |s| ≤ 0.10 in the middle) fails on all
  three numbers; the shape (s and f both rise toward the open end, τ_close falls a little) is as predicted.
- **Not gravity**: regressing on the gravity torque at the finger axes from the official MJCF forward kinematics gives
  slope 1.30 [0.11, 2.40] and an effect of 0.020 [0.002, 0.037] Nm; gravity alone would change s by < 0.012 Nm across
  the bins, against a measured change of 0.25 Nm; the bound for any mounting orientation is 0.045 Nm. **Not speed or
  acceleration** (|s_dyn| ≤ 0.002 Nm; post hoc), **not the object** (post hoc): dropping opening frames within 0.5 s
  after a release (0–14 %) and closing frames within 0.5 s before a grasp (0–17 %) moves s by ≤ 0.02 Nm, except one
  session's −42..−37° cell (+0.125).
- **Still frames give a larger offset than steady motion**: in the same sessions and bins the still one-sided offset
  exceeds s by +0.068! [0.023, 0.153] (−42..−37°) and +0.115 [0.054, 0.203] (−37..−32°). By the pre-registered rule this
  points to the still value also reflecting how the command approached the position; that reading is a suspicion, not a
  measurement of the effect.
- **Unit B is flat**, as predicted: still offset −0.002..+0.037 Nm (4 bins, 2 sessions), slope −0.014 [−0.049, 0.053];
  only one steady cell (newarm −32..−27°, s +0.018!, 6 / 7 frames).
- **Absolute angle or distance from the open end**: not decided (residual difference +0.029 [−0.008, 0.044]).
- **Anchors reproduce**: the still pulling-open p50 at −42..−37° is 0.886 (n 4352); the steady values of §4.2 come back
  as 0.637 / 0.877 under `band.py`'s masks and 0.676 / 0.874 under this script's own (the difference is the mask).

**Why "spring versus friction" is undecided, and this is a flaw of my pre-registration.** s is the direction-independent
part. A conservative force and a position-dependent friction that is asymmetric in direction (one that bites near the
open end only when opening) give the same s in a torque–angle sweep, and the still-frame test I had planned as the
tie-breaker has the same symmetry. What the data supports is exactly: 60–75 % of the opening-side rise is a
direction-independent function of angle with a zero near −28°. What separates the two needs the arm: release the jaw
with zero gains (kp = kd = 0, τ = 0) at −40° and at −15° and watch whether it drifts toward −28° (a conservative force)
or stays (friction). The data-only alternative, whether s scales with f across the 17 % session step of §4.3, has one
shared cell (−37..−32°: s +0.027! / +0.087!) and is too thin; a test that stays ≥ 6° from the hold angle cannot reach
the middle bins at all.

## 5. The official simulator (from documentation, with computed consequences)

`enactic/openarm_mujoco`, `openarm_bimanual.xml`: finger hinge range 0–0.7854 rad, class `motor_finger`
`frictionloss="0" damping="0.6"`, position actuator kp 10 kv 0.2, forcerange ±7 (set by
openarm_mujoco PRs #75/#76, merged 2026-09-24/28: finger damping 2 → 0.6, kp 10 kv 0.2; `frictionloss` stayed 0 and
master has no `noslip_iterations`; raw file checked 2026-10-13). `openarm_teleop`'s friction yaml
(Fc 0.051 + |Fo| 0.050 ≈ 0.10 Nm, unchanged since the v1.0 import) is within 2× of unit B's 0.20 Nm and 6–9× below unit
A's 0.61–0.92 Nm. Issue enactic/openarm#336 (open since 2025-11-19) asks for exactly these parameters.

Loading the measured floor into the official MJCF (`mjcf_floor.py`, *computed*): `frictionloss` 0.6 (A) / 0.15 (B) with
`noslip_iterations` raised from the official 0 to 10, because with 0 the soft friction constraint lets the finger creep
at 2 °/s under 0.11 Nm; `damping` 0, because the official 0.6 adds a viscous 1.2 × v that the data does not show
(§4.2). A session-faithful model of unit A would also carry the angle-dependent opening side of §4.4 and a per-session
level in 0.61–0.85 Nm (0.61–0.92 per dataset). After contact the MJCF's τ–q slope is 1050–1150 Nm/rad, 7–8× stiffer than
the ≈ 140 Nm/rad computed in §6.2.

## 6. The wrist camera versus the encoder (measured)

### 6.1 Idea and method (`padcal.py`)

The wrist camera is fixed to the gripper body, so the image position of each finger's inner edge is a function of finger
angle alone, independent of the arm's motion. That gives a self-calibration with no object and no ruler, and then a
test: after the pads land on a rigid object, does the encoder's further travel show up at the pads?

`gtgando/openarm-pick-place-100ep-v2`, episodes 0–47 (14,987 frames, 640×480, H.264, 30 fps), every frame decoded. On
rows 280–320 (where the pads meet the cube) scan outward from the image centre until max(R,G,B) < 70 for 3 consecutive
pixels; sub-pixel crossing by linear interpolation; median over rows. (a) Calibration from still, unloaded frames
(|v| < 2 °/s, |τ| < 1.3 Nm, no red between the pads), 2° bins. (b) Grasps: landing A = first frame with |v| < 40 °/s
and τ > 1.5 Nm after a quiet onset, hold B = first frame at 95 % of the hold plateau (A..B ≤ 10 frames). (c) Edge
scatter during holds.

### 6.2 Results on the wooden cube (unit A)

- Calibration is smooth and monotone: left edge +2.63 px/deg, right −2.51 px/deg, gap −5.14 px/deg over −45..−21°
  (4–5 px/deg near the hold angle). Extrapolated to the hold angle (−19.7°) the free-motion line gives a gap ≈ 97 px
  (−4.18 px/deg fitted over −29..−21°; ≈ 100 px with the wider-range slope in the notes). The measured gap at hold is
  117 px (median over 48 grasps, one value per grasp; IQR 111–122, range 95–136; the two example episodes in the notes
  sit at 109 px; the cube's red width is 95–110 px) (panel b). The ≈ 20 px excess has the sign a loaded, wound-up
  finger would give, but it is a 3° extrapolation into a changed scene, and the 40 px range across grasps says the hold
  gap depends on more than the finger angle (how the cube sits between the pads), so it is listed as consistent, not as
  a measurement.
- 32 grasps. Encoder A→B: +1.02° (median; 0.57–1.64). Torque: +2.49 Nm. Predicted gap change: −5.1 px (−5.14 px/deg
  × 1.02°). Measured: +0.09 / −0.36 px (left / right edge). Visible fraction: median 0.11 per edge (IQR 0.04–0.52).
  Arm-still subset (11 grasps, arm joint speed < 5 °/s): 0.14. Panel d of the figure shows a gap-based variant computed
  in `floor_fig.py` with the local slope near the hold angle (−4.2 px/deg): predicted −4.3 px, measured −0.58 px,
  fraction 0.13 (IQR 0.04–0.63); it is not in the notes.
- Edge scatter during holds: 0.70 px left, 0.49 px right (median of per-episode std). The predicted 2–4 px per pad is
  ≥ 4σ above it.
- Timing. The image lags the state by about one frame in the landing sequences. Recomputing (b) with assumed lags
  −2..+2 frames gives 2.40 / 1.40 / **0.11** / 0.05 / 0.02; only an image that *leads* the state would undo the result,
  and the landing sequences exclude that direction. In the hold the gap is flat for ten frames, so no alignment hides a
  5 px step (panel c).
- Unit B (newarm, 24 grasps): per-edge slope 2.5–2.6 px/deg within 2 % of A, but B's hold angle lies outside its
  calibration range (−45..−31°), so the fraction is an extrapolation (0.14, IQR −0.03..0.44) and is not counted. Its
  hold-time edge scatter is 1.0–2.1 px, 2–3× A's, dominated by slow drift rather than frame noise.

**Reading** (confidence medium–high; one object, one unit, compressed video). About 90 % (median; 48–96 % over the IQR)
of the ≈ 0.4 deg/Nm the encoder
shows after contact is wind-up between the motor's encoder and the pad. A wooden cube does not shrink 1 mm per side, and
the camera confirms the pads do not move. The gripper's own stiffness seen from the motor is ≈ 140 Nm/rad (*computed*
from 1.02° / 2.49 Nm); objects stiffer than that are invisible to the encoder. The camera resolves the pad–object
boundary to 0.5–0.7 px per edge, ≈ 0.2–0.27° of finger angle (per-edge slope 2.5–2.6 px/deg; unit B 0.4–0.8°), from
640×480 H.264 at 30 fps; the 2.0 hardware's 1920×1200 global-shutter camera (official docs; frame rate not verified)
could lower that floor (not measured). In the same 70–330 ms the motor saw +2.5 Nm, the encoder saw +1°, and the camera saw
"nothing moved". The disagreement itself is the contact signal.

What the camera could see below the motor floor (*computed*, `pixel_sens.py`, `notes/d3_pixel_ja.md`): with rigid
fingers the visible deformation is the object's, so the camera force floor is jitter × 2k ÷ (px/mm). With the contact
scale calibrated from the real edge slopes (1.55–2.06 px/mm at 640 width; the official MJCF camera does not reproduce
the real video and is not used) and unit A's jitter, assumed stiffnesses give 0.25–0.90 N for a strawberry, 0.06–0.38 N
for a grape at the P1-3 stiffness, 0.65–1.19 N for a blueberry and 0.98–1.80 N for a grape at 2 N/mm (all ≤ 0.6× unit
A's floor, taken there as 3–5 N for the pooled 0.60–0.65 Nm level; 3–7 N over the per-session range), 2.45–4.50 N for a
cherry (at the floor), 6.9–17.9 N for a grape at 14 N/mm or an apple (above the floor), and nothing for a rigid body
(≥ 49 N, above the 27–44 N hold). Below unit B's floor only the P1-3 grape and the softest strawberry remain. The
stiffnesses are assumptions; pad compression is not visible below the floor (k_pad ≥ 71–151 N/mm from §6.2).

### 6.3 A second object: the fake strawberry (`strawberry_ratio.py`, pre-registered, verdict undecided)

xhaka3456's two uploads (v2: 40 episodes, 21 using the left hand; v1: 20, 10), left wrist camera 640×480 at 30 fps; the
torque columns are all zero and the gripper's command equals its state in all 51,825 / 24,980 frames, so there is no
torque signal. The object is a fake
strawberry (judged from the video before the analysis). The pads were tracked by their through-holes (left L, right
upper Ru, right lower Rl), gap G = Ru − L; calibration at the hold angle L +2.90 / Ru −2.94 / Rl −2.73 px/deg; per-edge
jitter 0.20–0.24 px. r = visible pad motion ÷ encoder-predicted motion, as in §6.2.

- **Pre-registered landing window A→B5: undecided.** v2 0.391 (IQR −0.030..0.590, n 71 of 79 grasps, 8 excluded by the
  pre-registered rule); v1 0.638 (0.143..2.690, n 32 of 36, 4 excluded). v1 crosses 0.5 within the pre-registered
  ±2-frame timing shifts (0.172 at one shift), so the pre-registered rule returns "undecided"; the two uploads disagree.
- **Why: the estimator is ill-conditioned, and that is a flaw of my pre-registration.** The closing step moves ≈ 9° per
  frame (≈ 50 px of gap) while the post-landing advance is 0.37° (≈ 2 px). The camera leads the encoder by 1.89 frames
  (sd 0.44 per grasp; v1 1.71, sd 1.01), so integer alignment leaves a fractional error through which the tail of the
  fast step leaks into the window at the 10 px level. Shifting the alignment from −2.00 to −2.25 frames moves v2's r
  from 0.39 to 1.26. I fixed the window without counting the signal against the leakage.
- **Post-hoc windows that skip the landing transient** (chosen after seeing the data): A+2→B_end v2 0.667 (IQR
  0.36..0.81, n 65), v1 0.682 (0.56..0.91, n 32); A+5→B_end v2 0.581 (0.386..0.772, n 62 / 77), v1 0.702 (0.529..0.884,
  n 30 / 34), ranging 0.54–0.65 / 0.43–0.70 over shifts −2..+2; within-hold regression v2 0.552 (n 74), v1 0.607 (n 36).
  With the open-end calibration instead of the hold-angle one (slopes 19–24 % larger: +3.52 / −3.58 / −3.38 px/deg)
  these fall to 0.46–0.59. So "above 0.5" is not established.
- **The robust sentence**: the wooden cube of §6.2 under the same windows gives 0.05 (A→B_end, n 32), 0.07 (A+2, n 16),
  0.10 (A+5, n 9), while the encoder's advance in the windows compared is similar (cube A+2→B_end 0.39° at +0.30 Nm;
  strawberry A→B5 0.37°; the cube's full A→B_end advance is 1.25°). The pads follow about half of the encoder's advance
  on the strawberry (post hoc 0.55–0.70) and almost none of it on the cube. This is a different unit, camera and
  detector, not a same-system control.
- Under the series-spring model r = k_g / (k_g + 4 k_obj r_c²) with unit A's k_g borrowed (≈ 140 Nm/rad) and r_c 0.06–0.097
  m, the post-hoc r gives k_obj 1.6–7.9 N/mm (1.6–11.6 with the open-end calibration): at the lower end of the
  pre-registered band for a fake (2–50 N/mm; the prop's material is not known) and 2–10× stiffer than a real strawberry.
  Compression itself is not visible (≈ 0.4 px per pad expected; the montages show none). Camera motion during the hold
  can also be object rotation, slip or creep (grasps with r > 1.2, 10 of 64 in v2, include visibly rotating objects, and
  the same tail is what leakage of the fast closing step would produce), so the mechanism is not separated.
- Data facts worth recording: in this recording system the camera leads the encoder by ≈ 1.9 frames (in §6.2's it lagged
  by 1); the encoder value repeats in 17.4 % of moving frames (state updates possibly slower than frames; unverified);
  the gripper column's quantum matches the Damiao 16-bit position × the 0.044 m / 1.0472 rad conversion of
  `openarm_hardware`.

## 7. Converting torque to pad force (computed, assumptions stated)

Nothing public gives the 2.0 gripper's lever arm or transmission ratio. Assumptions: motor torque splits over two fingers,
pad lever 0.06–0.097 m, 1:1 motor-to-finger (the official MJCF couples finger2 to finger1 with a unit polycoef, and
`openarm_ros2` maps the pinch gripper 1:1 "approximately"). Then F_pad = τ / (2r):

| quantity | motor torque | per-finger pad force |
|---|---|---|
| torque LSB | 0.0049 Nm | 0.025–0.04 N |
| unit B, still band p50 / p90 (§4.1) | 0.05 / 0.14 Nm | 0.3–0.4 / 0.7–1.2 N |
| unit B, kinetic closing, 40–100 °/s (§4.2) | 0.20 Nm | 1.0–1.7 N |
| unit A, kinetic closing, 100ep-v2 pooled (§4.2) | 0.60–0.65 Nm | 3–5 N |
| unit A, kinetic closing, per upload batch (§4.3) | 0.61–0.85 Nm | 3–7 N |
| hold on the cube, unit A (§3) | 5.3 Nm | 27–44 N |

If the whole torque goes to one finger (τ / r) multiply by 2. The relative statements (3–4× between units; band ≫
resolution; hold ≈ 6–9× unit A's band) do not depend on these assumptions. The jaw-range mismatch of §4.1 (unit A opens
to −55.7°, the official MJCF hinge is 45°) suggests a ratio ≥ 1.24 between motor and finger on unit A, which would scale
the newton figures; it is not separated from a zero offset.

## 8. What this means for datasets, simulators and controllers

- **Datasets.** No current OpenArm dataset records which unit it came from or its friction level, and a policy trained on
  one unit inherits a 0.2 Nm or a 0.6–0.9 Nm floor silently. A ten-minute protocol measures it: `notes/d2_protocol_en.md`
  (one page for an arm owner), `tools/openarm_floor_logger.py` (LeRobot `openarm_follower`, gripper only, kp 16 / kd 0.3,
  200 Hz CSV with raw integers and motor temperatures, safety stops) and `floor/d2_analyze.py` (the same card from the
  CSV or from a LeRobot v3 dataset). It has run only in dry-run and mock mode (the card recovers injected values; on the
  four real datasets it reproduces the values of `unit_floor.py` / `within_stroke.py`: A 0.286 / 0.545, 0.70, 0.066; B
  0.046 / 0.136, 0.20, 0.019). Its loop rate with the CAN round trip is unknown (199.8 Hz in the mock). Because of §4.3 a
  card is one session's value: run it on two days.
- **Simulators.** Add the floor (§5). Do not read the finger joint's position after contact as pad position (§6.2).
- **Controllers.** The smallest force a motor-only controller can regulate on this hardware is set by the unit's friction
  band, not by torque resolution: of the order of 3–7 N per finger on unit A (per-session kinetic level) and 0.3–1.7 N
  on unit B (still band to kinetic level) (§7), against a 27–44 N hold. The approach side (closing) is lighter near the
  open end by 0.07 Nm per 10° (§4.4). After contact the encoder stops being a displacement sensor; the wrist camera does
  not, and for objects softer than about 2 N/mm it would see the object below unit A's floor (below unit B's floor, only
  the softest of them) (§6.2, computed).

## 9. Confidence per number

| number | basis | confidence |
|---|---|---|
| kp_eff, torque scale | regression over thousands of still frames, all sets and batches | high |
| hold angle −19.6 / −19.3°, hold torque 5.01–5.31 Nm | five sets (three A, two B) | high |
| unit A still band 0.29 / 0.54 Nm | 155 still frames, one dataset; across five sessions the p90 is 0.48–0.65 Nm | high for 100ep-v2; session-dependent |
| unit A kinetic closing 0.64–0.68 Nm (100ep-v2 pooled), 0.61–0.92 across datasets (0.61–0.85 across upload batches) | hundreds of frames; six datasets / nine batches with ≥ 30 frames, episode bootstrap | medium–high that it varies (the step shows in two datasets); medium on the range (sessions = upload batches, identity of the unit assumed) |
| the 17 % step at batch boundaries | 100ep-v2 and blue-20ep, disjoint bootstrap intervals, angle and speed matched | medium–high; cause unknown |
| unit B still band 0.05 / 0.14 Nm, kinetic 0.203 Nm | 238 still frames; 69 frames / 27 episodes; a second session (operator-50ep) has 23 frames at 0.164 | high (still), medium (kinetic: one session), low on B's behaviour between sessions |
| A / B ratio 3.0–4.2× | the two ranges above | high that it is several-fold; the figure depends on which A session |
| within-stroke scatter 0.066 / 0.0187 Nm | DOF-corrected pooled std | medium (A's sessions span 0.030–0.082) |
| that A and B are different physical units | measured level only | medium; dataset names agree, author unconfirmed |
| opening side grows toward the open end | thousands of frames | high |
| direction-independent part 60–75 % of the rise, zero near −28° | within-session decomposition, episode bootstrap, most cells < 30 frames, share is post hoc | medium |
| spring vs friction | not separable by this design | undecided |
| gravity 0.020 [0.002, 0.037] Nm (bound 0.045 for any mounting), dynamics ≤ 0.002 Nm | MJCF kinematics regression; speed and acceleration terms, post hoc | medium–high |
| encoder +1.02°, τ +2.49 Nm after landing | 32 grasps | high |
| pads move 0.11 of prediction (IQR 0.04–0.52) | 32 grasps, predicted motion ≥ 4σ above jitter, lag checked | medium–high |
| 140 Nm/rad gripper stiffness | ratio of the two medians, 2.49 Nm / 1.02° | computed |
| camera resolution ≈ 0.2–0.27° per edge (A), 0.4–0.8° (B) | per-edge jitter ÷ per-edge slope | medium |
| fake strawberry r, pre-registered 0.39 / 0.64 | 71 / 32 grasps, ill-conditioned window | undecided |
| fake strawberry r, post hoc 0.55–0.70 (0.46–0.59 other calibration) | windows chosen after the data, two uploads agree, calibration-dependent | medium, post hoc |
| cube under the same windows 0.05–0.10 | 9–32 grasps, different system | medium |
| k_obj 1.6–11.6 N/mm | series-spring model, k_g borrowed from unit A, r post hoc | computed, low–medium |
| pad forces in newtons | §7 assumptions | computed, order of magnitude |
| camera force floor below the motor floor (fruit) | assumed stiffnesses, calibrated px/mm | computed, low–medium |
| the ten-minute protocol | dry run and mock only | not run on hardware |

## 10. Corrections made during this work

All in the Japanese notes with time stamps; collected here so the reader sees what moved and why.

- Unit B "5–10× more regular within a stroke" → 3× → **3.5–3.7×** (0.066 / 0.018–0.019 Nm). The first figure counted
  unit A's between-batch level change as within-stroke noise; the second subtracted per-episode medians from 1–3 frames
  per episode and shrank the residuals; the final figure is a DOF-corrected pooled std.
- Camera angle floor "0.1–0.14°" → **0.2–0.27° per edge**: the per-edge jitter had been divided by the two-edge slope.
- "Unit A's level drifts 0.78 → 0.77 → 0.69 → 0.64 inside 100ep-v2" → **a step at the upload-batch boundary** (§4.3);
  the quartile cut happened to straddle the boundary.
- "Unit A floor 0.60 Nm / 0.70 Nm" → **0.61–0.92 Nm depending on the session**; "4–6×" (still band, reference datasets)
  is kept with its definition, and the session-resolved kinetic ratio is **3.0–4.2×**.
- Joint range as a unit identifier → dropped; the open end is set by the operator (§1).
- "Spring versus friction unresolved, to be tested by the still-frame offset" → **not identifiable from torque–angle
  data** (§4.4); the pre-registered 5° grid left the key bin empty.
- F7 pre-registered landing window → **ill-conditioned** (§6.3); the post-hoc numbers are labelled post hoc.
- Unit B's §6.2 fraction (0.14) → extrapolation, not counted.
- Hold torque "5.01–5.45 Nm across seven sets" (10-12 draft and the first version of this note) → **5.01–5.31 Nm across
  five sets**; neither 5.45 nor a seventh set is in the notes.
- Camera fraction "0.11–0.13" → **0.11** (per edge, as in the notes); 0.13 is a gap-based variant computed only in the
  figure script. "Predicted −4.3 to −5.1 px" → **−5.1 px** (−4.3 px is the figure's local-slope value, panel d);
  "IQR 0.04–0.6" → **0.04–0.52**; "4σ–8σ" → **≥ 4σ**. "Hold gap 109–120 px" was first narrowed to "≈ 109 px" (the two
  example episodes in the notes); when the figure was regenerated for release (2026-10-13) the median over 48 grasps was
  measured as **117 px** (IQR 111–122) against an extrapolated 97 px, and §6.2 now says so.
- Stiffness "≈ 143 Nm/rad" → **≈ 140 Nm/rad** (2026-10-13): 143 was 1 / (0.4 deg/Nm) with the ratio rounded first;
  2.49 Nm / 1.02° gives 140. Nothing downstream changes (§6.3 already used 140).
- Figure regenerated 2026-10-13 so that its titles say no more than the body: panel (e) "two units of the same gripper"
  → two dataset groups, most likely two arms (unconfirmed); panel (d) labels the 0.13 as gap-based and names the 0.11
  per edge; panel (b) shows both the measured hold gap (117 px) and the free-motion extrapolation (97 px).
- Newtons: the Japanese notes convert unit A's pooled level (0.60–0.65 Nm → 3–5 N) and unit B's still band (0.3–0.7 N);
  the per-session 3–7 N and unit B's kinetic 1.0–1.7 N in §7 are conversions made for this note. The first version of
  this note wrote 3–8 N and 1–2 N.
- `-clean`, counted as a unit A set in the 10-12 draft → unreadable (truncated footer); its stats match a subset of 50ep.
- Three headings of §0 in the first version of this note said more than the body: "two grippers of the same design"
  (two physical units are not confirmed), "batches uploaded the same day agree to 2–7 %" (that is one dataset's halves;
  same-day batches span 14 % and two sets move 15–16 % inside a batch), and "opening-side resistance mostly
  direction-independent" (it is most of the *rise* toward the open end; at the open end the direction-following part is
  larger). Found by a second reader on 2026-10-13 and corrected here.

## 11. Open, and what would settle each

- Transmission ratio and lever arm (needed for newtons): the official CAD, or the protocol's optional load step (a known
  mass on one finger).
- Cube size (needed for mm per pixel; at 45 mm the pads would be ≈ 150 mm from the camera and the camera floor
  ≈ 0.2–0.3 mm).
- Spring versus friction on the opening side: zero-gain release at −40° and −15° on the arm, one minute. The mechanism
  with a zero near −28° (cable neutral, link dead point) can be read from the CAD.
- The cause of the session step: two cards on two days on the same unit, with motor temperature logged (the logger
  records it).
  Temperature-dependent joint friction, with repeat-day identification, is documented for industrial arms (Hao et al.
  2021; Pagani et al. 2020), not for this gripper; the step here is between batches uploaded four days apart (upload
  dates, not recording dates), so temperature is one candidate among those listed in §0.
- Unit B's second session (operator-50ep has 23 frames), and whether "newarm" is a second arm (the author).
- The four unreadable parquet files: the same error text appears in huggingface/lerobot issue #2035 and on several other
  v3 dataset pages; cause and fix not confirmed.
- The hold-gap offset of §6.2 as a second, static measure of the wind-up.
- A soft object recorded with command or torque columns, or at ≥ 120 fps, on the same unit and camera as a rigid control;
  the jitter at 1920×1200; the real camera pose (the official MJCF camera does not match the video).
- Real recordings in the official dataset format (which stores arm torque) would be a second torque source; none found
  on Hugging Face as of 2026-10-13 (enactic's two sets are simulated).
- Where the friction sits relative to the encoder (motor side or finger side): decides whether a torque dither can
  linearise the floor; a protocol step exists (P6/P7 in the logger), not run.

## 12. Reproduce

Python 3.10, numpy, pyarrow, ffmpeg; run with `python3 -I` with the downloaded data in a separate directory. In
`floor/`: `hf_survey.py` (which Hugging Face OpenArm datasets log torque), `hf_pull.py` (download), `stiction.py`,
`band.py`, `torque_look.py`, `unit_floor.py`, `within_stroke.py` (§2–4.2), `f8_sessions.py` (§4.3; dates from the
Hugging Face commits API), `f9_direction.py` (§4.4), `padcal.py` (§6.1–6.2), `strawberry_ratio.py` (§6.3),
`mjcf_floor.py` and `pixel_sens.py` (§5, §6.2 computed parts), `d2_analyze.py` (§8), `floor_fig.py` (figure);
supporting: `steady_spread.py` (kinetic level and scatter per speed band), `hold_quiet.py` (encoder creep during holds),
`ep_montage.py` (traces and frame montages per episode), `squeeze2.py` (first version of the §6.2 grasp detector). In `tools/`: `openarm_floor_logger.py`. Notes with every intermediate table: `notes/floor_ja.md` (F1–F9, D2, C1),
`notes/f8_sessions_ja.md` + `f8_table.txt`, `notes/f9_direction_ja.md` + `f9_table.txt`, `notes/f7_strawberry_ja.md`,
`notes/d3_pixel_ja.md`, `notes/d2_protocol_en.md`. Runtime: minutes on a laptop per script;
the video passes decode 15k (§6.2) and 77k (§6.3) frames once.

## Sources

- Datasets (Hugging Face, LeRobot v3): https://huggingface.co/datasets/gtgando/openarm-pick-place-100ep-v2 and the
  other gtgando sets under https://huggingface.co/gtgando (names in §1 and `notes/f8_sessions_ja.md`);
  https://huggingface.co/datasets/xhaka3456/openarm_pick_strawberry_and_place_on_white_plate and
  https://huggingface.co/datasets/xhaka3456/openarm_pick_strawberry_and_place_on_white_plate2.
- Upload batches and dates: https://huggingface.co/api/datasets/gtgando/NAME/commits/main (NAME = dataset name) and each
  commit's
  `meta/info.json` (retrieved 2026-10-12).
- Unreadable parquet, same error text: https://github.com/huggingface/lerobot/issues/2035 (cause not confirmed).
- Official simulator: https://github.com/enactic/openarm_mujoco (openarm_bimanual.xml, class motor_finger).
- Friction yaml: https://github.com/enactic/openarm_teleop (config/follower.yaml, leader.yaml; unchanged since the v1.0
  import).
- Gripper mapping: https://github.com/enactic/openarm_ros2 (openarm_hardware, pinch_gripper 1:1 "approximate").
- System-identification request: https://github.com/enactic/openarm/issues/336
- LeRobot OpenArm driver: https://github.com/huggingface/lerobot (robots/openarm_follower).
- Hardware docs: https://docs.openarm.dev (What's new in 2.0; Hardware › Gripper; 1.0 gripper specifications).
- Damiao DM-J4310-2EC (torque limits, MIT protocol scales, 10:1 reduction, output-side encoder):
  https://github.com/dmBots/dmBot and the vendor pages cited in `notes/floor_ja.md`.
- Nearest prior work found when searching these claims in their own words (none measures them):
  wrist RGB-D on deforming soft fingers for force, arXiv 2605.00307; eye-in-hand camera to contact pressure,
  arXiv 2303.07344; proprioceptive grasping with a series-elastic gripper, arXiv 1803.09674; actuator friction
  identification bench (BAM), https://bam.readthedocs.io.
  Closest in kind, checked in the original text on 2026-10-13: TF-Gripper, arXiv 2602.10013 (force range 0.45–45 N
  with the lower bound "the deadband ... below which the motor torque cannot overcome the transmission friction";
  force from tactile sensors, not motor current); MIM-VLA, arXiv 2610.08425 ("Joint position alone also cannot
  distinguish free closing, contact, stable holding, and continued compression"; motor current used uncalibrated);
  sensorless damage-safe grasping, arXiv 2608.23983 (simulation); openarm_description issue #77 (joint-7 parameter
  identification from the same MIT torque feedback).
- Simulator parameter history: https://github.com/enactic/openarm_mujoco/pull/75 and
  https://github.com/enactic/openarm_mujoco/pull/76 (merged 2026-09-24 and 2026-09-28); raw v2 file https://raw.githubusercontent.com/enactic/openarm_mujoco/master/v2/openarm_bimanual.xml.
- Also checked in the original text, 2026-10-13: Pavlic et al., arXiv 2609.12737 (IROS 2026; safe grasping of fragile
  objects with a coarse position-controlled gripper, contact estimated from a force/torque sensor plus adaptive
  admittance: the sensor this pipeline lacks); Hao et al. 2021, Robotics 10(1):49, doi 10.3390/robotics10010049 (joint
  friction of two industrial arms changes with operating temperature, repeat-day identification); Pagani et al. 2020
  (friction falls as joint temperature rises, 6-DOF manipulator); Nouri 2004, ISA Trans. 43:205–216 (presliding friction
  position-dependent, gross sliding velocity-dependent; pneumatic axis). Not verifiable from here and therefore not
  cited: a 2025 Science Robotics paper (10.1126/scirobotics.adq5046; page blocked) and a 2026 ICCA paper found only
  by name.
- OpenArm dataset format (stores arm qtorque): https://docs.openarm.dev/dataset/ and
  https://github.com/enactic/openarm_dataset; the only Hugging Face sets under enactic are MuJoCo cube sets (simulated).
