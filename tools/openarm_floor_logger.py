#!/usr/bin/env python3
"""D2: ten-minute friction-floor measurement of one OpenArm 2.0 gripper, logged to one CSV.

Two grippers of the same design differ several-fold in friction in public torque-logged data (gtgando/openarm-pick-place-*,
floor/d2_analyze.py on the dataset roots: closing |tau| at 40-100 deg/s p50 0.70 Nm on unit A, 0.20 Nm on unit B; push-still
closing p90 0.545 vs 0.136 Nm). Grasp forces below that friction never show in the logged torque, so the floor is a
per-unit quantity. This script measures it on the gripper only and writes one CSV for floor/d2_analyze.py.
Arm owners: read notes/d2_protocol_en.md first.

Phases (angles follow the range found in P0; the 1 deg/s span is shortened so the whole run fits --budget-s, default 600):
  P0  range: kp --p0-kp (10), 1 deg/s toward closed, then toward open, until tau >= --stop-torque with the jaw stalled
      1.5 s. Working range = [open stop + 2, closed stop - 2] deg.
  P1  unloaded triangle strokes at kp 16: 1 deg/s x2 round trips (central span), then 5, 20, 50 deg/s x3 over the range.
  P2  creep: 0.2 deg/s over 5 deg x4 (close, open, close, open) at mid-range. Each run is approached 8 deg in its own
      direction, then the command is set to the jaw position (torque ~0) and held 3 s, so every ramp starts from rest.
  P3  hold quiet: q_cmd fixed at mid-range 30 s, then at the open end 30 s.
  P5  drift bracket: the P1 1 deg/s strokes again, so start-vs-end friction and motor temperatures can be compared.
  P4  optional, separate run (--bottle KG --finger 1|2 --append CSV): water-bottle lever calibration, 2 deg/s, +-6 deg,
      x3 round trips without and x3 with the bottle on one finger.
  P6  optional phase (--phases ...,P6, or --phases P6 alone with --assume-range): dither linearization, unloaded. The P2
      creep (0.2 deg/s over 5 deg from rest) with a sine of amplitude a and frequency f added to the command, a from
      --dither-amps (default 0,0.5,1,2,3 deg and 0 again, to bracket warming), f = --dither-hz (2), plus one run at 1 deg
      and 5 Hz; closing, then opening. Question: at which amplitude does the cycle-mean torque of a free jaw lose its
      friction offset and its scatter (classical dither), and from which amplitude does the encoder show the jaw reversing.
      In the mock the jaw reverses fully once kp a exceeds the friction (A: 2.4 deg); below that it moves in bursts and
      the offset shrinks gradually while the scatter drops early. The real thresholds are the measurement.
  P7  optional, separate run (--scale --finger 1|2 --append CSV): the same dither while that finger presses on a kitchen
      scale, so the pad cannot move and only the motor side of the wind-up can. Finds the contact at 1 deg/s (kp 10, stall
      at 1 Nm), backs off 2 deg, sends 3 marker steps (sync marks for a video of the scale display), then for each a:
      the command ramps 0.2 deg/s into the scale until the cycle-mean logged torque reaches --press-torque (1.2 Nm), then
      back out (the scale sees the wind-up trapped by the friction, so the first press may start loaded). Scale reading vs
      applied torque, press and release as one loop, gives the friction still hidden between the commanded torque and the
      pad at each amplitude (the loop's half width; the mock predicts max(0, f_s - kp a) because the dither peaks do the
      breakaway, and a force swing resets Coulomb friction at the pad too, so this does not separate motor-side from
      pad-side friction). The encoder's own dither amplitude while the pad is blocked does: the motor side starts
      slipping against the wind-up once kp a exceeds the friction upstream of the encoder. Readings are transcribed from
      the video into a CSV (t_s = CSV time, kg) for d2_analyze.py --scale-csv; the mock logs its object torque as 'obj'.

LeRobot facts this relies on (huggingface/lerobot @ 24f15b92b556, paths under src/lerobot/):
  robots/openarm_follower/config_openarm_follower.py:69-71  use_velocity_and_torque defaults to False (no .vel/.torque)
    :102-105 default gripper kp 25.0, kd 0.3   :107-120 with side=None the gripper command is clipped to [-5, 0] deg
    :31,42 gripper limits (-65, 0) deg for both sides, i.e. open is negative
  robots/openarm_follower/openarm_follower.py:56-58 positions in degrees   :143-147 connect() runs calibrate(), which
    re-zeroes the motors (:188-198, gripper closed = 0), when no calibration file exists; this script uses calibrate=False
    :266-342 send_action(action, custom_kp, custom_kd) sends MIT (kp, kd, q, v=0, tau_ff=0) for the given keys only,
    after clipping to config.joint_limits   :237-244 get_observation reads all 8 motors and exposes pos/vel/torque only
  motors/damiao/damiao.py:491-527 the reply to an MIT command is the motor's feedback frame and updates the state cache, so
    one CAN round trip per tick returns the gripper state   :553-581 the decode keeps the temperatures (bytes 6-7) in the
    cache but drops the status nibble (byte 0) and the raw integers; this script wraps bus._process_response (:569) to keep
    the 8 raw bytes of every gripper frame. Frame layout cross-checked with enactic/openarm_can dm_motor_control.cpp:100-106.
usage: python openarm_floor_logger.py --port can0 --side right --id my_follower --unit-note "..." --thermal cold
       python openarm_floor_logger.py --dry-run          python openarm_floor_logger.py --mock A --out mockA.csv"""
import argparse, collections, csv, datetime, math, os, random, signal, sys, threading, time

LEROBOT_COMMIT = "24f15b92b556cb861ca65e989462e23505a701bd"
PMAX, VMAX, TMAX = 12.5, 30.0, 10.0      # DM4310 MIT scaling: lerobot motors/damiao/tables.py:100, openarm_can constants:152
Q_LSB, V_LSB, T_LSB = math.degrees(2 * PMAX / 65535), math.degrees(2 * VMAX / 4095), 2 * TMAX / 4095
KP_TOP, KD_TOP = 500.0, 5.0              # 12-bit MIT gain ranges: tables.py:196-197
OPEN_SIGN = -1                           # opening = negative degrees (see docstring); confirmed by the probe before P0
CLOSE = -OPEN_SIGN                       # sign of q when closing; logged 'dir' is +1 closing, -1 opening
ARM = tuple(f"joint_{i}" for i in range(1, 8))
SPEED_CAP, G = 50.0, 9.81
P0_SOFT = (-90.0, 20.0)                  # command limits before P0 has found the stops; P0 torque is bounded by its lag cap
COLS = ["run", "t", "phase", "tag", "seg", "dir", "vcmd", "bottle", "finger", "kp", "kd", "q_cmd", "q", "v", "tau",
        "q_raw", "v_raw", "tau_raw", "err", "t_mos", "t_rotor", "fresh", "obj"]


class Abort(Exception):
    pass


class Stop(Exception):
    pass


def enc_gain(x, top):
    """Gain the motor actually receives: damiao.py:446-447 and :529-534 encode kp/kd over 12 bits with int() truncation."""
    return int(max(0.0, min(top, x)) / top * 4095) * top / 4095


def u2f(u, top, bits):
    return u / ((1 << bits) - 1) * 2 * top - top                        # damiao.py:536-540


def f2u(x, top, bits):
    return int(round((max(-top, min(top, x)) + top) / (2 * top) * ((1 << bits) - 1)))


def decode_frame(b):
    """MIT feedback frame: byte0 = status << 4 | id, q 16 bit, v 12 bit, tau 12 bit, T_mos, T_rotor (degC)."""
    q_u, v_u, t_u = (b[1] << 8) | b[2], (b[3] << 4) | (b[4] >> 4), ((b[4] & 0xF) << 8) | b[5]
    return {"q": math.degrees(u2f(q_u, PMAX, 16)), "v": math.degrees(u2f(v_u, VMAX, 12)), "tau": u2f(t_u, TMAX, 12),
            "q_raw": q_u, "v_raw": v_u, "tau_raw": t_u, "err": b[0] >> 4, "t_mos": b[6], "t_rotor": b[7]}


# ----------------------------------------------------------------------------------------------------------- mock motor
class MockMotor:
    """Simulated gripper for --mock (closing = +q). MIT law tau = kp (q_cmd - q) - kd v on a small inertia; stick-slip
    Coulomb friction (static = kinetic + ds), closing level flat, opening level linear from the closed to the open end
    (unit A grows toward open); viscous term; a ripple locked to position; stiff end stops; readout white noise and the CAN
    quantization; temperatures warming exponentially, friction scaled by (1 + drift * (T_rotor - T0)). Mock settings only.
    Behind the encoder sits a massless pad on the wind-up spring k_g (F5: ~140 Nm/rad) that carries the share 1 - f_up of
    the friction and, when 'contact' is set, presses on a stiff object (k_obj). With f_up = 1 (default) the pad follows the
    encoder exactly and the model is the one the D2 recovery runs were made with."""
    PRESETS = {
        "A": dict(q_closed=0.3, q_open=-55.7, fk_close=0.65, ds_close=0.03, fo_closed=0.44, fo_open=0.97, ds_open=0.03,
                  visc=0.0004, rip_amp=0.02, rip_steps=32, noise=0.06),
        "B": dict(q_closed=0.5, q_open=-45.1, fk_close=0.17, ds_close=0.05, fo_closed=0.17, fo_open=0.17, ds_open=0.05,
                  visc=0.0002, rip_amp=0.01, rip_steps=40, noise=0.018)}
    COMMON = dict(J=1e-3, k_stop=140.0, c_stop=0.3, q_noise=0.004, v_noise=0.4, T0_rotor=28.0, dT_rotor=8.0,
                  T0_mos=30.0, dT_mos=5.0, T_tau=300.0, drift=0.0, sub_hz=2000.0, v_eps=1e-3,
                  f_up=1.0, k_g=140.0, k_obj=1e4, contact=None)

    def __init__(self, preset, seed=1, drift=0.0, q0=-20.0):
        self.p = dict(self.COMMON, **self.PRESETS[preset]); self.p["drift"] = drift
        self.rng = random.Random(seed)
        self.q, self.v, self.t, self.tau_m, self.load = math.radians(q0), 0.0, 0.0, 0.0, 0.0
        self.qp, self.obj = self.q, 0.0

    def temps(self, t=None):
        p = self.p; f = 1.0 - math.exp(-(self.t if t is None else t) / p["T_tau"])
        return p["T0_mos"] + p["dT_mos"] * f, p["T0_rotor"] + p["dT_rotor"] * f

    def friction(self, qd, closing, T):
        """(kinetic, static) friction [Nm] at qd [deg], temperature T [degC]; ripple included."""
        p = self.p
        if closing:
            fk, ds = p["fk_close"], p["ds_close"]
        else:
            x = min(1.0, max(0.0, (p["q_closed"] - qd) / (p["q_closed"] - p["q_open"])))
            fk, ds = p["fo_closed"] + (p["fo_open"] - p["fo_closed"]) * x, p["ds_open"]
        s = 1.0 + p["drift"] * (T - p["T0_rotor"])
        rip = p["rip_amp"] * math.sin(2 * math.pi * qd / (p["rip_steps"] * Q_LSB))
        return fk * s + rip, (fk + ds) * s + rip

    def frame(self):
        p, r = self.p, self.rng
        qd, vd = math.degrees(self.q) + r.gauss(0, p["q_noise"]), math.degrees(self.v) + r.gauss(0, p["v_noise"])
        tm, tr = self.temps()
        qu, vu = f2u(math.radians(qd), PMAX, 16), f2u(math.radians(vd), VMAX, 12)
        tu = f2u(self.tau_m + r.gauss(0, p["noise"]), TMAX, 12)
        return bytes([0x18, qu >> 8, qu & 0xFF, vu >> 4, ((vu & 0xF) << 4) | (tu >> 8), tu & 0xFF, round(tm), round(tr)])

    def pad(self, q, fr):
        """Quasi-static pad: the spring k_g (q - q_pad) and the external load push it toward closing; the object (once touched)
        and the downstream share of the friction hold it back. fr(closing) -> (kinetic, static) of that share. Returns the
        spring torque seen on the encoder side and sets self.obj (object torque, closing-positive)."""
        p = self.p; kg, L, ko = p["k_g"], self.load, p["k_obj"]
        qc = math.radians(p["contact"]) if p["contact"] is not None else None
        obj = lambda qp: ko * (qp - qc) if qc is not None and qp > qc else 0.0
        n0 = kg * (q - self.qp) + L - obj(self.qp)
        fk, fs = fr(n0 > 0)
        if abs(n0) > fs:
            s = 1.0 if n0 > 0 else -1.0
            qp = q + (L - s * fk) / kg                                     # slides until the spring balances load and friction
            if qc is not None and qp > qc: qp = (kg * q + L - s * fk + ko * qc) / (kg + ko)
            if (qp - self.qp) * s > 0: self.qp = qp
        self.obj = obj(self.qp)
        return kg * (q - self.qp)

    def step(self, q_cmd_deg, kp, kd, dt):
        """Reply with the state on receipt (torque from the previous command), then run dt with the new command."""
        out = self.frame()
        p = self.p; qc = math.radians(q_cmd_deg); n = max(1, round(dt * p["sub_hz"])); h = dt / n
        T = self.temps()[1]; J, b, fu = p["J"], math.degrees(p["visc"]), p["f_up"]
        qlo, qhi = math.radians(p["q_open"]), math.radians(p["q_closed"])
        for _ in range(n):
            q, v = self.q, self.v; qd = math.degrees(q)
            tm = max(-TMAX, min(TMAX, kp * (qc - q) - kd * v)); self.tau_m = tm
            fr = lambda closing: self.friction(qd, closing, T)
            ext = tm - self.pad(q, lambda closing: tuple(x * (1.0 - fu) for x in fr(closing)))
            if q > qhi: ext -= p["k_stop"] * (q - qhi) + p["c_stop"] * v
            elif q < qlo: ext -= p["k_stop"] * (q - qlo) + p["c_stop"] * v
            if abs(v) < p["v_eps"]:
                fk, fs = (x * fu for x in fr(ext > 0))
                if abs(ext) <= fs:
                    self.v = 0.0; continue
                a = (ext - math.copysign(fk, ext)) / J
            else:
                fk = fr(v > 0)[0] * fu
                a = (ext - math.copysign(fk, v) - b * v) / J
            nv = v + a * h
            if v != 0.0 and nv * v < 0: nv = 0.0
            self.v, self.q = nv, q + nv * h
        self.t += dt
        return out


class MockBackend:
    def __init__(self, a):
        self.a, self.m = a, MockMotor(a.mock, a.seed, a.mock_drift, a.mock_start)
        self.m.p["f_up"] = a.mock_f_up
        self.pose = self.bottle = 0.0

    def initial_q(self):
        return math.degrees(self.m.q)

    def step(self, q_cmd, kp, kd, hold_arm, dt):
        r = decode_frame(self.m.step(q_cmd, enc_gain(kp, KP_TOP), enc_gain(kd, KD_TOP), dt)); r["fresh"] = 1; r["obj"] = self.m.obj
        return r

    def user(self, event):
        """What the operator does at a prompt. The bottle pushes its finger toward open: -m g r_eff (closing-positive). The
        scale is a rigid object 1.5 deg beyond the pad; with the finger pointing down its own weight helps closing."""
        a = self.a
        if event == "pose": self.pose = 0.02 if a.finger == 1 else -0.02          # finger weight; cancels in the difference
        elif event == "pose_down": self.pose = -0.02 if a.finger == 1 else 0.02
        elif event == "bottle_on": self.bottle = -a.bottle * G * a.mock_r_eff[a.finger - 1]
        elif event == "bottle_off": self.bottle = 0.0
        elif event == "scale_on": self.m.p["contact"] = math.degrees(self.m.qp) + 1.5
        elif event == "scale_off": self.m.p["contact"] = None
        self.m.load = CLOSE * (self.pose + self.bottle)

    def close(self):
        pass


# ------------------------------------------------------------------------------------------------------- LeRobot backend
class LeRobotBackend:
    def __init__(self, a):
        from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig   # lazy: --mock runs without it
        cfg = OpenArmFollowerConfig(port=a.port, side=a.side, id=a.robot_id, can_interface=a.can_interface,
                                    use_velocity_and_torque=True, max_relative_target=None)
        if a.arm == "ignore": cfg.motor_config = {"gripper": cfg.motor_config["gripper"]}
        self.robot = OpenArmFollower(cfg); bus = self.robot.bus
        missing = [n for n in ("_process_response", "_last_known_states") if not hasattr(bus, n)]
        if missing:                                                    # private names, checked before anything moves
            sys.exit(f"this LeRobot has no bus.{', bus.'.join(missing)}; install the pinned commit {LEROBOT_COMMIT[:12]}")
        self.frames, self.seen, self.raw, self.warned = 0, 0, None, False
        orig = bus._process_response                                   # damiao.py:569-583

        def keep_raw(motor, msg, _orig=orig):
            if motor == "gripper": self.raw, self.frames = bytes(msg.data[:8]), self.frames + 1
            return _orig(motor, msg)
        bus._process_response = keep_raw                               # instance attribute; the class is untouched
        self.robot.connect(calibrate=False)                            # never re-zero the motors
        st = bus.sync_read_all_states()                                # damiao.py:654-673 (cache filled by the enable replies)
        self.arm_hold = {f"{j}.pos": st[j]["position"] for j in ARM if j in st}
        lim = dict(self.robot.config.joint_limits)
        for j in ARM:                                                  # widen so the held pose is never clipped (moved)
            if j in st:
                p = st[j]["position"]; lo, hi = lim.get(j, (p, p)); lim[j] = (min(lo, p), max(hi, p))
        lim["gripper"] = P0_SOFT                                       # our own soft limits apply first
        self.robot.config.joint_limits = lim
        self.q0 = st["gripper"]["position"]
        if self.arm_hold: self.robot.send_action(dict(self.arm_hold))  # hold the arm where it is, LeRobot default gains

    def initial_q(self):
        return self.q0

    def set_gripper_limits(self, lo, hi):
        self.robot.config.joint_limits = dict(self.robot.config.joint_limits, gripper=(lo, hi))

    def step(self, q_cmd, kp, kd, hold_arm, dt):
        act = {"gripper.pos": q_cmd}
        if hold_arm: act.update(self.arm_hold)
        sent = self.robot.send_action(act, custom_kp={"gripper": kp}, custom_kd={"gripper": kd})
        if abs(sent["gripper.pos"] - q_cmd) > 1e-9:
            raise Abort(f"LeRobot clipped the gripper command {q_cmd:.2f} -> {sent['gripper.pos']:.2f} deg")
        st = self.robot.bus._last_known_states["gripper"]              # what get_observation would report (damiao.py:575-581)
        r = decode_frame(self.raw) if self.raw else {k: float("nan") for k in ("q_raw", "v_raw", "tau_raw", "err")}
        if self.raw and not self.warned and max(abs(r["q"] - st["position"]), abs(r["tau"] - st["torque"])) > 1e-6:
            print("WARNING: raw-frame decode differs from LeRobot's decoded state; logging LeRobot's values"); self.warned = True
        r.update(q=st["position"], v=st["velocity"], tau=st["torque"], t_mos=st["temp_mos"], t_rotor=st["temp_rotor"],
                 fresh=int(self.frames != self.seen))
        self.seen = self.frames
        return r

    def close(self):
        self.robot.disconnect()      # disables torque on every configured motor (config disable_torque_on_disconnect=True)


# ---------------------------------------------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, a, dev, out, run_id):
        self.a, self.dev, self.run = a, dev, run_id
        self.w = csv.writer(out); self.out = out
        self.dt = 1.0 / a.rate; self.sim = bool(a.mock) and not a.mock_realtime
        self.n, self.seg, self.t, self.t_prev, self.t_fresh = 0, 0, 0.0, 0.0, 0.0
        self.next = None; self.t0 = time.perf_counter()
        self.q_cmd = dev.initial_q(); self.last = None
        self.kp, self.kd, self.bottle, self.finger = a.kp, a.kd, 0, 0
        self.soft, self.window = P0_SOFT, None
        self.over, self.err0, self.stop_req, self.exiting = 0, None, False, False
        self.arm_every = max(1, round(a.rate / a.arm_hz)); self.dts = []; self.fresh_n = 0
        self.phase_t = collections.OrderedDict()

    def now(self):
        return self.t

    def wait(self):
        if self.sim:
            self.t = self.n * self.dt; return
        if self.next is None: self.next = time.perf_counter()
        while True:
            left = self.next - time.perf_counter()
            if left <= 0: break
            if left > 0.002: time.sleep(left - 0.001)
        now = time.perf_counter()
        if now - self.next > self.dt: self.next = now                 # late: do not try to catch up
        self.next += self.dt; self.t = now - self.t0

    def tick(self, phase, tag, d, vcmd):
        if self.stop_req and not self.exiting: raise Stop()
        self.wait()
        dtp = self.t - self.t_prev if self.n else self.dt
        self.t_prev = self.t
        self.q_cmd = min(max(self.q_cmd, self.soft[0]), self.soft[1])
        r = self.dev.step(self.q_cmd, self.kp, self.kd, self.n % self.arm_every == 0, dtp)
        if self.n: self.dts.append(dtp)
        self.n += 1
        self.w.writerow([self.run, f"{self.t:.5f}", phase, tag, self.seg, d, f"{vcmd:g}", self.bottle, self.finger,
                         f"{self.kp:g}", f"{self.kd:g}", f"{self.q_cmd:.5f}", f"{r['q']:.5f}", f"{r['v']:.4f}",
                         f"{r['tau']:.5f}", r["q_raw"], r["v_raw"], r["tau_raw"], r["err"], r["t_mos"], r["t_rotor"],
                         r["fresh"], f"{r.get('obj', float('nan')):.5f}"])
        self.phase_t[phase] = self.phase_t.get(phase, 0.0) + dtp
        self.check(r)
        self.last = r
        return r

    def check(self, r):
        a, t = self.a, self.t
        if not r["fresh"]:
            if t - self.t_fresh > 0.1: raise Abort(f"no feedback frame from the gripper for {t - self.t_fresh:.2f} s")
            return
        self.t_fresh = t; self.fresh_n += 1
        self.over = self.over + 1 if abs(r["tau"]) > a.tau_abort else 0
        if self.over >= 3:
            raise Abort(f"|tau| {abs(r['tau']):.2f} Nm > --tau-abort {a.tau_abort} Nm for 3 frames at q {r['q']:.2f} deg")
        e = r["err"]
        if e != e: pass                                                 # NaN: no raw frame, nothing to compare
        elif self.err0 is None: self.err0 = e
        elif e != self.err0:
            raise Abort(f"motor status nibble (feedback byte 0 >> 4) changed {self.err0} -> {r['err']}")
        if max(r["t_mos"], r["t_rotor"]) > a.temp_abort:
            raise Abort(f"motor temperature {max(r['t_mos'], r['t_rotor'])} C > --temp-abort {a.temp_abort} C")
        if self.window and not self.window[0] <= r["q"] <= self.window[1]:
            raise Abort(f"jaw at {r['q']:.2f} deg left the measured range {self.window[0]:.1f}..{self.window[1]:.1f}")

    def move(self, phase, tag, target, speed):
        speed = min(speed, self.a.max_speed); self.seg += 1
        d = 1 if target > self.q_cmd else -1
        cap = math.degrees(0.8 * self.a.tau_abort / self.kp); paused = None
        while abs(target - self.q_cmd) > 1e-9:
            if (self.q_cmd - self.last["q"]) * d < cap:
                self.q_cmd += d * min(speed * self.dt, abs(target - self.q_cmd)); paused = None
            else:
                paused = self.t if paused is None else paused
                if self.t - paused > 2.0:
                    raise Abort(f"the jaw stopped following the command at q {self.last['q']:.2f} deg")
            self.tick(phase, tag, d * CLOSE, speed)

    def hold(self, phase, tag, seconds):
        self.seg += 1; end = self.t + seconds
        while self.t < end: self.tick(phase, tag, 0, 0.0)

    def seek(self, phase, tag, dq, stop_torque=None):
        """P0: creep toward a stop (dq = +1/-1 in q) at low gain until tau pushes >= --stop-torque with the jaw stalled.
        P7 uses it with stop_torque 1.0 to find the scale."""
        a = self.a; self.seg += 1; self.kp = a.p0_kp; st = a.stop_torque if stop_torque is None else stop_torque
        cap = math.degrees((st + 0.3) / self.kp); q_start = self.last["q"]; self.q_cmd = q_start
        hist = collections.deque(); t_end = self.t + a.max_travel / a.p0_speed + 30.0
        while True:
            if self.t > t_end: raise Abort("P0 found no end stop in time (command at its limit or the jaw blocked?)")
            if (self.q_cmd - self.last["q"]) * dq < cap: self.q_cmd += dq * a.p0_speed * self.dt
            r = self.tick(phase, tag, dq * CLOSE, a.p0_speed)
            hist.append((self.t, r["q"]))
            while hist[0][0] < self.t - 1.5: hist.popleft()
            if abs(r["q"] - q_start) > a.max_travel:
                raise Abort(f"no stop found within {a.max_travel} deg of travel toward {'closed' if dq * CLOSE > 0 else 'open'}")
            if r["tau"] * dq >= st and self.t - hist[0][0] > 1.45 and abs(r["q"] - hist[0][1]) < 0.3:
                qs = sorted(q for t, q in hist if t > self.t - 0.3); stop = qs[len(qs) // 2]; break
        self.q_cmd = self.last["q"]; self.hold(phase, "relax", 0.5)
        return stop

    def dither(self, phase, tag, d, amp, freq, speed, travel, tau_stop=None):
        """P6 / P7 run: the base command moves d * travel deg at `speed` deg/s with amp sin(2 pi freq t) deg added; the
        amplitude rises over 1 s before the motion and tapers over 1 s after it, so the jaw never sees a step. The lag cap
        applies to the base. With tau_stop the motion ends early once the mean logged torque over one dither cycle (0.5 s
        for amp 0) reaches tau_stop in the direction d. The logged q_cmd includes the dither; the tag carries a and f.
        Returns the base travel done [deg]."""
        speed = min(speed, self.a.max_speed); self.seg += 1
        base = start = self.q_cmd; t0 = self.t; cap = math.degrees(0.8 * self.a.tau_abort / self.kp) - amp
        per = max(1, round(self.a.rate / freq)) if freq > 0 else round(0.5 * self.a.rate)
        hist = collections.deque(maxlen=per); paused = None

        def send(sub, env):
            self.q_cmd = base + env * amp * math.sin(2 * math.pi * freq * (self.t + self.dt - t0))
            r = self.tick(phase, sub, d * CLOSE if sub == tag else 0, speed if sub == tag else 0.0)
            hist.append(r["tau"] * d)
        t1 = self.t
        while self.t - t1 < 1.0: send("rise", self.t - t1)
        while travel - abs(base - start) > 1e-9:
            if tau_stop is not None and len(hist) == per and sum(hist) / per >= tau_stop: break
            if (base - self.last["q"]) * d < cap:
                base += d * min(speed * self.dt, travel - abs(base - start)); paused = None
            else:
                paused = self.t if paused is None else paused
                if self.t - paused > 2.0:
                    raise Abort(f"the jaw stopped following the command at q {self.last['q']:.2f} deg")
            send(tag, 1.0)
        t1 = self.t
        while self.t - t1 < 1.0: send("taper", 1.0 - (self.t - t1))
        self.q_cmd = base
        return abs(base - start)

    def marker(self, phase, n=3, step=5.0, on=0.8, off=0.8):
        """P7 sync marks for the video of the scale display: n command steps of `step` deg toward closing from the backed-off
        command (kp 16: up to 1.4 Nm, below --tau-abort), each `on` s on and `off` s off. Returns their CSV times."""
        base = self.q_cmd; self.seg += 1; marks = []
        for _ in range(n):
            marks.append(self.t); self.q_cmd = base + CLOSE * step; end = self.t + on
            while self.t < end: self.tick(phase, "mark", 0, 0.0)
            self.q_cmd = base; end = self.t + off
            while self.t < end: self.tick(phase, "mark", 0, 0.0)
        return marks

    def ask(self, phase, text, event=None):
        """Print a question and keep commanding the motors (q_cmd held) until the operator presses Enter."""
        print(text, end="" if text.endswith(" ") else "\n", flush=True)
        if self.a.mock:
            if event: self.dev.user(event)
            self.hold(phase, "prompt", 1.0); return "y"
        box = []
        th = threading.Thread(target=lambda: box.append(sys.stdin.readline()), daemon=True); th.start()
        self.seg += 1
        while th.is_alive(): self.tick(phase, "prompt", 0, 0.0)
        return box[0].strip().lower() if box else ""

    def run_plan(self, steps):
        for st in steps:
            if st[0] == "move": self.move(st[1], st[2], st[3], st[4])
            elif st[0] == "ramp": self.move(st[1], st[2], self.q_cmd + st[3], st[4])
            elif st[0] == "relax": self.q_cmd = self.last["q"]               # command = jaw: torque ~0, jaw at rest
            elif st[0] == "dither": self.dither(st[1], st[2], 1 if st[3] > 0 else -1, st[5], st[6], st[4], abs(st[3]))
            else: self.hold(st[1], st[2], st[3])

    def note(self, key, value):
        self.out.write(f"# {key}: {value}\n")


# ------------------------------------------------------------------------------------------------------------------ plan
def main_plan(lo, hi, s1, p6=False, amps=(0.0, 0.5, 1.0, 2.0, 3.0, 0.0), hz=2.0):
    """Protocol after P0 on the working range [lo, hi]; s1 = span of the 1 deg/s strokes (centred); P6 only when asked."""
    mid = 0.5 * (lo + hi)
    op, cl = (lo, hi) if CLOSE > 0 else (hi, lo)
    a, b = mid - CLOSE * s1 / 2, mid + CLOSE * s1 / 2
    S = []

    def tri(ph, o_end, c_end, v, n):
        S.extend([("move", ph, "approach", o_end, 5.0), ("hold", ph, "dwell", 1.0)])
        for _ in range(n):
            S.extend([("move", ph, "stroke", c_end, v), ("hold", ph, "turn", 0.5),
                      ("move", ph, "stroke", o_end, v), ("hold", ph, "turn", 0.5)])
    tri("P1", a, b, 1.0, 2)
    for v in (5.0, 20.0, 50.0): tri("P1", op, cl, v, 3)
    for d in (1, -1, 1, -1):                                          # d = +1 closing run, -1 opening run
        start = mid - CLOSE * d * 2.5                                 # 8 deg run-in reverses friction up to ~1.1 Nm at kp 16
        S.extend([("move", "P2", "approach", start - CLOSE * d * 8.0, 5.0), ("move", "P2", "approach", start, 5.0),
                  ("relax", "P2", "relax"), ("hold", "P2", "dwell", 3.0), ("ramp", "P2", "creep", CLOSE * d * 5.0, 0.2)])
    S.extend([("move", "P3", "move", mid, 5.0), ("hold", "P3", "hold_mid", 30.0),
              ("move", "P3", "move", op, 5.0), ("hold", "P3", "hold_open", 30.0)])
    tri("P5", a, b, 1.0, 2)
    if p6:
        for d in (1, -1):                                             # same approach and dwell as P2, then the dithered creep
            for amp, f in [(x, hz) for x in amps] + [(1.0, 5.0)]:
                start = mid - CLOSE * d * 2.5
                S.extend([("move", "P6", "approach", start - CLOSE * d * 8.0, 5.0), ("move", "P6", "approach", start, 5.0),
                          ("relax", "P6", "relax"), ("hold", "P6", "dwell", 2.0),
                          ("dither", "P6", f"dither{amp:g}@{f:g}", CLOSE * d * 5.0, 0.2, amp, f)])
    S.append(("move", "END", "exit", op, 5.0))
    return S


def p4_plan(center, amp=6.0, v=2.0):
    o, k = center - CLOSE * amp, center + CLOSE * amp
    S = [("move", "P4", "approach", o, v), ("hold", "P4", "dwell", 1.0)]
    for _ in range(3):
        S.extend([("move", "P4", "stroke", k, v), ("hold", "P4", "turn", 0.5),
                  ("move", "P4", "stroke", o, v), ("hold", "P4", "turn", 0.5)])
    return S


def estimate(steps, q):
    out = collections.OrderedDict()
    for st in steps:
        if st[0] == "move": dt = abs(st[3] - q) / min(st[4], SPEED_CAP); q = st[3]
        elif st[0] == "ramp": dt = abs(st[3]) / min(st[4], SPEED_CAP); q += st[3]
        elif st[0] == "dither": dt = abs(st[3]) / min(st[4], SPEED_CAP) + 2.0; q += st[3]
        elif st[0] == "relax": dt = 0.0
        else: dt = st[3]
        out[st[1]] = out.get(st[1], 0.0) + dt
    return out


def p0_estimate(a, q0, closed, opened):
    build = (a.stop_torque - 0.3) / (a.p0_kp * math.radians(a.p0_speed)) + 2.0      # torque build-up, stall window, relax
    return (abs(closed - q0) + abs(closed - opened)) / a.p0_speed + 2 * build


def fit_plan(a, closed, opened, t_used):
    """Working range and the 1 deg/s span that fits the budget (P1 and P5 shrink together, never below 10 deg)."""
    lo, hi = min(closed, opened) + 2.0, max(closed, opened) - 2.0
    W = hi - lo; s1 = W
    phases = a.phases.split(",")
    for _ in range(6):
        steps = [st for st in main_plan(lo, hi, s1, "P6" in phases, a.dither_amps, a.dither_hz) if st[1] in phases or st[1] == "END"]
        tot = t_used + sum(estimate(steps, lo if CLOSE > 0 else hi).values())
        if tot <= a.budget_s or s1 <= 10.0: break
        s1 = max(10.0, s1 - (tot - a.budget_s) / 8.0 - 0.2)
    return lo, hi, s1, steps, tot


def print_budget(t_p0, steps, q, s1, W, total, budget):
    est = estimate(steps, q)
    print(f"time budget (estimate, s): P0 {t_p0:.0f}" + "".join(f", {k} {v:.0f}" for k, v in est.items())
          + f"  | total {total:.0f} s = {total / 60:.1f} min of {budget / 60:.0f} min "
          + ("(fits)" if total <= budget else "(OVER BUDGET)"))
    print(f"   working range {W:.1f} deg; 1 deg/s strokes (P1, P5) over the central {s1:.1f} deg")


# ------------------------------------------------------------------------------------------------------------------ main
def lerobot_version():
    v, head = "not installed", "unknown"
    try:
        import importlib.metadata as md; v = md.version("lerobot")
    except Exception:
        pass
    try:
        import importlib.util
        root = os.path.dirname(os.path.dirname(os.path.dirname(importlib.util.find_spec("lerobot").origin)))
        ref = open(os.path.join(root, ".git", "HEAD")).read().strip()
        head = open(os.path.join(root, ".git", ref[5:])).read().strip() if ref.startswith("ref:") else ref
    except Exception:
        pass
    return v, head


def read_header(path):
    h = {}
    with open(path) as f:
        for line in f:
            if line.startswith("# ") and ": " in line:
                k, v = line[2:].rstrip("\n").split(": ", 1); h[k] = v
    return h


def parse():
    ap = argparse.ArgumentParser(description="D2 friction-floor measurement of one OpenArm 2.0 gripper (see module docstring).")
    g = ap.add_argument_group("robot")
    g.add_argument("--port", default="can0"); g.add_argument("--side", choices=["left", "right"])
    g.add_argument("--id", dest="robot_id", default=None, help="LeRobot robot id (selects the calibration file)")
    g.add_argument("--can-interface", default="socketcan")
    g.add_argument("--arm", choices=["hold", "ignore"], default="hold",
                   help="hold: command the 7 arm joints to stay where they are; ignore: talk to the gripper motor only")
    g.add_argument("--arm-hz", type=float, default=50.0, help="rate of the arm hold commands")
    g = ap.add_argument_group("record")
    g.add_argument("--unit-note", default="", help="free text: serial, firmware, assembly date, anything known")
    g.add_argument("--thermal", choices=["cold", "running"], help="cold: powered up < 10 min ago; running: used before")
    g.add_argument("--out", default=None)
    g = ap.add_argument_group("protocol and safety")
    g.add_argument("--kp", type=float, default=16.0); g.add_argument("--kd", type=float, default=0.3)
    g.add_argument("--p0-kp", type=float, default=10.0); g.add_argument("--p0-speed", type=float, default=1.0)
    g.add_argument("--stop-torque", type=float, default=2.0, help="P0 end-stop detection [Nm]")
    g.add_argument("--max-travel", type=float, default=80.0, help="P0 aborts without a stop within this travel [deg]")
    g.add_argument("--tau-abort", type=float, default=3.0); g.add_argument("--temp-abort", type=float, default=70.0)
    g.add_argument("--max-speed", type=float, default=SPEED_CAP); g.add_argument("--rate", type=float, default=200.0)
    g.add_argument("--budget-s", type=float, default=600.0)
    g.add_argument("--phases", default="P0,P1,P2,P3,P5", help="main-run phases; without P0 the range is --assume-range")
    g.add_argument("--range-check", type=float, nargs=2, default=[30.0, 75.0], help="plausible range of motion [deg]")
    g.add_argument("--dry-run", action="store_true"); g.add_argument("--assume-range", type=float, nargs=2, default=[-45.1, 0.5])
    g.add_argument("--assume-start", type=float, default=-20.0)
    g = ap.add_argument_group("P4 bottle calibration (separate run)")
    g.add_argument("--bottle", type=float, default=None, metavar="KG"); g.add_argument("--finger", type=int, choices=[1, 2])
    g.add_argument("--append", default=None, help="CSV of the main run to append the P4 / P7 rows to")
    g = ap.add_argument_group("P6 dither phase and P7 scale press (separate run)")
    g.add_argument("--dither-amps", default="0,0.5,1,2,3,0", help="command dither amplitudes [deg] in run order, 0..3")
    g.add_argument("--dither-hz", type=float, default=2.0)
    g.add_argument("--scale", action="store_true", help="P7: this finger presses on a kitchen scale (needs --finger, --append)")
    g.add_argument("--press-torque", type=float, default=1.2, help="P7: cycle-mean logged torque at which a press turns back [Nm]")
    g = ap.add_argument_group("mock")
    g.add_argument("--mock", choices=["A", "B"], default=None); g.add_argument("--mock-drift", type=float, default=0.0)
    g.add_argument("--mock-f-up", type=float, default=1.0, help="mock: share of the friction on the motor side of the wind-up")
    g.add_argument("--mock-r-eff", type=float, nargs=2, default=[0.058, 0.058]); g.add_argument("--mock-start", type=float, default=-20.0)
    g.add_argument("--mock-realtime", action="store_true"); g.add_argument("--seed", type=int, default=1)
    g.add_argument("--yes", action="store_true", help="(mock only) answer prompts automatically")
    a = ap.parse_args()
    if a.max_speed > SPEED_CAP: ap.error(f"--max-speed is capped at {SPEED_CAP} deg/s")
    try: a.dither_amps = tuple(float(x) for x in a.dither_amps.split(","))
    except ValueError: ap.error("--dither-amps: comma-separated degrees")
    a.dither_amps_text = ",".join(f"{x:g}" for x in a.dither_amps)
    if not 0.0 <= min(a.dither_amps) <= max(a.dither_amps) <= 3.0 or a.dither_hz <= 0 or a.dither_hz > 10:
        ap.error("--dither-amps must lie in 0..3 deg (kp 16: 3 deg = 0.84 Nm of ripple) and --dither-hz in 0..10")
    if a.press_torque > 2.0: ap.error("--press-torque is capped at 2.0 Nm (plus up to 0.84 Nm of dither ripple)")
    if a.dry_run: return a
    if a.bottle is not None and (a.finger is None or a.append is None): ap.error("P4 needs --finger and --append RUN_CSV")
    if a.scale and (a.finger is None or a.append is None): ap.error("P7 needs --finger and --append RUN_CSV")
    if a.scale and a.bottle is not None: ap.error("P4 and P7 are separate runs")
    if not a.mock and (a.side is None or a.thermal is None or not a.unit_note):
        ap.error("on the robot, --side, --thermal and --unit-note are required")
    if not a.mock and a.yes: ap.error("--yes is for --mock only")
    return a


def dry_run(a):
    closed, opened = max(a.assume_range), min(a.assume_range)
    if CLOSE < 0: closed, opened = opened, closed
    t0 = p0_estimate(a, a.assume_start, closed, opened)
    lo, hi, s1, steps, tot = fit_plan(a, closed, opened, t0)
    print(f"dry run, assumed stops: closed {closed:.1f}, open {opened:.1f} deg; start {a.assume_start:.1f} deg; rate {a.rate:g} Hz")
    t, q = t0, (lo if CLOSE > 0 else hi)
    print(f"   0:00  P0 range finding, kp {a.p0_kp:g}, {a.p0_speed:g} deg/s to each stop (~{t0:.0f} s)")
    last = None
    for st in steps:
        dt = sum(estimate([st], q).values())
        if st[0] == "dither":
            if last != "P6":
                print(f"  {int(t // 60):2d}:{int(t % 60):02d}  P6 dithered creep runs 0.2 deg/s over 5 deg: a = {a.dither_amps_text} deg at "
                      f"{a.dither_hz:g} Hz, then 1 deg at 5 Hz; closing, then opening")
                last = "P6"
            t += dt; q += st[3]; continue
        key = (st[1], st[2] if st[2] in ("stroke", "creep", "hold_mid", "hold_open") else "", st[4] if st[0] in ("move", "ramp") else 0)
        if st[2] in ("stroke", "creep", "hold_mid", "hold_open", "exit") and key != last:
            v = st[4] if st[0] in ("move", "ramp") else 0.0
            what = {"stroke": f"strokes at {v:g} deg/s", "creep": "creep runs 0.2 deg/s over 5 deg",
                    "hold_mid": "hold at mid-range 30 s", "hold_open": "hold at the open end 30 s", "exit": "return to the open end"}[st[2]]
            print(f"  {int(t // 60):2d}:{int(t % 60):02d}  {st[1]} {what}")
            last = key
        t += dt; q = st[3] if st[0] == "move" else q + st[3] if st[0] == "ramp" else q
    print_budget(t0, steps, lo if CLOSE > 0 else hi, s1, hi - lo, tot, a.budget_s)
    print(f"P4 (optional, separate run per finger): ~{sum(estimate(p4_plan(0.5 * (lo + hi)), lo).values()) * 2 + 20:.0f} s each")
    print(f"P7 (optional, separate run per finger, --scale): ~{len(a.dither_amps) * (4.0 + 2 * (1.0 + a.press_torque / a.kp / math.radians(0.2))) + 40:.0f} s each")


def direction_probe(R):
    """Before anything fast: show the first reading, move 3 deg toward open and back at 2 deg/s, ask what was seen."""
    r = R.last
    print(f"first reading: gripper q {r['q']:.2f} deg, v {r['v']:.2f} deg/s, tau {r['tau']:.3f} Nm, T_mos {r['t_mos']} C, "
          f"T_rotor {r['t_rotor']} C, status nibble {r['err']}")
    print("   expected convention: q ~ 0 deg closed, about -45 deg fully open (opening = negative degrees)")
    q0 = R.q_cmd; R.kp = R.a.p0_kp
    R.move("PC", "probe", q0 + OPEN_SIGN * 3.0, 2.0); R.hold("PC", "probe", 0.5); dq = R.last["q"] - q0
    R.move("PC", "probe", q0, 2.0); R.hold("PC", "probe", 0.5)
    print(f"   probe: commanded 3.0 deg toward open, the jaw moved {dq:+.2f} deg")
    ans = R.ask("PC", "Did the fingers OPEN a little and then close back? [y/N] ")
    if ans != "y":
        raise Abort("direction not confirmed: this unit may use the opposite sign; do not continue, report the probe numbers")


def main():
    a = parse()
    if a.dry_run: dry_run(a); return
    p4, p7 = a.bottle is not None, a.scale; sep = p4 or p7                  # separate runs appended to the main CSV
    hdr = read_header(a.append) if sep else {}
    run_id = int(hdr.get("run", "0")) + 1 if sep else 0
    path = a.append if sep else (a.out or datetime.datetime.now().strftime("d2_floor_%Y%m%d_%H%M.csv"))
    if not sep and os.path.exists(path): sys.exit(f"{path} exists; choose another --out")
    out = open(path, "a" if sep else "w", newline="")
    dev = MockBackend(a) if a.mock else LeRobotBackend(a)
    R = Runner(a, dev, out, run_id)
    lv, head = ("mock", "mock") if a.mock else lerobot_version()
    meta = {"run": run_id, "format": "d2-floor-csv 1", "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "lerobot_commit_pinned": LEROBOT_COMMIT, "lerobot_installed": f"{lv} @ {head}", "unit_note": a.unit_note,
            "thermal": a.thermal or "n/a", "kp_requested": a.kp, "kp_sent": f"{enc_gain(a.kp, KP_TOP):.5f}",
            "kd_requested": a.kd, "kd_sent": f"{enc_gain(a.kd, KD_TOP):.5f}", "p0_kp": a.p0_kp, "rate_hz": a.rate,
            "tau_abort_nm": a.tau_abort, "stop_torque_nm": a.stop_torque, "closing_sign": CLOSE, "port": a.port,
            "side": a.side, "arm_mode": a.arm, "lsb": f"q {Q_LSB:.6f} deg, v {V_LSB:.4f} deg/s, tau {T_LSB:.6f} Nm",
            "mock": a.mock or "no"}
    if a.mock: meta.update(mock_params=MockMotor(a.mock).p | {"drift": a.mock_drift}, mock_r_eff=a.mock_r_eff, seed=a.seed)
    if p4: meta.update(phase="P4", bottle_kg=a.bottle, finger=a.finger)
    if p7: meta.update(phase="P7", finger=a.finger, dither_amps=a.dither_amps_text, dither_hz=a.dither_hz, press_torque=a.press_torque)
    for k, v in meta.items(): R.note(k, v)
    if not sep: out.write(",".join(COLS) + "\n")
    if not a.mock and head not in ("unknown", LEROBOT_COMMIT):
        print(f"note: LeRobot HEAD {head[:12]} is not the pinned {LEROBOT_COMMIT[:12]}; this script uses bus._process_response "
              "and bus._last_known_states, check them if it fails")

    def on_sigint(sig, frm):
        if R.stop_req: raise KeyboardInterrupt
        R.stop_req = True
        print("\nCtrl-C: returning the gripper to the open end, then disabling torque (Ctrl-C again: disable now)")
    signal.signal(signal.SIGINT, on_sigint)
    wall0, status, stops = time.perf_counter(), "complete", None
    try:
        R.tick("PC", "settle", 0, 0.0); R.hold("PC", "settle", 0.5)
        if p4:
            stops = [float(x) for x in hdr["p0_stops_closed_open"].split()]
            lo, hi = min(stops) + 2.0, max(stops) - 2.0
            R.soft, R.window = (min(stops) + 1.0, max(stops) - 1.0), (min(stops) - 3.0, max(stops) + 3.0)
            if not a.mock: dev.set_gripper_limits(*R.soft)
            center = 0.5 * (lo + hi); R.kp = a.kp
            R.move("P4", "approach", center, 5.0)
            R.ask("P4", f"Finger {a.finger}: wrist turned so this finger closes straight UP, bottle OFF. Press Enter.", "pose")
            R.run_plan(p4_plan(center))
            R.ask("P4", f"Put the {a.bottle:.3f} kg bottle on finger {a.finger} (full weight on its pad). Press Enter.", "bottle_on")
            R.bottle, R.finger = 1, a.finger
            R.run_plan(p4_plan(center))
            R.bottle = 0
            R.ask("P4", "Take the bottle off. Press Enter.", "bottle_off")
            R.move("P4", "approach", center, 5.0)
        elif p7:
            stops = [float(x) for x in hdr["p0_stops_closed_open"].split()]
            lo, hi = min(stops) + 2.0, max(stops) - 2.0
            R.soft, R.window = (min(stops) + 1.0, max(stops) - 1.0), (min(stops) - 3.0, max(stops) + 3.0)
            if not a.mock: dev.set_gripper_limits(*R.soft)
            center = 0.5 * (lo + hi); R.kp = a.kp
            R.move("P7", "approach", center, 5.0)
            R.ask("P7", f"Finger {a.finger}: wrist turned so this finger closes straight DOWN; nothing within 3 cm of its pad. "
                        "Zero the kitchen scale and start filming its display (keep filming to the end). Press Enter.", "pose_down")
            R.ask("P7", f"Put the scale under finger {a.finger}: platform level, pad 1-3 cm above it. Press Enter.", "scale_on")
            contact = R.seek("P7", "seek_contact", CLOSE, stop_torque=1.0); R.kp = a.kp
            R.note("p7_contact", f"{contact:.4f}"); print(f"P7: contact at q {contact:.2f} deg")
            R.q_cmd = contact - CLOSE * 2.0; R.hold("P7", "backoff", 1.0); R.finger = a.finger
            marks = R.marker("P7")
            print("P7: marker steps at CSV t = " + " ".join(f"{m:.2f}" for m in marks)
                  + " s; the first jump of the scale reading in the video is the first of these")
            for amp in a.dither_amps:
                R.hold("P7", "dwell", 2.0)
                tr = R.dither("P7", f"press{amp:g}@{a.dither_hz:g}", CLOSE, amp, a.dither_hz, 0.2, 10.0, tau_stop=a.press_torque)
                if tr >= 10.0 - 1e-6: raise Abort("P7: the press found no contact within 10 deg (did the scale move?)")
                R.dither("P7", f"release{amp:g}@{a.dither_hz:g}", -CLOSE, amp, a.dither_hz, 0.2, tr)
            marks = R.marker("P7")
            print("P7: end marker steps at CSV t = " + " ".join(f"{m:.2f}" for m in marks) + " s")
            R.ask("P7", "Take the scale away. Press Enter.", "scale_off")
            R.move("P7", "approach", center, 5.0)
        else:
            phases = a.phases.split(",")
            if not a.mock: direction_probe(R)
            q0 = R.last["q"]
            if "P0" in phases:
                closed = R.seek("P0", "seek_close", CLOSE)
                opened = R.seek("P0", "seek_open", -CLOSE)
            else:
                closed, opened = (max(a.assume_range), min(a.assume_range)) if CLOSE > 0 else (min(a.assume_range), max(a.assume_range))
            rng = abs(closed - opened)
            R.note("p0_stops_closed_open", f"{closed:.4f} {opened:.4f}")
            print(f"P0: closed stop {closed:.2f} deg, open stop {opened:.2f} deg, range {rng:.1f} deg")
            if not a.range_check[0] <= rng <= a.range_check[1]:
                raise Abort(f"measured range {rng:.1f} deg is outside {a.range_check}; check the setup (nothing between the fingers?)")
            stops = (closed, opened)
            R.soft, R.window = (min(stops) + 1.0, max(stops) - 1.0), (min(stops) - 3.0, max(stops) + 3.0)
            if not a.mock: dev.set_gripper_limits(*R.soft)
            lo, hi, s1, steps, tot = fit_plan(a, closed, opened, R.t)
            R.note("working_range", f"{lo:.4f} {hi:.4f}"); R.note("span_1dps", f"{s1:.3f}")
            print_budget(R.t, steps, lo if CLOSE > 0 else hi, s1, hi - lo, tot, a.budget_s)
            if [ph for ph in ("P1", "P2", "P3", "P5") if ph not in phases] or "P6" in phases:
                print(f"   (this run executes {','.join(phases)}; the budget counts those phases only)")
            R.kp = a.kp
            R.run_plan([st for st in steps if st[1] in phases or st[1] == "END"])
    except Stop:
        status = "stopped by Ctrl-C"
    except Abort as e:
        status = f"ABORT: {e}"
        print(status)
    except KeyboardInterrupt:
        status = "interrupted (second Ctrl-C)"
    finally:
        try:
            if status.startswith("stopped") and stops:
                R.exiting, R.kp = True, a.kp
                R.move("END", "exit", min(stops) + 2.0 if CLOSE > 0 else max(stops) - 2.0, 10.0)
            elif status.startswith("ABORT") and R.last:
                R.exiting = True; R.q_cmd = R.last["q"]; R.hold("END", "relax", 0.3)
        except BaseException as e:
            print(f"(safe exit skipped: {type(e).__name__} {e})")
        dev.close()
        wall = time.perf_counter() - wall0
        dts = sorted(R.dts) or [R.dt]
        pick = lambda f: dts[min(len(dts) - 1, int(f * len(dts)))]
        rate = R.n / R.t if R.t > 0 else float("nan")
        R.note("end", f"{status}; ticks {R.n}; run time {R.t:.1f} s; rate {rate:.1f} Hz; fresh {R.fresh_n / max(R.n, 1):.3f}")
        out.close()
        print(f"{status}. {R.n} ticks in {R.t:.1f} s -> achieved loop rate {rate:.1f} Hz (target {a.rate:g}); tick interval "
              f"p50 {pick(0.5) * 1e3:.2f} / p99 {pick(0.99) * 1e3:.2f} / max {dts[-1] * 1e3:.2f} ms; fresh frames "
              f"{R.fresh_n / max(R.n, 1):.1%}" + (f"; simulated, wall {wall:.1f} s ({R.n / wall:.0f} ticks per wall second)"
                                                  if R.sim else ""))
        print("time per phase (s): " + ", ".join(f"{k} {v:.0f}" for k, v in R.phase_t.items())
              + f"; total {R.t:.0f} s = {R.t / 60:.1f} min")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
