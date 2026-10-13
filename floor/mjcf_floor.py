"""Official OpenArm v2 MJCF with the gripper "floor" physics measured on two real Gripper 2.0 units (notes/floor_ja.md).

What is changed (in memory, through mujoco.MjSpec; the official file is never written):
  - finger frictionloss: the measured closing-direction Coulomb friction at the motor, 0.6 Nm (unit A) or 0.15 Nm (unit B),
    put on the driven hinge (finger_joint1) only; finger_joint2 follows through the official equality (polycoef 0 1 0 0 0)
    and keeps frictionloss 0, so the torque the actuator needs to move the pair is exactly the unit value.
  - gripper actuator: kp 16, kv 0.3 (the gains the public datasets ran with; MIT position servo tau = kp (q_cmd - q) - kv v).
  - wrist cameras: resolution 1920 x 1200 (Arducam B0495 full frame), fovy kept at the official 66 deg.
  - optional: finger damping (official 0.6 Nm s/rad on each finger hinge, 1.2 at the motor). The real logs show no viscous
    term between 21 and 290 deg/s (floor_ja.md F2-4), so --damping 0 is the measured-consistent choice; it is not the default
    because the default keeps every official value that the measurement does not replace.
  - option noslip_iterations 10 (official 0). MuJoCo's dof frictionloss is a soft constraint: without the noslip pass the
    finger creeps through it, and at 2 deg/s the actuator holds only ~0.11 Nm against "0.6 Nm" of frictionloss (self-test).
    The noslip post-solver removes that drift in friction dimensions (MuJoCo docs, option/noslip_iterations), so the
    torque needed to move the finger becomes the unit value. It also stiffens contact friction. --noslip 0 restores it.
Not changed: the motor-to-hinge ratio is 1 (the MJCF drives the hinge directly); the real ratio (1 or 1.25) is unknown.

CAN readback quantization (DM4310 MIT feedback frame, floor_ja.md F2-1): position 16 bit over +-12.5 rad (0.021854 deg),
velocity 12 bit over +-30 rad/s (0.8395 deg/s), torque 12 bit over +-10 Nm (0.004884 Nm). quantize_obs() maps a hinge state
to what the motor would report (dataset sign: q = 0 closed, negative open).

usage: python mjcf_floor.py [--unit A|B] [--frictionloss F] [--damping B] [--noslip N] [--selftest [--side left|right]]
"""
import argparse
import os

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_XML = os.path.normpath(os.path.join(HERE, "..", "openarm", "v2", "openarm_bimanual.xml"))

UNIT_FRICTION = {"A": 0.60, "B": 0.15}      # Nm at the motor, closing direction (floor_ja.md F2-6, F2-8)
KP, KV = 16.0, 0.3                          # dataset gains (floor_ja.md F2-2)
CAM_RES = (1920, 1200)
NOSLIP = 10                                 # noslip_iterations; see the module docstring

Q_LIM, Q_BITS = 12.5, 16                    # rad
V_LIM, V_BITS = 30.0, 12                    # rad/s
T_LIM, T_BITS = 10.0, 12                    # Nm
Q_STEP = 2 * Q_LIM / (2 ** Q_BITS - 1)
V_STEP = 2 * V_LIM / (2 ** V_BITS - 1)
T_STEP = 2 * T_LIM / (2 ** T_BITS - 1)


def finger_names(side):
    return (f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2", f"{side}_finger1_ctrl", f"camera_wrist_{side}")


def floor_spec(xml=OFFICIAL_XML, frictionloss=UNIT_FRICTION["A"], kp=KP, kv=KV, damping=None, cam_res=CAM_RES,
               sides=("left", "right"), noslip=NOSLIP):
    """MjSpec of the official model with the floor physics applied. damping=None keeps the official finger damping;
    a number puts that total (Nm s/rad at the motor) on finger_joint1 and zeroes finger_joint2. noslip sets
    option/noslip_iterations (0 = official)."""
    spec = mujoco.MjSpec.from_file(xml)
    spec.option.noslip_iterations = int(noslip)
    for side in sides:
        j1n, j2n, an, cn = finger_names(side)
        j1, j2 = spec.joint(j1n), spec.joint(j2n)
        j1.frictionloss = float(frictionloss)
        j2.frictionloss = 0.0
        if damping is not None:
            j1.damping = [float(damping), 0.0, 0.0]
            j2.damping = [0.0, 0.0, 0.0]
        a = spec.actuator(an)
        gp, bp = np.array(a.gainprm), np.array(a.biasprm)
        gp[0], bp[1], bp[2] = kp, -kp, -kv
        a.gainprm, a.biasprm = gp, bp
        cam = spec.camera(cn)
        cam.resolution = list(cam_res)
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), cam_res[0])
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), cam_res[1])
    return spec


def load_model(**kw):
    return floor_spec(**kw).compile()


def _quant(x, lim, step):
    u = np.round((np.clip(x, -lim, lim) + lim) / step)
    return u * step - lim


def quantize_obs(theta, theta_dot, tau_hinge, ratio=1.0, sign=-1.0):
    """Hinge angle [rad], rate [rad/s], torque [Nm] -> motor readback (q [rad], v [rad/s], tau [Nm]) quantized like the CAN
    feedback. ratio = motor angle / hinge angle; sign = -1 maps the left hinge (0 closed, + open) to the dataset convention
    (0 closed, negative open). Rounding to the nearest code is assumed (the firmware's encoder is not public)."""
    q = sign * ratio * np.asarray(theta, float)
    v = sign * ratio * np.asarray(theta_dot, float)
    t = sign * np.asarray(tau_hinge, float) / ratio
    return _quant(q, Q_LIM, Q_STEP), _quant(v, V_LIM, V_STEP), _quant(t, T_LIM, T_STEP)


# ----------------------------------------------------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------------------------------------------------

def _hand_box(spec, side, size_y, half_x=0.010, half_z=0.012, z_c=-0.163):
    """Rigid box held between the fingertips. It hangs off ee_base_link through a slide joint with a +-1 um range: a
    jointless child would be merged into the hand's weld body, and MuJoCo's parent filter then drops every finger-box
    contact (the fingers' parent is that weld body)."""
    hand = spec.body(f"openarm_{side}_ee_base_link")
    b = hand.add_body()
    b.name = "selftest_box"
    b.pos = [-0.00143, 0.0, z_c]
    j = b.add_joint()
    j.name = "selftest_box_slide"
    j.type = mujoco.mjtJoint.mjJNT_SLIDE
    j.axis = [1.0, 0.0, 0.0]
    j.range = [-1e-6, 1e-6]
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    g = b.add_geom()
    g.name = "selftest_box_geom"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [half_x, size_y / 2, half_z]
    g.rgba = [0.9, 0.1, 0.1, 1]
    return b


def _run(m, d, side, ctrl_fn, T, box_geom=-1):
    """Step T seconds with ctrl_fn(t) on the finger actuator; record t, hinge q, hinge v, actuator force, box contacts."""
    j1n, _, an, _ = finger_names(side)
    j1 = m.joint(j1n)
    a = m.actuator(an).id
    adr, dof = j1.qposadr[0], j1.dofadr[0]
    n = int(round(T / m.opt.timestep))
    out = np.zeros((n, 5))
    for i in range(n):
        t = i * m.opt.timestep
        d.ctrl[a] = ctrl_fn(t)
        mujoco.mj_step(m, d)
        nb = 0
        if box_geom >= 0:
            for c in d.contact[:d.ncon]:
                if c.geom1 == box_geom or c.geom2 == box_geom:
                    nb += 1
        out[i] = (t, d.qpos[adr], d.qvel[dof], d.actuator_force[a], nb)
    return out


def selftest(frictionloss, damping, noslip, side="left"):
    """Runs in opening-positive hinge coordinates: the right hinge opens toward negative angles, so its states, command and
    torque are mirrored with sg = -1 (printed values are the same quantities for either side)."""
    j1n, j2n, an, _ = finger_names(side)
    sg = 1.0 if side == "left" else -1.0
    th0 = np.radians(40.0)
    print(f"self-test: {side} gripper, frictionloss {frictionloss} Nm on finger_joint1, kp {KP}, kv {KV}, damping "
          f"{'official (0.6 + 0.6)' if damping is None else damping}, noslip_iterations {noslip}; gravity off, arm at qpos 0")

    # 1. closing on nothing: ramp the command toward closed at a constant rate from rest at 40 deg
    m = floor_spec(frictionloss=frictionloss, damping=damping, noslip=noslip).compile()
    m.opt.gravity[:] = 0
    bsum = sum(m.dof_damping[m.joint(jn).dofadr[0]] for jn in (j1n, j2n))
    print("  1) free closing from rest at 40 deg (no object); command ramps toward closed")
    for rate_deg in (0.5, 2.0, 10.0, 40.0):
        d = mujoco.MjData(m)
        for jn in (j1n, j2n):
            d.qpos[m.joint(jn).qposadr[0]] = sg * th0
        rate = np.radians(rate_deg)
        T = (1.5 + np.radians(35) / rate) if rate_deg >= 2 else 8.0
        rec = _run(m, d, side, lambda t: sg * (th0 - rate * t), T)
        t, q, v, f, _ = rec.T
        q, v, f = sg * q, sg * v, sg * f
        moved = np.nonzero(np.abs(q - th0) > np.radians(0.05))[0]
        brk = abs(f[moved[0]]) if len(moved) else np.nan
        steady = (t > 0.5 * T) if rate_deg < 2 else ((q < th0 - np.radians(10)) & (q > np.radians(5)))
        fs, vs = abs(np.median(f[steady])), np.median(v[steady])
        print(f"     {rate_deg:4.1f} deg/s: |torque| at 0.05 deg of travel {brk:.3f} Nm; steady (n={steady.sum()} steps): "
              f"velocity {np.degrees(vs):+6.2f} deg/s, |actuator torque| {fs:.3f} Nm "
              f"= {frictionloss:.3f} frictionloss {fs - frictionloss:+.3f} (damping {bsum:.2f} x |v| = {bsum * abs(vs):.3f})")

    # 2. closing on a rigid box held between the fingertips
    size = 0.040
    print(f"  2) closing on a rigid box ({size * 1000:.0f} mm across the jaw), command ramps 10 deg/s to the ctrlrange limit")
    spec = floor_spec(frictionloss=frictionloss, damping=damping, noslip=noslip)
    _hand_box(spec, side, size)
    m = spec.compile()
    m.opt.gravity[:] = 0
    d = mujoco.MjData(m)
    for jn in (j1n, j2n):
        d.qpos[m.joint(jn).qposadr[0]] = sg * th0
    rate = np.radians(10.0)
    lo = m.actuator(an).ctrlrange[0] if sg > 0 else -m.actuator(an).ctrlrange[1]     # closed end, opening-positive
    rec = _run(m, d, side, lambda t: sg * max(th0 - rate * t, lo), 6.0, box_geom=m.geom("selftest_box_geom").id)
    t, q, v, f, nc = rec.T
    q, v, f = sg * q, sg * v, sg * f
    on = np.nonzero(nc > 0)[0]
    if len(on) == 0:
        print("     no finger-box contact")
        return
    i0 = on[0]
    sel = (np.arange(len(t)) > i0) & (np.abs(f) > 1.0) & (np.abs(f) < 6.0) & (nc > 0)
    slope = np.polyfit(q[sel], f[sel], 1)[0] if sel.sum() > 5 else np.nan
    print(f"     first box contact at hinge {np.degrees(q[i0]):.2f} deg (t {t[i0]:.2f} s, {int(nc[i0])} contacts); "
          f"|torque| 1 -> 6 Nm over {np.degrees(np.ptp(q[sel])) if sel.any() else np.nan:.4f} deg of hinge travel "
          f"(n={sel.sum()} steps): tau-q slope {abs(slope):.0f} Nm/rad")
    print(f"     compare: real wind-up after contact ~140 Nm/rad at the motor (floor_ja.md); the sim has no structural "
          f"compliance, only the contact's soft constraint")
    print(f"     final: hinge {np.degrees(q[-1]):.3f} deg, torque {f[-1]:+.3f} Nm, command {np.degrees(lo):.1f} deg "
          f"(kp x error = {KP * (lo - q[-1]):+.3f}, kv x v = {KV * v[-1]:+.4f})")
    qq, vv, tt = quantize_obs(q[-1], v[-1], f[-1])
    print(f"     quantized readback (motor convention, ratio 1): q {np.degrees(qq):+.4f} deg, v {np.degrees(vv):+.3f} deg/s, "
          f"tau {tt:+.4f} Nm")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unit", choices=("A", "B"), default="A")
    ap.add_argument("--frictionloss", type=float, default=None, help="overrides --unit (Nm at the motor)")
    ap.add_argument("--damping", type=float, default=None, help="total finger damping at the motor; default official")
    ap.add_argument("--noslip", type=int, default=NOSLIP, help="option/noslip_iterations (0 = official)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--side", choices=("left", "right"), default="left", help="gripper for the self-test")
    a = ap.parse_args()
    fl = UNIT_FRICTION[a.unit] if a.frictionloss is None else a.frictionloss
    m = load_model(frictionloss=fl, damping=a.damping, noslip=a.noslip)
    for side in ("left", "right"):
        j1n, j2n, an, cn = finger_names(side)
        act = m.actuator(an).id
        print(f"{side}: frictionloss j1 {m.dof_frictionloss[m.joint(j1n).dofadr[0]]:.3f} j2 {m.dof_frictionloss[m.joint(j2n).dofadr[0]]:.3f}, "
              f"damping j1 {m.dof_damping[m.joint(j1n).dofadr[0]]:.3f} j2 {m.dof_damping[m.joint(j2n).dofadr[0]]:.3f}, "
              f"kp {m.actuator_gainprm[act, 0]:.1f} kv {-m.actuator_biasprm[act, 2]:.2f}, camera {tuple(m.cam_resolution[m.camera(cn).id])} "
              f"fovy {m.cam_fovy[m.camera(cn).id]:.0f}")
    print(f"noslip_iterations {m.opt.noslip_iterations}")
    print(f"quantization steps: q {np.degrees(Q_STEP):.6f} deg, v {np.degrees(V_STEP):.4f} deg/s, tau {T_STEP:.6f} Nm")
    if a.selftest:
        selftest(fl, a.damping, a.noslip, side=a.side)


if __name__ == "__main__":
    main()
