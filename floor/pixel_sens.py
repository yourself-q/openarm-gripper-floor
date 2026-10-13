"""Wrist-camera pixel sensitivity of the OpenArm 2.0 gripper in the official MJCF (task D3): what the camera can see of
the grasp force below the motor's floor.

The model is mjcf_floor.floor_spec() (official XML, camera at 1920 x 1200, fovy 66 kept). Only kinematics and rendering
are used here. A box (the object) hangs off the left hand between the fingertips; it is placed, not simulated.

(a) Inner-edge pixel of each finger vs hinge angle theta 0..45 deg, by padcal.py's rule (scan each row outward from the
    image centre until 3 consecutive gripper pixels; median over the rows, >= 10 rows). Segmentation replaces padcal's
    darkness test (every gripper pixel counts as dark). Rows: padcal's 280..320 of 480 = 700..800 of 1200 under both
    mappings of the 1920 x 1200 sensor to the 640 x 480 dataset frames:
      hyp i : full frame resized anisotropically      x640 = x / 3,            y480 = y / 2.5
      hyp ii: centre crop to 1600 x 1200, scale 0.4    x640 = (x - 160) * 0.4,  y480 = y * 0.4
    Compared with the real calibration (padcal on the public datasets, unit A and B): slopes, positions, the jaw gap at
    hold, the cube width, the opening range; ratio n = motor angle / hinge angle in {1, 1.25}.
(b) px per mm of pad-face displacement along the closing direction (hand y) at the contact point, for boxes 10..60 mm
    across the jaw: by projection (pinhole Jacobian at the contact point from mj_geomDistance) and by render (box shrunk
    by 2 Delta with the fingers following, edge shift / Delta). Foreshortening angle, visibility (mj_ray), in-frame.
(c) S = (px/mm) / k and the camera force floor F_cam = jitter / S for pad and object compression, next to the motor floor.
(d) floor/pixel_sens.png and a text table.

usage: python pixel_sens.py [--table PATH] [--fig PATH]
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mjcf_floor as MF  # noqa: E402

W, H = MF.CAM_RES
XC = W // 2
BAND = (700, 800)                       # rows at 1200 = padcal rows 280..320 at 480
HYP = {"i": (1.0 / 3.0, 0.0), "ii": (0.4, 160.0)}
ROW_SCALE = 0.4

# object box: x (finger width) and z (along the finger) extents; y extent = size across the jaw
OBJ_X = -0.00143                        # finger plane (hand frame)
OBJ_HX, OBJ_ZC, OBJ_HZ = 0.010, -0.158, 0.010
SIZES_MM = (10, 20, 30, 45, 60)
DELTA_MM = 2.0

# real calibration (floor/padcal.py on the public datasets; still, unloaded frames; motor q bin centre, x_left, x_right
# at 640 x 480, frames per bin). Unit A: gtgando/openarm-pick-place-100ep-v2 ep 0..47. Unit B: openarm-pick-place-newarm.
REAL_CAL = {
    "A": [(-43, 241.18, 453.74, 175), (-41, 247.63, 446.87, 200), (-39, 252.58, 442.79, 1356), (-37, 257.88, 437.62, 1547),
          (-35, 262.28, 432.98, 1027), (-33, 267.67, 428.94, 820), (-31, 273.46, 422.68, 586), (-29, 281.30, 415.30, 128),
          (-27, 283.24, 412.80, 39), (-25, 289.07, 407.60, 75), (-23, 293.92, 403.83, 27)],
    "B": [(-45, 238.50, 468.58, 83), (-43, 245.62, 461.31, 233), (-41, 249.50, 457.61, 541), (-39, 254.08, 452.98, 926),
          (-37, 258.23, 448.23, 486), (-35, 264.54, 443.39, 374), (-33, 270.07, 436.55, 128), (-31, 274.27, 431.91, 25)],
}
# hold (|v| < 2 deg/s, tau > 1.5 Nm): motor q median, jaw gap x_right - x_left median and IQR, frames
REAL_HOLD = {"A": (-19.73, 117.4, (110.1, 121.9), 3062), "B": (-19.29, 107.8, (101.9, 113.5), 2062)}
CUBE_PX = (95.0, 110.0)                 # red width of the held cube (floor_ja.md F5)
Q_OPEN = {"A": -55.7, "B": -45.1}       # most open motor angle seen (floor_ja.md F2-8)
HINGE_MAX = 45.0                        # official MJCF finger range 0..0.7854 rad
RATIOS = (1.0, 1.25)
JITTER = {"A": (0.5, 0.7), "B": (1.0, 2.0)}         # px at 640, one edge, during holds (floor_ja.md F5, F5-2)
MOTOR_FLOOR_N = {"A": (3.0, 5.0), "B": (0.3, 0.7)}  # per finger at the pad, as specified for D3
R_LEVER = (0.060, 0.097)
# motor torques [Nm] that set a floor, for the alternative conversions F = tau / (2 r) (floor_ja.md F2-6, F2-8, P1-2)
TORQUE_FLOORS = [("A closing friction (uncompensated)", 0.60, 0.68), ("B closing friction, kinetic", 0.15, 0.19),
                 ("B pushed-still p50", 0.04, 0.12), ("A within-episode scatter", 0.066, 0.066),
                 ("B within-episode scatter", 0.018, 0.018), ("torque LSB", MF.T_STEP, MF.T_STEP)]
# objects: name, k [N/mm] on total compression (F at each pad / total squeeze), assumed size [mm], source tag
OBJECTS = [("strawberry", 0.5, 30, "task"), ("strawberry", 1.0, 30, "task"), ("grape (P1-3 low)", 0.13, 20, "P1-3"),
           ("grape (P1-3 high)", 0.42, 20, "P1-3"), ("blueberry", 1.33, 10, "P1-3"), ("grape (task low)", 2.0, 20, "task"),
           ("cherry", 5.0, 20, "task"), ("grape (task high)", 14.0, 20, "task"), ("apple", 20.0, 60, "task")]
K_PAD = (5.0, 20.0, 50.0, 200.0)
# F5 (unit A, 32 grasps): from landing to full hold tau rises 2.49 Nm and the jaw gap changes 0.11 x 5.1 px (median)
F5_DTAU, F5_GAP_PX = 2.49, 0.11 * 5.1


def to640(x, hyp):
    """1920-wide x to 640-wide x; outside the 640 frame (hyp ii crops 160 px each side) -> nan."""
    s, x0 = HYP[hyp]
    y = (np.asarray(x, float) - x0) * s
    return np.where((y >= 0) & (y < 640), y, np.nan)


class Scene:
    def __init__(self):
        spec = MF.floor_spec(sides=("left", "right"))
        hand = spec.body("openarm_left_ee_base_link")
        b = hand.add_body()
        b.name = "d3_obj"
        b.pos = [OBJ_X, 0.0, OBJ_ZC]
        g = b.add_geom()
        g.name = "d3_obj_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [OBJ_HX, 0.01, OBJ_HZ]
        g.rgba = [0.85, 0.15, 0.15, 1.0]
        g.contype, g.conaffinity, g.group = 0, 0, 0
        self.m = m = spec.compile()
        self.d = mujoco.MjData(m)
        self.obj = m.geom("d3_obj_geom").id
        self.seg = mujoco.Renderer(m, H, W)
        self.seg.enable_segmentation_rendering()
        self.rgb = None
        self.side = {}
        for side in ("left", "right"):
            j1, j2, _, cam = MF.finger_names(side)
            inner, outer = m.body(f"openarm_{side}_ee_inner_finger").id, m.body(f"openarm_{side}_ee_outer_finger").id
            handb = m.body(f"openarm_{side}_ee_base_link").id
            lut = np.zeros(m.ngeom, np.int8)          # 1 outer finger, 2 inner finger, 3 hand, 4 object
            for gi in range(m.ngeom):
                bi = m.geom_bodyid[gi]
                lut[gi] = 1 if bi == outer else 2 if bi == inner else 3 if bi == handb else 0
            lut[self.obj] = 4 if side == "left" else 0
            coll = {k: [gi for gi in range(m.ngeom) if m.geom_bodyid[gi] == bid and m.geom_group[gi] == 3]
                    for k, bid in (("inner", inner), ("outer", outer))}
            self.side[side] = dict(j1=m.joint(j1).id, j2=m.joint(j2).id, cam=m.camera(cam).id, sign=1.0 if side == "left" else -1.0,
                                   hand=handb, body={"inner": inner, "outer": outer}, joint={"inner": m.joint(j1).id, "outer": m.joint(j2).id},
                                   lut=lut, coll=coll)
        self.f = (H / 2) / np.tan(np.radians(m.cam_fovy[self.side["left"]["cam"]]) / 2)
        self.set_size(20.0)
        self.show_obj(False)

    # --- state -------------------------------------------------------------------------------------------------------
    def set_theta(self, th_deg, side="left"):
        s = self.side[side]
        for j in (s["j1"], s["j2"]):
            self.d.qpos[self.m.jnt_qposadr[j]] = s["sign"] * np.radians(th_deg)
        mujoco.mj_forward(self.m, self.d)

    def set_size(self, size_mm):
        self.m.geom_size[self.obj] = [OBJ_HX, size_mm / 2000.0, OBJ_HZ]
        self.m.geom_rbound[self.obj] = float(np.linalg.norm(self.m.geom_size[self.obj]))

    def show_obj(self, on):
        self.m.geom_group[self.obj] = 0 if on else 5

    # --- camera ------------------------------------------------------------------------------------------------------
    def project(self, p, side="left"):
        c = self.side[side]["cam"]
        R, o = self.d.cam_xmat[c].reshape(3, 3), self.d.cam_xpos[c]
        pc = R.T @ (np.asarray(p) - o)
        return np.array([W / 2 + self.f * pc[0] / -pc[2], H / 2 - self.f * pc[1] / -pc[2]]), -pc[2]

    def classes(self, side="left"):
        self.seg.update_scene(self.d, camera=self.side[side]["cam"])
        s = self.seg.render()
        ids, typ = s[:, :, 0], s[:, :, 1]
        cls = np.zeros((H, W), np.int8)
        ok = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        cls[ok] = self.side[side]["lut"][ids[ok]]
        return cls

    def render_rgb(self, side="left"):
        if self.rgb is None:
            self.rgb = mujoco.Renderer(self.m, H, W)
        self.rgb.update_scene(self.d, camera=self.side[side]["cam"])
        return self.rgb.render()

    # --- contact -----------------------------------------------------------------------------------------------------
    def dist(self, finger, side="left"):
        best = (np.inf, -1, None)
        for g in self.side[side]["coll"][finger]:
            ft = np.zeros(6)
            dd = mujoco.mj_geomDistance(self.m, self.d, g, self.obj, 0.05, ft)
            if dd < best[0]:
                best = (dd, g, ft.copy())
        return best

    def theta_contact(self, size_mm, finger):
        """Hinge angle at which the finger's collision geoms first touch the box while closing (bisection)."""
        self.set_size(size_mm)
        lo, hi = 0.0, HINGE_MAX
        self.set_theta(hi)
        if self.dist(finger)[0] < 0:
            return np.nan
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            self.set_theta(mid)
            if self.dist(finger)[0] < 0:
                lo = mid
            else:
                hi = mid
        self.set_theta(hi)
        return hi

    def theta_c(self, size_mm):
        return max(self.theta_contact(size_mm, "inner"), self.theta_contact(size_mm, "outer"))


def scan_edges(dark, rows):
    """padcal's rule on a boolean mask: per row, the innermost of 3 consecutive dark px left and right of XC."""
    xl, xr = [], []
    for r in rows:
        if r < 0 or r >= H:
            continue
        b = dark[r]
        d3 = b[2:XC + 1] & b[1:XC] & b[0:XC - 1]
        ii = np.nonzero(d3)[0]
        if len(ii):
            xl.append(ii[-1] + 2 + 0.5)
        d3r = b[XC:W - 2] & b[XC + 1:W - 1] & b[XC + 2:W]
        jj = np.nonzero(d3r)[0]
        if len(jj):
            xr.append(jj[0] + XC - 0.5)
    L = float(np.median(xl)) if len(xl) >= 10 else np.nan
    R = float(np.median(xr)) if len(xr) >= 10 else np.nan
    return L, R


def edge_class(cls, x, rows, right=False):
    """Majority class of the first gripper pixel at the edge (left edge: the pixel at floor(x); right edge: the next one)."""
    if not np.isfinite(x):
        return 0
    xi = int(np.clip(np.floor(x) + (1 if right else 0), 0, W - 1))
    v = cls[[r for r in rows if 0 <= r < H], xi]
    v = v[v > 0]
    return int(np.bincount(v).argmax()) if len(v) else 0


def obj_extent(cls, rows):
    lo, hi = [], []
    for r in rows:
        if 0 <= r < H:
            xs = np.nonzero(cls[r] == 4)[0]
            if len(xs):
                lo.append(xs[0] - 0.5)
                hi.append(xs[-1] + 0.5)
    if len(lo) < 10:
        return np.nan, np.nan
    return float(np.median(lo)), float(np.median(hi))


CLS_NAME = {0: "-", 1: "outer finger", 2: "inner finger", 3: "hand", 4: "object"}


def fit_slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() >= 3 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", default=None, help="write the text table here as well")
    ap.add_argument("--fig", default=os.path.join(HERE, "pixel_sens.png"))
    a = ap.parse_args()
    t0 = time.time()
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    S = Scene()
    m, d = S.m, S.d
    sl = S.side["left"]
    out(f"official camera_wrist_left at {W}x{H}, fovy {m.cam_fovy[sl['cam']]:.0f} deg, f = {S.f:.1f} px (pinhole); "
        f"rows {BAND[0]}..{BAND[1]} = padcal rows 280..320 of 480")
    S.set_theta(0.0)
    Rh, ph = d.xmat[sl["hand"]].reshape(3, 3), d.xpos[sl["hand"]]
    cam_h = Rh.T @ (d.cam_xpos[sl["cam"]] - ph)
    view_h = Rh.T @ (-d.cam_xmat[sl["cam"]].reshape(3, 3)[:, 2])
    right_h = Rh.T @ d.cam_xmat[sl["cam"]].reshape(3, 3)[:, 0]
    out(f"camera in the hand frame: pos {np.round(cam_h * 1000, 1)} mm; view dir {np.round(view_h, 3)}; image +x = {np.round(right_h, 3)} "
        f"(hinges at y +-18 mm, z -68 mm; closing direction = hand y)")

    # ------------------------------------------------------------------------------------------------------------------
    # (a) edges vs theta
    # ------------------------------------------------------------------------------------------------------------------
    out("\n(a) inner-edge x vs hinge angle (fingers only), rows 700..800 of 1200; x at 1920 and at 640 under hyp i / ii")
    TH = np.arange(0, 46, 1.0)
    rows_band = list(range(BAND[0], BAND[1] + 1))
    EL, ER, CL, CR = [], [], [], []
    for th in TH:
        S.set_theta(th)
        cls = S.classes()
        xl, xr = scan_edges(cls > 0, rows_band)
        EL.append(xl); ER.append(xr)
        CL.append(edge_class(cls, xl, rows_band)); CR.append(edge_class(cls, xr, rows_band, right=True))
    EL, ER = np.array(EL), np.array(ER)
    out("   theta  xL1920  xR1920  | hyp i: xL   xR   gap | hyp ii: xL   xR   gap | left edge / right edge belongs to")
    for k, th in enumerate(TH):
        if th % 5 == 0 or th in (16, 18, 19, 20):
            out(f"   {th:5.1f}  {EL[k]:7.1f} {ER[k]:7.1f}  | {to640(EL[k],'i'):6.1f} {to640(ER[k],'i'):6.1f} {to640(ER[k],'i')-to640(EL[k],'i'):5.1f} | "
                f"{to640(EL[k],'ii'):6.1f} {to640(ER[k],'ii'):6.1f} {to640(ER[k],'ii')-to640(EL[k],'ii'):5.1f} | {CL_name(CL[k])} / {CL_name(CR[k])}")
    # mirror check with the right camera
    mir = []
    for th in (10.0, 20.0, 30.0):
        S.set_theta(th, "right")
        cr = S.classes("right")
        xl2, xr2 = scan_edges(cr > 0, rows_band)
        k = int(th)
        mir.append(max(abs((W - xr2) - EL[k]), abs((W - xl2) - ER[k])))
    out(f"   mirror check, right camera at 10/20/30 deg: |x_right_cam mirrored - x_left_cam| max {np.nanmax(mir):.1f} px at 1920")
    S.set_theta(0.0, "right")

    # real slopes and the sim slope over the matching hinge band
    out("\n   real calibration (padcal, still unloaded frames, fit over the bins) vs sim, px per deg at 640; sim slope is per hinge deg,")
    out("   sign: + = edge moves right as the gripper closes. implied ratio n = sim slope / real slope (per motor deg)")
    real_slopes = {}
    for unit, cal in REAL_CAL.items():
        c = np.array(cal)
        real_slopes[unit] = (fit_slope(c[:, 0], c[:, 1]), fit_slope(c[:, 0], c[:, 2]))
    cons = {}
    for unit in ("A", "B"):
        c = np.array(REAL_CAL[unit])
        qlo, qhi = c[:, 0].min(), c[:, 0].max()
        sL, sR = real_slopes[unit]
        out(f"   unit {unit}: real q {qlo:.0f}..{qhi:.0f} deg ({len(c)} bins, {int(c[:,3].sum())} frames): left {sL:+.2f}, right {sR:+.2f} px/motor deg")
        for n in RATIOS:
            band = (TH >= abs(qhi) / n - 1e-9) & (TH <= abs(qlo) / n + 1e-9)
            for hyp in ("i", "ii"):
                xs, xr_ = to640(EL, hyp), to640(ER, hyp)
                gL, gR = -fit_slope(TH[band], xs[band]), -fit_slope(TH[band], xr_[band])
                # positions at the reference motor angle -33
                th_ref = 33.0 / n
                pL, pR = np.interp(th_ref, TH, xs), np.interp(th_ref, TH, xr_)
                rL = c[c[:, 0] == -33][0]
                out(f"      n {n:4.2f} hyp {hyp:2s}: theta {abs(qhi)/n:4.1f}..{abs(qlo)/n:4.1f} -> sim left {gL:+6.2f} right {gR:+6.2f} px/hinge deg; "
                    f"implied n {gL/sL:5.2f} / {gR/sR:5.2f}; at q -33: sim x {pL:7.1f} {pR:7.1f} vs real {rL[1]:.1f} {rL[2]:.1f}")
                cons[(unit, n, hyp)] = dict(slope=(gL, gR), impl=(gL / sL, gR / sR), pos=(pL, pR), real_pos=(rL[1], rL[2]))

    # ------------------------------------------------------------------------------------------------------------------
    # contact geometry per size
    # ------------------------------------------------------------------------------------------------------------------
    out("\n(b) contact geometry and px/mm per object size (box across the jaw; x 20 mm, z 20 mm, centred on the fingertips)")
    S.show_obj(True)
    geo = {}
    for s in SIZES_MM:
        thc = {f: S.theta_contact(s, f) for f in ("inner", "outer")}
        th_c = max(thc.values())
        S.set_size(s)
        S.set_theta(th_c)
        for f in ("inner", "outer"):
            dist, g, ft = S.dist(f)
            # flat pad on a flat box face: mj_geomDistance returns one vertex of the contact patch; use the patch centre
            # across the finger width (box centre x) and keep the returned height
            Rh, ph = d.xmat[sl["hand"]].reshape(3, 3), d.xpos[sl["hand"]]
            q_h = Rh.T @ (ft[:3] - ph)
            pts = []
            for dx in (0.0, -OBJ_HX, OBJ_HX):
                qh = q_h.copy()
                qh[0] = OBJ_X + dx
                pts.append(ph + Rh @ qh)
            p = pts[0]
            jb, ji = S.side["left"]["body"][f], S.side["left"]["joint"][f]
            xb, Rb = d.xpos[jb].copy(), d.xmat[jb].reshape(3, 3).copy()
            pl = Rb.T @ (p - xb)
            anc, ax = d.xanchor[ji].copy(), d.xaxis[ji].copy()
            rv = p - anc
            r_c = np.linalg.norm(rv - (rv @ ax) * ax)
            cdir = Rh[:, 1] * (1.0 if f == "inner" else -1.0)       # closing direction: toward the jaw centre
            hstep = 1e-4
            Jp = []
            for pp in pts:
                uv_, _ = S.project(pp)
                uv2, _ = S.project(pp + hstep * cdir)
                Jp.append((uv2 - uv_) / (hstep * 1000.0))             # px per mm at 1920, translation along closing
            J = Jp[0]
            J_span = (min(abs(j[0]) for j in Jp), max(abs(j[0]) for j in Jp))
            uv, depth = S.project(p)
            # rotation of the finger by a small closing step: contact point moves along the arc
            dth = 1e-3
            S.set_theta(th_c - np.degrees(dth))
            p_rot = d.xpos[jb] + d.xmat[jb].reshape(3, 3) @ pl
            uv3, _ = S.project(p_rot)
            dclose = (p_rot - p) @ cdir * 1000.0                       # mm of closing for dth rad
            S.set_theta(th_c)
            J_arc = (uv3 - uv) / dclose
            mm_per_hdeg = dclose / np.degrees(dth)
            ray = (p - d.cam_xpos[sl["cam"]])
            L = np.linalg.norm(ray)
            ray /= L
            alpha = np.degrees(np.arcsin(abs(cdir @ ray)))
            gid = np.zeros(1, np.int32)
            vis = []
            for tgt in (p - 0.0005 * cdir, p + 0.0005 * cdir):         # just inside the finger / just inside the box
                vv = tgt - d.cam_xpos[sl["cam"]]
                LL = np.linalg.norm(vv)
                hit = mujoco.mj_ray(m, d, d.cam_xpos[sl["cam"]].copy(), vv / LL, np.array([1, 1, 1, 0, 0, 0], np.uint8), True, -1, gid)
                vis.append((LL * 1000, hit * 1000 if hit >= 0 else np.inf, m.geom(int(gid[0])).name if hit >= 0 else "none"))
            ph_c = Rh.T @ (p - ph)
            geo[(s, f)] = dict(th=th_c, thf=thc[f], p=p, ph=ph_c, r=r_c, uv=uv, depth=depth, J=J, J_span=J_span, J_arc=J_arc,
                               mm_per_hdeg=mm_per_hdeg, alpha=alpha, vis=vis, geom=m.geom(g).name, cdir=cdir)
    out("   contact patch centre (hand frame, mm; x = box centre, z as returned by mj_geomDistance) and the touching collision geom:")
    for s in SIZES_MM:
        out("   " + "; ".join(f"{s} mm {f}: {np.round(geo[(s, f)]['ph'] * 1000, 1)} {geo[(s, f)]['geom']}" for f in ("inner", "outer")))
    out("   size  finger  theta_c  r_c[mm]  contact (u,v)@1920   depth  v@480  in-frame(i/ii)  closing mm/hinge deg  alpha[deg]  "
        "px/mm@1920 u-part centre [across the 20 mm patch]  ray: length / first hit (0.5 mm inside the finger; inside the box)")
    for s in SIZES_MM:
        for f in ("inner", "outer"):
            gq = geo[(s, f)]
            u, v = gq["uv"]
            inf_i = 0 <= u < W and 0 <= v < H
            inf_ii = 160 <= u < W - 160 and 0 <= v < H
            (L1, h1, n1), (L2, h2, n2) = gq["vis"]
            out(f"   {s:3d}   {f:6s}  {gq['th']:6.2f}  {gq['r']*1000:6.1f}  ({u:7.1f},{v:7.1f})  {gq['depth']*1000:5.1f}mm  {v*ROW_SCALE:5.1f}  "
                f"{'y' if inf_i else 'n'}/{'y' if inf_ii else 'n'}  {gq['mm_per_hdeg']:6.3f}  {gq['alpha']:5.1f}  "
                f"{abs(gq['J'][0]):6.2f} [{gq['J_span'][0]:5.2f}..{gq['J_span'][1]:5.2f}]   finger {L1:5.1f}/{h1:5.1f} {n1}; box {L2:5.1f}/{h2:5.1f} {n2}")

    # render-based px/mm: shrink the box by 2 Delta, fingers follow
    out(f"\n   render check: box {DELTA_MM} mm smaller per side, fingers at the new contact angle; edge shift per mm of closing")
    out("   size  finger  rows        edge px/mm@1920: render (finger edge)  render (box edge)  projection u-part (translation / arc)")
    rend = {}
    for s in SIZES_MM:
        s2 = s - 2 * DELTA_MM
        th1, th2 = S.theta_c(s), S.theta_c(s2)
        S.set_size(s); S.set_theta(th1); c1 = S.classes()
        S.set_size(s2); S.set_theta(th2); c2 = S.classes()
        for f, side_ix in (("outer", 0), ("inner", 1)):
            gq = geo[(s, f)]
            vc = int(round(gq["uv"][1]))
            for rname, rows in (("contact", list(range(vc - 25, vc + 26))), ("band", rows_band)):
                e1 = scan_edges(c1 > 0, rows)
                e2 = scan_edges(c2 > 0, rows)
                g1 = scan_edges(np.isin(c1, (1, 2, 3)), rows)
                g2 = scan_edges(np.isin(c2, (1, 2, 3)), rows)
                o1, o2 = obj_extent(c1, rows), obj_extent(c2, rows)
                fe = abs(g2[side_ix] - g1[side_ix]) / DELTA_MM
                be = abs(o2[side_ix] - o1[side_ix]) / DELTA_MM
                ecls = edge_class(c1, g1[side_ix], rows, right=bool(side_ix))
                rend[(s, f, rname)] = dict(finger=fe, box=be, edge=g1[side_ix], edge_cls=ecls, obj=o1)
                out(f"   {s:3d}   {f:6s}  {rname:7s} v{rows[0]:4d}..{rows[-1]:4d}  {fe:7.2f} (shift {fe * DELTA_MM:5.1f} px; edge pixel: {CL_name(ecls)})   {be:7.2f}   "
                    f"{abs(gq['J'][0]):7.2f} / {abs(gq['J_arc'][0]):7.2f}")
    S.set_size(20.0)

    # ------------------------------------------------------------------------------------------------------------------
    # consistency with the real footage: hold, cube size
    # ------------------------------------------------------------------------------------------------------------------
    out("\n   jaw gap with a box held (fingers at the box's contact angle), rows 700..800, vs box size")
    grid = np.arange(5, 131, 5.0)
    gap1920, obj1920, thg = [], [], []
    for s in grid:
        th = S.theta_c(s)
        thg.append(th)
        if not np.isfinite(th):
            gap1920.append((np.nan, np.nan)); obj1920.append((np.nan, np.nan)); continue
        S.set_size(s); S.set_theta(th)
        c = S.classes()
        gap1920.append(scan_edges(np.isin(c, (1, 2, 3)), rows_band))
        obj1920.append(obj_extent(c, rows_band))
    gap1920, obj1920, thg = np.array(gap1920), np.array(obj1920), np.array(thg)
    for k, s in enumerate(grid):
        if s in (10, 20, 30, 45, 60, 80, 100, 120):
            gl, gr = gap1920[k]
            out(f"   box {s:5.0f} mm: theta_c {thg[k]:5.2f}; gap @1920 {gr-gl:7.1f}, hyp i {(gr-gl)/3:6.1f}, hyp ii {(gr-gl)*0.4:6.1f} px; "
                f"box visible width @1920 {obj1920[k,1]-obj1920[k,0]:7.1f}")

    def size_for_theta(th_target):
        ok = np.isfinite(thg)
        return float(np.interp(th_target, thg[ok], grid[ok])) if th_target <= np.nanmax(thg) else np.nan

    def size_for_gap(gap640, hyp):
        g = (gap1920[:, 1] - gap1920[:, 0]) * HYP[hyp][0]
        ok = np.isfinite(g)
        gg, ss = g[ok], grid[ok]
        o = np.argsort(gg)
        return float(np.interp(gap640, gg[o], ss[o])) if gg.min() <= gap640 <= gg.max() else np.nan

    out("\n   consistency table (unit A unless noted): sim official camera vs the real footage")
    out("   hyp  n     edge slope L/R vs real (ratio)   x at q-33 L/R (real 267.7/428.9)   jaw gap at hold (real A 117.4 [110,122], B 107.8)   "
        "box for the hold angle: size / visible gap   box size the image implies   opening A -55.7 / B -45.1 within 45 deg hinge")
    table_rows = []
    for hyp in ("i", "ii"):
        for n in RATIOS:
            cA = cons[("A", n, hyp)]
            thA = abs(REAL_HOLD["A"][0]) / n
            thB = abs(REAL_HOLD["B"][0]) / n
            s_kin = size_for_theta(thA)
            gs = (gap1920[:, 1] - gap1920[:, 0]) * HYP[hyp][0]
            gap_at = float(np.interp(s_kin, grid[np.isfinite(gs)], gs[np.isfinite(gs)])) if np.isfinite(s_kin) else np.nan
            # jaw with no object at the hold angle
            jawL, jawR = np.interp(thA, TH, to640(EL, hyp)), np.interp(thA, TH, to640(ER, hyp))
            s_img_A = size_for_gap(REAL_HOLD["A"][1], hyp)
            s_img_B = size_for_gap(REAL_HOLD["B"][1], hyp)
            s_img_100 = size_for_gap(100.0, hyp)
            openA, openB = abs(Q_OPEN["A"]) / n, abs(Q_OPEN["B"]) / n
            rl, rr = real_slopes["A"]
            row = dict(hyp=hyp, n=n, slope=cA["slope"], sratio=(cA["slope"][0] / rl, cA["slope"][1] / rr), pos=cA["pos"],
                       gap_free=jawR - jawL, s_kin=s_kin, gap_kin=gap_at, s_img=(s_img_A, s_img_B, s_img_100), openA=openA, openB=openB)
            table_rows.append(row)
            out(f"   {hyp:3s} {n:4.2f}  {cA['slope'][0]:+6.2f}/{cA['slope'][1]:+6.2f} ({row['sratio'][0]:4.2f}x/{row['sratio'][1]:4.2f}x)   "
                f"{cA['pos'][0]:7.1f}/{cA['pos'][1]:7.1f}   no box {jawR-jawL:6.1f} (theta {thA:4.1f})   "
                f"{s_kin:5.1f} mm / {gap_at:6.1f} px   A {s_img_A:5.1f} mm, B {s_img_B:5.1f} mm, 100 px {s_img_100:5.1f} mm   "
                f"{openA:4.1f} {'ok' if openA <= HINGE_MAX + 0.5 else 'EXCEEDS'} / {openB:4.1f} {'ok' if openB <= HINGE_MAX + 0.5 else 'EXCEEDS'}")

    # ------------------------------------------------------------------------------------------------------------------
    # (c) S and F_cam
    # ------------------------------------------------------------------------------------------------------------------
    out("\n(c) camera sensitivity S = (px/mm)/k and floor F_cam = jitter / S, per edge (one pad); jaw (two edges) is sqrt(2) lower")
    out("   px/mm sources at the contact point, closing direction, x-component:")
    out("     sim1920  = official camera, (b): projection over the contact patch and render at the contact rows, both fingers;  sim640 i/ii = sim1920 / 3 or x 0.4")
    out("     real640  = real edge slope (units A and B, |L|..|R| 2.51..2.63 px/motor deg) x n / (closing mm per hinge deg at the contact, sim kinematics), n 1..1.25")
    out("     real1920 = real640 x 2.5 (hyp ii) .. x 3 (hyp i), same px jitter ASSUMED")
    pxmm = {}
    sAabs = sorted(abs(v) for u in ("A", "B") for v in real_slopes[u])
    for s in SIZES_MM:
        js = [v for f in ("inner", "outer") for v in geo[(s, f)]["J_span"]] + [rend[(s, f, "contact")]["finger"] for f in ("inner", "outer")]
        mmd = [geo[(s, f)]["mm_per_hdeg"] for f in ("inner", "outer")]
        real640 = [sl_ * n / mm for sl_ in sAabs for n in RATIOS for mm in mmd]
        pxmm[s] = dict(sim1920=(min(js), max(js)), sim640=(min(js) / 3, max(js) * 0.4),
                       real640=(min(real640), max(real640)), real1920=(min(real640) * 2.5, max(real640) * 3.0))
        out(f"   size {s:3d} mm: sim1920 {min(js):6.2f}..{max(js):6.2f}  sim640 {min(js)/3:5.2f}..{max(js)*0.4:5.2f}  "
            f"real640 {min(real640):5.2f}..{max(real640):5.2f}  real1920 {min(real640)*2.5:5.2f}..{max(real640)*3:5.2f} px/mm")

    def pxmm_at(size):
        ks = sorted(pxmm)
        return {key: tuple(np.interp(size, ks, [pxmm[k][key][i] for k in ks]) for i in (0, 1)) for key in pxmm[ks[0]]}

    def fcam(sig, k_eff, p):
        # per edge: boundary moves F / k_eff per side; F_cam = sig * k_eff / p
        return sig[0] * k_eff / p[1], sig[1] * k_eff / p[0]

    out("\n   motor floor at the pad (per finger), F = n tau / (2 r), r 60..97 mm, n 1 (x1.25 for n 1.25):")
    for name, t1, t2 in TORQUE_FLOORS:
        out(f"     {name:38s} {t1:.4f}..{t2:.4f} Nm -> {t1/(2*R_LEVER[1]):6.3f}..{t2/(2*R_LEVER[0]):6.3f} N")
    out(f"     specified for D3: unit A {MOTOR_FLOOR_N['A'][0]}..{MOTOR_FLOOR_N['A'][1]} N, unit B {MOTOR_FLOOR_N['B'][0]}..{MOTOR_FLOOR_N['B'][1]} N")
    rc = [geo[(s, f)]["r"] * 1000 for s in SIZES_MM for f in ("inner", "outer")]
    out(f"     sim contact radius r_c for 10..60 mm boxes: {min(rc):.1f}..{max(rc):.1f} mm")

    out("\n   S [px/N] per edge: object = (px/mm)/(2k), pad = (px/mm)/k_pad")
    out("   item                     k[N/mm] size   real640        real1920       sim640         sim1920")
    for name, k, size, src in OBJECTS + [("pad", kp_, 45, "sweep") for kp_ in K_PAD]:
        P = pxmm_at(size)
        keff = k if name == "pad" else 2.0 * k
        out(f"   {name:24s} {k:7.2f} {size:4d}   " + "  ".join(f"{P[key][0]/keff:6.3f}..{P[key][1]/keff:6.3f}"
                                                         for key in ("real640", "real1920", "sim640", "sim1920")))
    out("\n   object compression (boundary moves F/(2k) per side): F_cam per edge [N] (range over px/mm source span and jitter span)")
    hdr = "   object               k[N/mm] size  real640 A-jit   real640 B-jit   real1920 A-jit  real1920 B-jit  sim640 A-jit    sim1920 A-jit   below floor A(3-5N)/B(0.3-0.7N) @640"
    out(hdr)
    summary = []

    def verdict(f, floor):
        if f[1] < floor[0]:
            return "yes"
        if f[0] > floor[1]:
            return "no"
        return "partly"

    for name, k, size, src in OBJECTS:
        P = pxmm_at(size)
        keff = 2.0 * k
        fa = fcam(JITTER["A"], keff, P["real640"])
        fb = fcam(JITTER["B"], keff, P["real640"])
        fa19 = fcam(JITTER["A"], keff, P["real1920"])
        fb19 = fcam(JITTER["B"], keff, P["real1920"])
        fs6 = fcam(JITTER["A"], keff, P["sim640"])
        fs19 = fcam(JITTER["A"], keff, P["sim1920"])
        vA, vB = verdict(fa, MOTOR_FLOOR_N["A"]), verdict(fb, MOTOR_FLOOR_N["B"])
        summary.append((name, k, size, fa, fb, fa19, fb19, fs6, fs19, vA, vB))
        out(f"   {name:20s} {k:6.2f} {size:4d}   {fa[0]:5.2f}..{fa[1]:5.2f}   {fb[0]:5.2f}..{fb[1]:5.2f}   {fa19[0]:5.2f}..{fa19[1]:5.2f}   "
            f"{fb19[0]:5.2f}..{fb19[1]:5.2f}   {fs6[0]:5.2f}..{fs6[1]:5.2f}   {fs19[0]:5.3f}..{fs19[1]:5.3f}   {vA:6s}/{vB}")
    out("   rigid object (wood cube, k >= 100 N/mm): F_cam per edge >= "
        f"{fcam(JITTER['A'], 200.0, pxmm_at(45)['real640'])[0]:.0f} N (A jitter, real640) -> the motor floor is all there is")

    out("\n   pad compression (pad face moves F/k_pad, the finger body does not): F_cam per edge [N]")
    out("   k_pad   real640 A-jit   real640 B-jit   real1920 A-jit   sim1920 A-jit   px at the unit-A hold force 27..44 N (real640)")
    padrows = []
    for kp in K_PAD:
        P = pxmm_at(45)
        fa, fb = fcam(JITTER["A"], kp, P["real640"]), fcam(JITTER["B"], kp, P["real640"])
        fa19, fs19 = fcam(JITTER["A"], kp, P["real1920"]), fcam(JITTER["A"], kp, P["sim1920"])
        hold_px = (27.0 / kp * P["real640"][0], 44.0 / kp * P["real640"][1])
        padrows.append((kp, fa, fb, fa19, fs19, hold_px))
        out(f"   {kp:5.0f}   {fa[0]:5.2f}..{fa[1]:5.2f}   {fb[0]:5.2f}..{fb[1]:5.2f}   {fa19[0]:5.2f}..{fa19[1]:5.2f}   {fs19[0]:5.3f}..{fs19[1]:5.3f}   "
              f"{hold_px[0]:6.2f}..{hold_px[1]:6.2f} px")
    # bound on the real pad stiffness from F5 (edge barely moves from landing to hold)
    P45 = pxmm_at(45)["real640"]
    dF = (F5_DTAU / (2 * R_LEVER[1]), F5_DTAU / (2 * R_LEVER[0]))
    per_pad_mm = (F5_GAP_PX / 2) / np.array([P45[1], P45[0]])
    kmin = (dF[0] / per_pad_mm[1], dF[1] / per_pad_mm[0])
    out(f"   from F5 (unit A, 32 grasps, median): landing -> hold adds {F5_DTAU} Nm = {dF[0]:.1f}..{dF[1]:.1f} N per finger while the gap "
        f"changes {F5_GAP_PX:.2f} px (both edges). If all of it were pad compression: {per_pad_mm[0]*1000:.0f}..{per_pad_mm[1]*1000:.0f} um per pad "
        f"-> k_pad >= {min(kmin):.0f}..{max(kmin):.0f} N/mm (45 mm cube px/mm assumed)")

    # ------------------------------------------------------------------------------------------------------------------
    # (d) figure
    # ------------------------------------------------------------------------------------------------------------------
    s_view = size_for_theta(abs(REAL_HOLD["A"][0]))
    th_view = S.theta_c(s_view)
    S.set_size(s_view); S.set_theta(th_view)
    img = S.render_rgb()
    make_figure(a.fig, TH, EL, ER, real_slopes, geo, rend, pxmm, summary, padrows, img, s_view, th_view)
    out(f"\nfigure: {a.fig}  (view: box {s_view:.1f} mm at theta {th_view:.1f} deg = hold angle at n 1)")
    out(f"runtime {time.time() - t0:.1f} s")
    if a.table:
        with open(a.table, "w") as fh:
            fh.write("\n".join(lines) + "\n")


def CL_name(c):
    return CLS_NAME.get(int(c), "?")


def make_figure(path, TH, EL, ER, real_slopes, geo, rend, pxmm, summary, padrows, img, s_view, th_view):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(2, 2, 1)
    for hyp, ls in (("i", "-"), ("ii", "--")):
        ax.plot(TH, to640(EL, hyp), "C0" + ls, lw=1.6, label=f"sim left edge, hyp {hyp}")
        ax.plot(TH, to640(ER, hyp), "C3" + ls, lw=1.6, label=f"sim right edge, hyp {hyp}")
    for unit, mk in (("A", "o"), ("B", "s")):
        c = np.array(REAL_CAL[unit])
        for n, col in ((1.0, "k"), (1.25, "0.55")):
            ax.plot(np.abs(c[:, 0]) / n, c[:, 1], mk, color=col, ms=4, mfc="none" if unit == "B" else col,
                    label=f"real unit {unit}, theta = |q|/{n:g}")
            ax.plot(np.abs(c[:, 0]) / n, c[:, 2], mk, color=col, ms=4, mfc="none" if unit == "B" else col)
    ax.axhspan(0, 640, color="0.95", zorder=-1)
    ax.set_xlabel("hinge angle theta [deg] (0 = closed)")
    ax.set_ylabel("inner-edge x at 640 px wide (rows 280..320 of 480)")
    ax.set_title("(a) finger inner edge vs hinge angle: official MJCF camera vs real calibration")
    ax.set_ylim(-20, 700)
    ax.axhline(0, color="0.6", lw=0.6); ax.axhline(640, color="0.6", lw=0.6)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 2, 2)
    sz = np.array(SIZES_MM, float)
    for f, col in (("inner", "C3"), ("outer", "C0")):
        J = np.array([abs(geo[(s, f)]["J"][0]) for s in SIZES_MM])
        ax.plot(sz, J / 3, col + "-o", ms=4, label=f"sim projection, {f} finger, hyp i")
        ax.plot(sz, J * 0.4, col + "--o", ms=4, mfc="none", label=f"sim projection, {f} finger, hyp ii")
        R = np.array([rend[(s, f, "contact")]["finger"] for s in SIZES_MM])
        ax.plot(sz, R / 3, col + "x", ms=7, label=f"sim render (finger edge), {f}, hyp i")
    lo = np.array([pxmm[s]["real640"][0] for s in SIZES_MM]); hi = np.array([pxmm[s]["real640"][1] for s in SIZES_MM])
    ax.fill_between(sz, lo, hi, color="k", alpha=0.2, label="real-calibrated (edge slope, n 1..1.25)")
    ax.set_ylim(0, 10)
    ax.set_xlabel("object size across the jaw [mm]")
    ax.set_ylabel("px per mm of closing at the contact (640 px wide)")
    ax.set_title("(b) px/mm at the contact row")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 2, 3)
    ax.axis("off")
    txt = ["(c) F_cam per edge [N] at 640 (real-calibrated px/mm), next to the motor floor",
           f"{'object':19s}{'k N/mm':>7s}{'mm':>4s}  {'jit 0.5-0.7':>12s}  {'jit 1-2':>12s}  {'1920':>12s}  vs A 3-5 / B 0.3-0.7",
           ]
    for (name, k, size, fa, fb, fa19, fb19, fs6, fs19, vA, vB) in summary:
        txt.append(f"{name:19s}{k:7.2f}{size:4d}  {fa[0]:5.2f}-{fa[1]:5.2f}  {fb[0]:5.2f}-{fb[1]:5.2f}  {fa19[0]:5.2f}-{fa19[1]:5.2f}  {vA:>6s} / {vB}")
    txt.append("")
    txt.append(f"{'pad k N/mm':12s}{'jit 0.5-0.7':>14s}{'jit 1-2':>14s}{'1920 (A jit)':>14s}  px at 27-44 N")
    for kp, fa, fb, fa19, fs19, hp in padrows:
        txt.append(f"{kp:10.0f}  {fa[0]:6.2f}-{fa[1]:5.2f}  {fb[0]:6.2f}-{fb[1]:5.2f}  {fa19[0]:6.2f}-{fa19[1]:5.2f}  {hp[0]:5.1f}-{hp[1]:5.1f}")
    txt.append("")
    txt.append("object: F_cam = jitter x 2k / (px/mm); pad: jitter x k_pad / (px/mm)")
    txt.append("1920: same px jitter assumed; px/mm x2.5 (hyp ii) .. x3 (hyp i)")
    ax.text(0.0, 1.0, "\n".join(txt), family="monospace", fontsize=7.6, va="top")

    ax = fig.add_subplot(2, 2, 4)
    from PIL import Image
    im = Image.fromarray(img).crop((160, 0, 1760, 1200)).resize((640, 480))
    ax.imshow(im)
    ax.axhspan(280, 320, color="y", alpha=0.25)
    ax.set_title(f"official camera, hyp ii view, box {s_view:.0f} mm held at theta {th_view:.1f} deg (rows 280..320 shaded)", fontsize=9)
    ax.set_xlim(0, 640); ax.set_ylim(480, 0)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
