"""F9: is the position dependence of the gripper's opening torque on unit A a conservative torque (spring, soft stop, cable
reaction, gravity), a position-dependent friction, or both, and in what proportion? (pre-registered in notes/floor_ja.md F9)

Three steps, three interpreters' worth of dependencies kept apart:
  export   (CPython with pyarrow)  reads each LeRobot v3 dataset root with f8_sessions.load() (the F8 reader, unchanged), adds
           the 7 arm joint positions from the same parquet rows, and writes one plain numeric npz per dataset.
  fk       (Python with mujoco, no pyarrow)  loads the official MJCF (openarm/v2/openarm_bimanual.xml), decides the arm side
           from the arm joint ranges against the MJCF joint limits, sets the 7 arm joints and the finger hinge per frame and
           writes the gravity torque on the two finger hinges, in the closing sense (Nm at the finger hinge, MJCF masses,
           MJCF mounting: world -z), and the same quantity as a 3-vector in the arm base frame (mount-agnostic fit).
           The v2 MJCF fingers are hinges, not slides: the "gravity component along the slide axis" is taken as -z . t,
           t = the unit direction in which a finger's centre of mass moves when the jaw closes, and the generalized gravity
           force is m g (-z . t) |a x r| summed over both fingers (checked against qfrc_bias of mj_forward).
  analyze  (CPython, numpy only)  per upload batch (= session, f8_sessions.hf_dates, cached JSON only) and per 5-deg bin of
           motor angle on a fixed grid with edges at ..., -52, -47, -42, ..., -2 deg:
             steady free motion (|v| 40-100 deg/s, |acc| < 150 deg/s^2, |tau| <= 1.5 Nm, same episode on both neighbours,
             not within 3 deg of the session's range minimum nor within 3 deg of 0): tau_open, tau_close = median |tau| per
             direction, f = (tau_open + tau_close) / 2 (direction-following, friction), s = (tau_open - tau_close) / 2
             (direction-independent, conservative, > 0 pushes toward closed);
             still frames (band.py's strict-stillness rule copied verbatim and checked against band.py's own output on
             100ep-v2): per sign of tau the p90 and p50 of |tau|, friction = half-sum of the p90s, offset = half-difference;
             the 15-40 deg/s band with the same exclusions (flagged: F8 found it contaminated by in-hand motion);
             opening frames within 0.5 s after the last |tau| > 1.5 Nm frame of the episode (release) apart from later ones;
             gravity: at fixed session, bin and direction, |tau| regressed on the MJCF gravity torque (1-D) and on the
             mount-agnostic 3-vector (3-D);
             absolute angle vs distance from the session's open end: the same cells on a grid of (theta - range minimum),
             compared by the residual of a session + bin additive model.
           Unit A is pooled only after the per-session decomposition (median over sessions, with the spread).
           95 % intervals: 2000 resamples of whole episodes within each session (fixed seed), the same resample for every
           quantity of a session so differences have a joint interval. '!' marks a number resting on < 30 frames.

usage:
  python3 -I f9_direction.py export ROOT [ROOT ...] --out NPZDIR
  <python with mujoco> -I f9_direction.py fk NPZDIR --mjcf openarm/v2/openarm_bimanual.xml
  python3 -I f9_direction.py analyze NPZDIR --hf-cache DIR --out TABLE.txt [--boot 2000]
NPZDIR holds <dataset>.npz from export and <dataset>_fk.npz from fk. d2_analyze.py and f8_sessions.py are imported, not
modified."""
import argparse, glob, math, os, sys
import numpy as np

FLOOR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FLOOR)        # -I leaves the script directory off sys.path

MIN_N = 30                       # '!' below this many frames (as F8)
CELL_N = 5                       # a steady cell needs >= 5 frames per direction to give f and s (F8 Table 5 rule)
STILL_N = 5                      # same for a still cell (band.py prints >= 20; shown with '!' below 30)
EDGES = np.arange(-62.0, 3.0, 5.0)   # lower bin edges: -62, -57, ..., -2 (band.py's grid on 100ep-v2 is floor(-51.3) = -52)
REL_EDGES = np.arange(3.0, 68.0, 5.0)  # bins of (theta - session range minimum): [3, 8), [8, 13), ...
UNIT = {"openarm-pick-place-50ep": "A", "openarm-pick-place-practice": "A", "openarm-pick-place-100ep-v2": "A",
        "openarm-pick-place-blue-20ep": "A", "openarm-pick-place-orange-20ep": "A", "openarm-pick-place-yellow-20ep": "A",
        "openarm-cube-organization-v2": "A", "openarm-cube-organization-full-clear-v2": "A",
        "openarm-pick-place-newarm": "B", "openarm-pick-place-operator-50ep": "B"}   # F8 Table 1 level marker
RELEASE_S = 0.5


# =========================================================================================================== export
def read_arm(root):
    """observation.state of every readable data parquet, in the order d2_analyze.load_dataset reads them."""
    import pyarrow as pa, pyarrow.parquet as pq
    tabs = []
    for f in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet"))):
        try: tabs.append(pq.read_table(f, columns=["observation.state", "episode_index", "frame_index", "timestamp"]))
        except Exception as e: print(f"  skip unreadable {os.path.basename(f)} ({type(e).__name__})")
    tab = pa.concat_tables(tabs)
    col = tab.column("observation.state").combine_chunks()
    S = col.flatten().to_numpy(zero_copy_only=False).astype(float).reshape(len(col), -1)
    return (S, np.asarray(tab.column("episode_index").to_numpy()), np.asarray(tab.column("frame_index").to_numpy()),
            np.asarray(tab.column("timestamp").to_numpy(), float))


def cmd_export(a):
    import f8_sessions as f8
    os.makedirs(a.out, exist_ok=True)
    for root in a.roots:
        base = os.path.basename(os.path.normpath(root)); name = base.split("__", 1)[1]
        V, err, log = f8.load(root)
        if V is None: print(f"{name}: unreadable ({err})"); continue
        S, ep, fi, ts = read_arm(root)
        sg = -1.0 if V["flipped"] else 1.0
        assert np.array_equal(ep, V["ep"]) and np.array_equal(sg * S[:, 21], V["pos"]) and np.array_equal(sg * S[:, 23], V["tau"])
        arm = S[:, [0, 3, 6, 9, 12, 15, 18]]
        np.savez(os.path.join(a.out, f"{name}.npz"), name=np.array(name), ep=V["ep"].astype(np.int64), frame_index=fi.astype(np.int64),
                 timestamp=ts, arm=arm, pos=V["pos"], vel=V["vel"], tau=V["tau"], act=V["act"], fps=np.array(V["fps"]),
                 flipped=np.array(V["flipped"]))
        print(f"{name}: {len(ep)} frames, {len(np.unique(ep))} episodes, fps {V['fps']:g}, flipped {V['flipped']}; arm joint "
              "p0.5/p99.5 deg: " + " ".join(f"j{k + 1} {np.percentile(arm[:, k], 0.5):.1f}/{np.percentile(arm[:, k], 99.5):.1f}" for k in range(7)))


# =============================================================================================================== fk
def cmd_fk(a):
    import mujoco
    m = mujoco.MjModel.from_xml_path(a.mjcf); d = mujoco.MjData(m)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    gvec = np.array(m.opt.gravity, float)
    for f in sorted(glob.glob(os.path.join(a.npzdir, "*.npz"))):
        if f.endswith("_fk.npz"): continue
        z = np.load(f, allow_pickle=False); name = str(z["name"]); arm = z["arm"]; pos = z["pos"]
        # side: the MJCF joint limits of the two arms differ in sign for joints 1 and 2; count frames outside each side's limits
        viol = {}
        for side in ("left", "right"):
            v = np.zeros(len(arm), bool); per = []
            for k in range(7):
                lo, hi = np.degrees(m.jnt_range[jid(f"openarm_{side}_joint{k + 1}")])
                out = (arm[:, k] < lo - 2) | (arm[:, k] > hi + 2); v |= out; per.append(int(out.sum()))
            viol[side] = (int(v.sum()), per)
        side = a.side if a.side != "auto" else min(viol, key=lambda s: viol[s][0])
        ja = [jid(f"openarm_{side}_joint{k + 1}") for k in range(7)]
        jf = [jid(f"openarm_{side}_finger_joint1"), jid(f"openarm_{side}_finger_joint2")]
        qa = [m.jnt_qposadr[j] for j in ja]; qf = [m.jnt_qposadr[j] for j in jf]; df = [m.jnt_dofadr[j] for j in jf]
        bf = [m.jnt_bodyid[j] for j in jf]; mass = np.array([m.body_mass[b] for b in bf])
        frng = m.jnt_range[jf[0]]
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_base_link")
        # motor angle (deg, 0 closed, negative open) -> finger hinge (rad): left opens toward +, right toward - (MJCF ranges);
        # ratio 1 (F2-6: unknown 1 or 1.25), clipped to the MJCF range
        qfin = np.clip(np.radians(-pos if side == "left" else pos), frng[0], frng[1])
        open_sign = 1.0 if side == "left" else -1.0          # +q of the finger joints opens (left) or closes (right)
        n = len(pos); xi = np.zeros((n, 2, 3)); xa = np.zeros((n, 2, 3)); ax = np.zeros((n, 2, 3))
        d.qpos[:] = m.qpos0
        for i in range(n):
            d.qpos[qa] = np.radians(arm[i]); d.qpos[qf] = qfin[i]
            mujoco.mj_kinematics(m, d)
            xi[i] = d.xipos[bf]; xa[i] = d.xanchor[jf]; ax[i] = d.xaxis[jf]
        R = d.xmat[base].reshape(3, 3).copy()                 # base link is fixed to the world: constant
        r = xi - xa; c = np.cross(ax, r)                      # a x r: COM velocity direction per unit +q, length |r_perp|
        Gj = (c @ gvec) * mass                                # generalized gravity force on each finger joint (+q sense)
        g_close = -open_sign * Gj.sum(1)                      # Nm at the finger hinge, toward closed
        U_close = -open_sign * np.einsum("nkj,k->nj", c, mass)  # kg m, world frame; g_close = gvec . U_close
        U_close_b = U_close @ R                               # same in the base frame: g_close = (R^T gvec) . U_close_b
        tdir = -open_sign * c / np.maximum(np.linalg.norm(c, axis=2, keepdims=True), 1e-12)  # unit closing direction per finger
        g_axis = -tdir[:, :, 2]                               # -z . t per finger (the pre-registered "component along the axis")
        # check against MuJoCo's own bias force on a sample of frames (qvel = 0: qfrc_bias is gravity only)
        rs = np.random.default_rng(0).choice(n, size=min(300, n), replace=False); err = 0.0
        for i in rs:
            d.qpos[:] = m.qpos0; d.qpos[qa] = np.radians(arm[i]); d.qpos[qf] = qfin[i]; d.qvel[:] = 0
            mujoco.mj_forward(m, d); err = max(err, float(np.max(np.abs(-d.qfrc_bias[df] - Gj[i]))))
        assert err < 1e-9, (name, err)
        np.savez(os.path.join(a.npzdir, f"{name}_fk.npz"), g_close=g_close, U_close_b=U_close_b, g_axis=g_axis,
                 gvec_b=R.T @ gvec, side=np.array(side), mass=mass,
                 viol_left=np.array(viol["left"][1]), viol_right=np.array(viol["right"][1]))
        print(f"{name}: side {side} (frames outside the MJCF joint limits +-2 deg: left {viol['left'][0]} {viol['left'][1]}, "
              f"right {viol['right'][0]} {viol['right'][1]}); finger masses {mass.round(4).tolist()} kg; g_close p1/p50/p99 "
              f"{np.percentile(g_close, 1):+.4f}/{np.median(g_close):+.4f}/{np.percentile(g_close, 99):+.4f} Nm; "
              f"g_axis(inner) p1/p99 {np.percentile(g_axis[:, 0], 1):+.2f}/{np.percentile(g_axis[:, 0], 99):+.2f}; "
              f"qfrc_bias check max |diff| {err:.1e} on {len(rs)} frames; gravity in base frame {np.round(R.T @ gvec, 3).tolist()}")


# ========================================================================================================== analyze
# band.py's printed output on gtgando/openarm-pick-place-100ep-v2 (whole dataset). The copied still mask and the F2-4 steady
# rule below must give the same n exactly and the same values to the printed 3 decimals.
#   still: bin -> (n_open, p50_open, p90_open, n_close[, p50_close, p90_close])   steady: bin -> (n_close, med, n_open, med)
BAND_STILL_REF = {-52: (509, 1.082, 1.643, 0), -47: (3433, 1.038, 1.432, 8), -42: (4352, 0.886, 1.179, 84, 0.286, 0.510),
                  -37: (3137, 0.681, 0.852, 55, 0.310, 0.627), -32: (528, 0.593, 0.667, 5), -27: (45, 0.549, 0.573, 3)}
BAND_STEADY_REF = {-52: (3, None, 21, 0.969), -42: (37, 0.637, 53, 0.877), -32: (66, 0.679, 55, 0.632), -22: (0, None, 71, 0.437)}


def band_still(pos, vel, tau, ep):
    """band.py's strict-still rule, vectorized: still = |v| < 1 deg/s and the position within 0.03 deg over 7 frames of one
    episode; mid = 3 deg inside the p0.5 / p99.5 range of the frames given; pull-open = still & mid & tau < 0; push-closed
    (free) = still & mid & tau > 0 & more than 6 deg wider than the episode's hold (median position of still tau > 1.5)."""
    n = len(pos); W = 3
    still = np.abs(vel) < 1.0
    win = np.lib.stride_tricks.sliding_window_view(pos, 2 * W + 1)
    rng = np.zeros(n, bool); rng[W:n - W] = ((win.max(1) - win.min(1)) < 0.03) & (ep[:-2 * W] == ep[2 * W:])
    still &= rng
    lo, hi = np.percentile(pos, 0.5), np.percentile(pos, 99.5)
    mid = (pos > lo + 3) & (pos < hi - 3)
    hold = {}
    for e in np.unique(ep):
        m = (ep == e) & still & (tau > 1.5)
        if m.sum() > 5: hold[e] = np.median(pos[m])
    hold_at = np.array([hold.get(e, np.nan) for e in ep])
    with np.errstate(invalid="ignore"): wider = pos < hold_at - 6
    return still & mid & (tau < 0), still & mid & (tau > 0) & wider, wider, mid, lo, hi


def check_band(V, L):
    """Assert the copied still mask and band.py's steady rule (F2-4) reproduce band.py's output on 100ep-v2."""
    pos, vel, tau, ep = V["pos"], V["vel"], V["tau"], V["ep"]
    op, cl, wider, mid, lo, hi = band_still(pos, vel, tau, ep)
    seen = set()
    L.append(f"  band.py still rule, whole 100ep-v2 (grid from floor(p0.5) = {np.floor(lo):g}): bin n_open p50 p90 | n_close p50 p90 "
             "(asserted equal to band.py's printed output)")
    for b0 in np.arange(np.floor(lo), np.ceil(hi), 5):
        mo = op & (pos >= b0) & (pos < b0 + 5); mc = cl & (pos >= b0) & (pos < b0 + 5)
        if mo.sum() >= 20 or mc.sum() >= 20:
            ref = BAND_STILL_REF[int(b0)]; seen.add(int(b0))
            assert int(mo.sum()) == ref[0] and int(mc.sum()) == ref[3], (b0, mo.sum(), mc.sum())
            po = np.percentile(-tau[mo], [50, 90]); assert abs(po[0] - ref[1]) < 5e-4 and abs(po[1] - ref[2]) < 5e-4, (b0, po)
            pc = (np.nan, np.nan)
            if mc.sum() >= 20:
                pc = np.percentile(tau[mc], [50, 90]); assert abs(pc[0] - ref[4]) < 5e-4 and abs(pc[1] - ref[5]) < 5e-4, (b0, pc)
            L.append(f"    [{b0:5.0f},{b0 + 5:4.0f})  {mo.sum():5d} {po[0]:.3f} {po[1]:.3f} | {mc.sum():4d} {fm(pc[0])} {fm(pc[1])}")
    assert seen == set(BAND_STILL_REF), seen
    acc = np.r_[0, np.diff(vel)] * 30.0
    steady = (np.abs(vel) > 15) & (np.abs(acc) < 150) & mid
    L.append("  band.py steady rule = F2-4 (|v| > 15, |a| < 150 with a backward difference, closing also > 6 deg wider than the "
             "hold, 10-deg bins from floor(p0.5)): bin n_close tau_close | n_open tau_open (asserted)")
    seen = set()
    for b0 in np.arange(np.floor(lo), np.ceil(hi), 10):
        mc = steady & (vel > 0) & wider & (pos >= b0) & (pos < b0 + 10); mo = steady & (vel < 0) & (pos >= b0) & (pos < b0 + 10)
        if mc.sum() >= 10 or mo.sum() >= 10:
            ref = BAND_STEADY_REF[int(b0)]; seen.add(int(b0))
            assert int(mc.sum()) == ref[0] and int(mo.sum()) == ref[2], (b0, mc.sum(), mo.sum())
            vc = float(np.median(tau[mc])) if mc.sum() >= 10 else np.nan; vo = float(np.median(-tau[mo])) if mo.sum() >= 10 else np.nan
            if ref[1] is not None: assert abs(vc - ref[1]) < 5e-4, (b0, vc)
            if ref[3] is not None: assert abs(vo - ref[3]) < 5e-4, (b0, vo)
            L.append(f"    [{b0:5.0f},{b0 + 10:4.0f})  {mc.sum():4d} {fm(vc, 3, mc.sum())} | {mo.sum():4d} {fm(vo, 3, mo.sum())}")
    assert seen == set(BAND_STEADY_REF), seen


def fm(x, d=3, n=None):
    if x is None or not np.isfinite(x): return "-"
    return f"{x:.{d}f}" + ("!" if n is not None and n < MIN_N else "")


def fci(c, d=3):
    return f"[{fm(c[0], d)},{fm(c[1], d)}]"


def bin_index(p, edges):
    k = np.floor((p - edges[0]) / (edges[1] - edges[0])).astype(int)
    k[(k < 0) | (k >= len(edges))] = -1
    return k


def blab(edges, b):
    return f"[{edges[b]:g},{edges[b] + edges[1] - edges[0]:g})"


def since_contact(tau, ep, fps):
    """Seconds since the last frame of the same episode with |tau| > 1.5 Nm (inf if none yet)."""
    n = len(tau); last = np.where(np.abs(tau) > 1.5, np.arange(n), -1)
    out = np.full(n, np.inf)
    starts = np.r_[0, np.flatnonzero(ep[1:] != ep[:-1]) + 1, n]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        lk = np.maximum.accumulate(last[s0:s1]); ok = lk >= s0
        out[s0:s1][ok] = (np.arange(s0, s1)[ok] - lk[ok]) / fps
    return out


def until_contact(tau, ep, fps):
    """Seconds until the next frame of the same episode with |tau| > 1.5 Nm (inf if none follows)."""
    n = len(tau); nxt = np.where(np.abs(tau) > 1.5, np.arange(n), n)
    out = np.full(n, np.inf)
    starts = np.r_[0, np.flatnonzero(ep[1:] != ep[:-1]) + 1, n]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        nk = np.minimum.accumulate(nxt[s0:s1][::-1])[::-1]; ok = nk < s1
        out[s0:s1][ok] = (nk[ok] - np.arange(s0, s1)[ok]) / fps
    return out


SETS = ("o40", "c40", "o15", "c15", "sto", "stc", "opost", "olate", "cfar", "cnpre")


def analyse_session(V, z, sel, B, seed, open_rule="min", edges=EDGES):
    """Every per-bin quantity of one session (frames sel), as a point estimate and as B episode-resample draws."""
    pos, vel, tau, ep, fps = V["pos"], V["vel"], V["tau"], V["ep"], V["fps"]
    n = len(pos); ps = pos[sel]
    omin = float(ps.min()) if open_rule == "min" else float(np.percentile(ps, 0.5))
    acc = np.gradient(vel) * fps
    same = np.r_[True, ep[1:] == ep[:-1]] & np.r_[ep[:-1] == ep[1:], True]
    base = sel & same & (np.abs(acc) < 150) & (np.abs(tau) <= 1.5) & (pos > omin + 3) & (pos < -3)
    spd = np.abs(vel)
    m40 = base & (spd >= 40) & (spd < 100); m15 = base & (spd >= 15) & (spd < 40)
    idx = np.flatnonzero(sel)
    op, cl, wid, *_ = band_still(pos[idx], vel[idx], tau[idx], ep[idx])
    sto = np.zeros(n, bool); sto[idx[op]] = True; stc = np.zeros(n, bool); stc[idx[cl]] = True
    wider = np.zeros(n, bool); wider[idx[wid]] = True       # > 6 deg wider than the episode's hold (band.py), False if no hold
    post = since_contact(tau, ep, fps) <= RELEASE_S
    pre = until_contact(tau, ep, fps) <= RELEASE_S
    masks = {"o40": m40 & (vel < 0), "c40": m40 & (vel > 0), "o15": m15 & (vel < 0), "c15": m15 & (vel > 0), "sto": sto, "stc": stc,
             "opost": m40 & (vel < 0) & post, "olate": m40 & (vel < 0) & ~post,
             "cfar": m40 & (vel > 0) & wider, "cnpre": m40 & (vel > 0) & ~pre}
    ue = np.unique(ep[sel]); E = len(ue); nb, nr = len(edges), len(REL_EDGES)
    F = {}
    for k, m in masks.items():
        i = np.flatnonzero(m)
        S = {"i": i, "x": np.abs(tau[i]), "e": np.searchsorted(ue, ep[i]), "ba": bin_index(pos[i], edges), "br": bin_index(pos[i] - omin, REL_EDGES)}
        for fld, nbin in (("ba", nb), ("br", nr)):
            S["sl_" + fld] = [(b, np.flatnonzero(S[fld] == b)) for b in range(nbin) if np.any(S[fld] == b)]
            S["n_" + fld] = np.bincount(S[fld][S[fld] >= 0], minlength=nbin)
            S["eps_" + fld] = np.array([len(np.unique(S["e"][S[fld] == b])) for b in range(nbin)])
        F[k] = S
    # within-cell regressions on the steady 40-100 frames of both directions with a bin; cell = (bin, direction)
    have_g = z.get("g_close") is not None
    go, gc_ = F["o40"], F["c40"]
    gi = np.r_[go["i"][go["ba"] >= 0], gc_["i"][gc_["ba"] >= 0]]
    gs = np.r_[np.ones(np.sum(go["ba"] >= 0)), -np.ones(np.sum(gc_["ba"] >= 0))]
    gb = np.r_[go["ba"][go["ba"] >= 0], gc_["ba"][gc_["ba"] >= 0]]
    gcell = gb * 2 + (gs < 0); ncell = 2 * nb
    gx = np.abs(tau[gi]); ge = np.searchsorted(ue, ep[gi])
    gv = np.abs(vel[gi]); ga = acc[gi] * np.sign(vel[gi])          # speed, and acceleration along the motion (> 0 = speeding up)
    if have_g: gg = z["g_close"][gi]; gU = z["U_close_b"][gi]

    def cell(S, w, fld, q, nbin):
        out = np.full(nbin, np.nan); wt = w[S["e"]]
        for b, sl in S["sl_" + fld]:
            xs = np.repeat(S["x"][sl], wt[sl])
            if len(xs): out[b] = np.percentile(xs, q)
        return out

    def regs(w):
        """Sums for the within-cell least-squares fits (cell means removed, weights = episode resample counts): gravity
        (1-D on g_close, 3-D on U, regressor x +1 opening / -1 closing) and dynamics (2-D on speed and along-motion acc)."""
        wt = w[ge].astype(float); sw = np.bincount(gcell, wt, minlength=ncell); ok = sw > 0
        mean = lambda v: np.where(ok, np.bincount(gcell, wt * v, minlength=ncell) / np.where(ok, sw, 1), 0.0)[gcell]
        dx = gx - mean(gx)
        D = np.stack([gv - mean(gv), ga - mean(ga)], 1)
        R = {"d_xtx": (D * wt[:, None]).T @ D, "d_xty": (D * wt[:, None]).T @ dx}
        if have_g:
            dg = gs * (gg - mean(gg)); dU = gs[:, None] * (gU - np.stack([mean(gU[:, j]) for j in range(3)], 1))
            R.update(g_sxy=float(np.sum(wt * dg * dx)), g_sxx=float(np.sum(wt * dg * dg)), g_xtx=(dU * wt[:, None]).T @ dU,
                     g_xty=(dU * wt[:, None]).T @ dx)
        return R

    def stats(w):
        R = {k: cell(F[k], w, "ba", 50, nb) for k in ("o40", "c40", "o15", "c15", "opost", "olate", "cfar", "cnpre")}
        R["p90o"], R["p90c"] = cell(F["sto"], w, "ba", 90, nb), cell(F["stc"], w, "ba", 90, nb)
        R["p50o"], R["p50c"] = cell(F["sto"], w, "ba", 50, nb), cell(F["stc"], w, "ba", 50, nb)
        R["o40r"], R["c40r"] = cell(F["o40"], w, "br", 50, nr), cell(F["c40"], w, "br", 50, nr)
        R["p50or"] = cell(F["sto"], w, "br", 50, nr)
        R["f40"], R["s40"] = (R["o40"] + R["c40"]) / 2, (R["o40"] - R["c40"]) / 2
        R["f15"], R["s15"] = (R["o15"] + R["c15"]) / 2, (R["o15"] - R["c15"]) / 2
        R["f40r"], R["s40r"] = (R["o40r"] + R["c40r"]) / 2, (R["o40r"] - R["c40r"]) / 2
        R["Fst"], R["off"] = (R["p90o"] + R["p90c"]) / 2, (R["p90o"] - R["p90c"]) / 2
        R["off50"] = (R["p50o"] - R["p50c"]) / 2
        R["dss"] = R["off"] - R["s40"]
        R["slate"] = (R["olate"] - R["c40"]) / 2
        R["sfar"], R["snpre"] = (R["o40"] - R["cfar"]) / 2, (R["o40"] - R["cnpre"]) / 2
        R.update(regs(w))
        return R

    pt = stats(np.ones(E, int))
    rs = np.random.default_rng(seed); draws = []
    for _ in range(B):
        draws.append(stats(np.bincount(rs.integers(0, E, E), minlength=E)))
    bt = {k: np.array([d[k] for d in draws]) for k in pt} if B else {}
    out = {"pt": pt, "bt": bt, "F": F, "omin": omin, "E": E, "frames": int(sel.sum()), "range": (float(ps.min()), float(ps.max()))}
    out["n"] = {k: F[k]["n_ba"] for k in SETS}; out["nr"] = {k: F[k]["n_br"] for k in SETS}
    out["eps"] = {k: F[k]["eps_ba"] for k in SETS}
    out["post_share"] = float(len(F["opost"]["i"]) / max(1, len(F["o40"]["i"])))
    out["pre_share"] = 1.0 - float(len(F["cnpre"]["i"]) / max(1, len(F["c40"]["i"])))
    out["far_share"] = float(len(F["cfar"]["i"]) / max(1, len(F["c40"]["i"])))
    out["g_n"] = len(gi)
    for k, key in (("o40", "o"), ("c40", "c")):        # per bin median speed and along-motion acceleration (point only)
        mv, ma = np.full(nb, np.nan), np.full(nb, np.nan)
        for b, sl in F[k]["sl_ba"]:
            ii = F[k]["i"][sl]; mv[b] = np.median(np.abs(vel[ii])); ma[b] = np.median(acc[ii] * np.sign(vel[ii]))
        out["v" + key], out["a" + key] = mv, ma
    if have_g:
        cm = np.bincount(gcell, gg, minlength=ncell) / np.maximum(np.bincount(gcell, minlength=ncell), 1)
        dg = gs * (gg - cm[gcell])
        out["g_spread"] = float(np.percentile(dg, 95) - np.percentile(dg, 5)) if len(dg) else np.nan
        sgm = np.full(nb, np.nan)
        for b in np.unique(gb): sgm[b] = float(np.mean(gg[gb == b]))
        out["s_grav"] = sgm
        ub = 9.81 * np.linalg.norm(z["U_close_b"][sel], axis=1)   # |g_close| <= 9.81 |U| for any direction of gravity
        out["g_bound"] = (float(np.median(ub)), float(np.percentile(ub, 99)), float(ub.max()))
    return out


def valid(S, kind, b=None):
    n = S["n"]
    v = {"40": (n["o40"] >= CELL_N) & (n["c40"] >= CELL_N), "15": (n["o15"] >= CELL_N) & (n["c15"] >= CELL_N),
         "st": (n["sto"] >= STILL_N) & (n["stc"] >= STILL_N), "late": (n["olate"] >= CELL_N) & (n["c40"] >= CELL_N),
         "40r": (S["nr"]["o40"] >= CELL_N) & (S["nr"]["c40"] >= CELL_N), "po": n["sto"] >= 15, "por": S["nr"]["sto"] >= 15,
         "far": (n["o40"] >= CELL_N) & (n["cfar"] >= CELL_N), "npre": (n["o40"] >= CELL_N) & (n["cnpre"] >= CELL_N),
         "dss": (n["sto"] >= STILL_N) & (n["stc"] >= STILL_N) & (n["o40"] >= CELL_N) & (n["c40"] >= CELL_N)}[kind]
    return v if b is None else bool(v[b])


THIN = {"40": ("o40", "c40"), "15": ("o15", "c15"), "st": ("sto", "stc"), "late": ("olate", "c40"), "dss": ("sto", "stc", "o40", "c40"),
        "far": ("o40", "cfar"), "npre": ("o40", "cnpre")}


def pooled(Ss, key, kind, B):
    """Per bin: median over the sessions whose cell is valid, its 95 % interval (per draw the median of the sessions' draws:
    episodes resampled within every session), the spread (min, max) over sessions, k sessions, frames in the thinner direction."""
    nb = len(Ss[0]["pt"][key]); rows = []
    for b in range(nb):
        use = [S for S in Ss if valid(S, kind)[b]]
        if not use: rows.append(None); continue
        v = np.array([S["pt"][key][b] for S in use])
        ci = (np.nan, np.nan)
        if B:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                d = np.nanmedian(np.stack([S["bt"][key][:, b] for S in use]), axis=0)
            ci = tuple(np.nanpercentile(d, [2.5, 97.5]))
        keys = THIN.get(kind)
        thin = min(sum(int(S["n"][q][b]) for S in use) for q in keys) if keys else None
        rows.append({"k": len(use), "med": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "ci": ci, "thin": thin})
    return rows


def ls_rms(vals, sess, bins, two_way):
    """RMS residual (dof-corrected) of an additive model: bin effect (+ session effect if two_way) fitted by least squares."""
    ok = np.isfinite(vals)
    vals, sess, bins = vals[ok], sess[ok], bins[ok]
    if len(vals) < 4: return np.nan
    cols = [(bins == b).astype(float) for b in np.unique(bins)]
    if two_way: cols += [(sess == s).astype(float) for s in np.unique(sess)[1:]]
    X = np.stack(cols, 1); c, *_ = np.linalg.lstsq(X, vals, rcond=None)
    dof = len(vals) - np.linalg.matrix_rank(X)
    return float(np.sqrt(np.sum((vals - X @ c) ** 2) / dof)) if dof > 0 else np.nan


def abs_vs_rel(Ss, key_abs, key_rel, kind_abs, kind_rel, two_way, B):
    """Residual RMS when the sessions' cells are lined up by absolute angle vs by distance from the session's open end."""
    def cells(key, kind, draw):
        v, s, b = [], [], []
        for j, S in enumerate(Ss):
            ok = valid(S, kind)
            for bb in np.flatnonzero(ok):
                v.append(S["pt"][key][bb] if draw is None else S["bt"][key][draw, bb]); s.append(j); b.append(bb)
        return np.array(v, float), np.array(s), np.array(b)
    pa = ls_rms(*cells(key_abs, kind_abs, None), two_way); pr = ls_rms(*cells(key_rel, kind_rel, None), two_way)
    na = len(cells(key_abs, kind_abs, None)[0]); nr = len(cells(key_rel, kind_rel, None)[0])
    d = np.array([ls_rms(*cells(key_rel, kind_rel, i), two_way) - ls_rms(*cells(key_abs, kind_abs, i), two_way) for i in range(B)]) if B else np.array([np.nan])
    return pa, pr, na, nr, tuple(np.nanpercentile(d, [2.5, 97.5])) if B else (np.nan, np.nan)


def slopes(Ss, key, kind, edges, B):
    """Per-session least-squares slope of a cell quantity against the bin centre (valid bins only), as Nm per 10 deg toward open;
    median over sessions with an interval from the sessions' episode resamples."""
    w = edges[1] - edges[0]; per, pts, drs = [], [], []
    for S in Ss:
        vb = np.flatnonzero(valid(S, kind))
        if len(vb) < 2 or (vb.max() - vb.min()) * w < 10: continue
        x = edges[vb] + w / 2
        def fit(y):
            ok = np.isfinite(y)
            if ok.sum() < 2 or np.ptp(x[ok]) < 10: return np.nan
            return -10.0 * np.polyfit(x[ok], y[ok], 1)[0]
        pts.append(fit(S["pt"][key][vb])); per.append((S.get("label", "?"), pts[-1]))
        if B: drs.append([fit(S["bt"][key][i, vb]) for i in range(B)])
    if not pts: return None
    dr = np.nanmedian(np.array(drs), axis=0) if B else np.full(1, np.nan)
    ci = tuple(np.nanpercentile(dr, [2.5, 97.5])) if B else (np.nan, np.nan)
    return {"med": float(np.median(pts)), "ci": ci, "min": float(min(pts)), "max": float(max(pts)), "k": len(pts), "per": per, "dr": dr}


def short(name):
    return name.replace("openarm-pick-place-", "").replace("openarm-", "")


def cmd_analyze(a):
    import warnings
    import f8_sessions as f8
    warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN medians of empty resampled cells are expected and shown as '-'
    L = []
    data = []
    for f in sorted(glob.glob(os.path.join(a.npzdir, "*.npz"))):
        if f.endswith("_fk.npz"): continue
        z = dict(np.load(f, allow_pickle=False)); name = str(z["name"])
        fk = os.path.join(a.npzdir, f"{name}_fk.npz")
        if os.path.exists(fk): z.update(dict(np.load(fk, allow_pickle=False)))
        else: z["g_close"] = None
        V = dict(pos=z["pos"], vel=z["vel"], tau=z["tau"], act=z["act"], ep=z["ep"], grp=z["ep"], fps=float(z["fps"]), flipped=bool(z["flipped"]))
        data.append((name, V, z))
    order = list(UNIT)
    data.sort(key=lambda t: order.index(t[0]) if t[0] in order else 99)
    L.append("F9 table: direction-following (friction) and direction-independent (conservative) parts of the gripper torque, per "
             "session (upload batch) and 5-deg bin of motor angle (0 = closed, negative = open). Nm. '!' = rests on < 30 frames "
             "(for f, s: the thinner direction). CI = 95 % interval, whole episodes resampled within the session, "
             f"{a.boot} draws, seed 8 + session number. '-' = empty or below the cell minimum (steady >= {CELL_N} frames per "
             f"direction, still >= {STILL_N} per sign).")
    L.append("Steady free motion: |v| 40-100 deg/s (15-40 flagged separately), |acc| < 150 deg/s^2 (np.gradient), |tau| <= 1.5, both "
             "neighbours in the same episode, theta > session range minimum + 3 and theta < -3. tau_open / tau_close = median "
             "|tau| moving toward open / closed; f = (tau_open + tau_close) / 2; s = (tau_open - tau_close) / 2, > 0 = pushes "
             "toward closed. Still: band.py rule on the session's frames; p90 / p50 of |tau| per sign; F = half-sum of the "
             "p90s, off = half-difference (> 0 = pushes toward closed), off50 = the same with p50s.")
    L.append(f"Bins: fixed grid, lower edges {EDGES[0]:g}, {EDGES[1]:g}, ..., {EDGES[-1]:g} (band.py's grid on 100ep-v2). "
             f"Relative bins: theta - session range minimum, lower edges {REL_EDGES[0]:g}, {REL_EDGES[1]:g}, ...")
    # ---------------------------------------------------------------------------------------------------- sessions
    sess = []
    for name, V, z in data:
        h = f8.hf_dates("gtgando/" + name, a.hf_cache, False); bt = h.get("batches", [])
        ue = np.unique(V["ep"])
        if not bt: parts = [("(whole)", int(ue.min()), int(ue.max()), h.get("lastModified"))]
        else: parts = [(f"b{i + 1}" if len(bt) > 1 else "", b["ep0"], b["ep1"], b["date"]) for i, b in enumerate(bt)]
        covered = np.zeros(len(V["ep"]), bool)
        for lab, e0, e1, date in parts:
            sel = (V["ep"] >= e0) & (V["ep"] <= e1); covered |= sel
            sess.append({"name": name, "label": (short(name) + (" " + lab if lab else "")).strip(), "unit": UNIT.get(name, "?"),
                         "sel": sel, "date": (date or "?")[:16].replace("T", " "), "V": V, "z": z, "epr": (e0, e1)})
        assert covered.all(), (name, "episodes outside the upload batches")
    L.append("\nSessions (upload batches, f8_sessions.hf_dates on the cached Hugging Face JSON; a dataset without info.json at any "
             "revision is one session '(whole)', dated by lastModified). Unit from F8 Table 1 (level marker).")
    for j, S in enumerate(sess):
        R = analyse_session(S["V"], S["z"], S["sel"], a.boot, 8 + j); S.update(R)
        side = str(S["z"]["side"]) if S["z"].get("g_close") is not None else "-"
        L.append(f"  {j:2d} {S['label']:34s} unit {S['unit']}  uploaded {S['date']}  ep {S['epr'][0]}-{S['epr'][1]} ({S['E']} episodes, "
                 f"{S['frames']} frames)  range {S['range'][0]:.1f}..{S['range'][1]:.1f}  open-end cut at {S['omin'] + 3:.1f}  "
                 f"steady40 open/close {int(S['n']['o40'].sum())}/{int(S['n']['c40'].sum())} in bins, still open/close "
                 f"{int(S['n']['sto'].sum())}/{int(S['n']['stc'].sum())}, arm side {side}")
        print(f"  session {S['label']} done", file=sys.stderr)
    A = [S for S in sess if S["unit"] == "A"]; Bu = [S for S in sess if S["unit"] == "B"]
    nb = len(EDGES)
    # ----------------------------------------------------------------------------------------------------- anchors
    L.append("\nAnchors (100ep-v2, whole dataset)")
    ref = next((t for t in data if t[0] == "openarm-pick-place-100ep-v2"), None)
    if ref is not None:
        check_band(ref[1], L)
        V = ref[1]; whole = np.ones(len(V["pos"]), bool)
        W_ = analyse_session(V, ref[2], whole, 0, 0)
        pos = V["pos"]
        L.append(f"  pre-registered steady mask (40-100 deg/s), whole dataset, open-end cut at {W_['omin'] + 3:.1f}; F2-4 binning "
                 "(10-deg, edges -52, -42, ...): bin n_close tau_close | n_open tau_open")
        for b0 in (-52, -42, -32, -22):
            xo = W_["F"]["o40"]["x"][(pos[W_["F"]["o40"]["i"]] >= b0) & (pos[W_["F"]["o40"]["i"]] < b0 + 10)]
            xc = W_["F"]["c40"]["x"][(pos[W_["F"]["c40"]["i"]] >= b0) & (pos[W_["F"]["c40"]["i"]] < b0 + 10)]
            L.append(f"    [{b0:4d},{b0 + 10:4d})  {len(xc):4d} {fm(np.median(xc) if len(xc) else np.nan, 3, len(xc))} | {len(xo):4d} "
                     f"{fm(np.median(xo) if len(xo) else np.nan, 3, len(xo))}")
        L.append("  same mask, 5-deg bins: bin n_close tau_close | n_open tau_open | f s")
        for b in range(nb):
            no, nc = W_["n"]["o40"][b], W_["n"]["c40"][b]
            if no or nc:
                p = W_["pt"]
                L.append(f"    {blab(EDGES, b):12s} {nc:4d} {fm(p['c40'][b], 3, nc)} | {no:4d} {fm(p['o40'][b], 3, no)} | "
                         f"{fm(p['f40'][b], 3, min(no, nc))} {fm(p['s40'][b], 3, min(no, nc))}")
    # ------------------------------------------------------------------------------------------------ steady tables
    def steady_table(title, ko, kc, kf, ks, kind):
        L.append(f"\n{title}")
        L.append(f"  {'session':30s} {'bin':12s} {'n_o/ep':>8s} {'tau_open [CI]':>22s} {'n_c/ep':>8s} {'tau_close [CI]':>22s} "
                 f"{'f [CI]':>22s} {'s [CI]':>23s}")
        for S in sess:
            p, bt = S["pt"], S["bt"]
            for b in range(nb):
                no, nc = int(S["n"][ko][b]), int(S["n"][kc][b])
                if not (no or nc): continue
                ok = no >= CELL_N and nc >= CELL_N
                ci = lambda k: tuple(np.nanpercentile(bt[k][:, b], [2.5, 97.5])) if a.boot and np.any(np.isfinite(bt[k][:, b])) else (np.nan, np.nan)
                L.append(f"  {S['label']:30s} {blab(EDGES, b):12s} {no:4d}/{S['eps'][ko][b]:<3d} {fm(p[ko][b], 3, no):>7s} {fci(ci(ko)):>14s} "
                         f"{nc:4d}/{S['eps'][kc][b]:<3d} {fm(p[kc][b], 3, nc):>7s} {fci(ci(kc)):>14s} "
                         + (f"{fm(p[kf][b], 3, min(no, nc)):>7s} {fci(ci(kf)):>14s} {fm(p[ks][b], 3, min(no, nc)):>7s} {fci(ci(ks)):>15s}" if ok else
                            f"{'-':>7s} {'':14s} {'-':>7s}"))

    steady_table("Table 1. steady free motion 40-100 deg/s, per session and bin", "o40", "c40", "f40", "s40", "40")
    # -------------------------------------------------------------------------------------------------- still table
    L.append("\nTable 2. still frames (band.py rule on each session's frames): per sign n, p50, p90 [CI]; F = (p90o + p90c) / 2, "
             "off = (p90o - p90c) / 2, off50 = (p50o - p50c) / 2; s40 = the steady s of the same bin; off - s40 [CI] (joint resample)")
    L.append(f"  {'session':30s} {'bin':12s} {'n_o':>5s} {'p50o':>7s} {'p90o [CI]':>22s} {'n_c':>4s} {'p50c':>7s} {'p90c [CI]':>22s} "
             f"{'F [CI]':>22s} {'off [CI]':>23s} {'off50':>7s} {'s40':>7s} {'off - s40 [CI]':>24s}")
    for S in sess:
        p, bt = S["pt"], S["bt"]
        for b in range(nb):
            no, nc = int(S["n"]["sto"][b]), int(S["n"]["stc"][b])
            if not (no or nc): continue
            ok = no >= STILL_N and nc >= STILL_N; oks = ok and valid(S, "40")[b]
            ci = lambda k: tuple(np.nanpercentile(bt[k][:, b], [2.5, 97.5])) if a.boot and np.any(np.isfinite(bt[k][:, b])) else (np.nan, np.nan)
            m = min(no, nc)
            L.append(f"  {S['label']:30s} {blab(EDGES, b):12s} {no:5d} {fm(p['p50o'][b], 3, no):>7s} {fm(p['p90o'][b], 3, no):>7s} {fci(ci('p90o')):>14s} "
                     f"{nc:4d} {fm(p['p50c'][b], 3, nc):>7s} {fm(p['p90c'][b], 3, nc):>7s} {fci(ci('p90c')):>14s} "
                     + (f"{fm(p['Fst'][b], 3, m):>7s} {fci(ci('Fst')):>14s} {fm(p['off'][b], 3, m):>7s} {fci(ci('off')):>15s} "
                        f"{fm(p['off50'][b], 3, m):>7s} " if ok else f"{'-':>7s} {'':14s} {'-':>7s} {'':15s} {'-':>7s} ")
                     + (f"{fm(p['s40'][b], 3, min(S['n']['o40'][b], S['n']['c40'][b])):>7s} {fm(p['dss'][b], 3, min(m, S['n']['o40'][b], S['n']['c40'][b])):>7s} {fci(ci('dss')):>16s}" if oks else ""))
    steady_table("Table 3. FLAGGED: steady 15-40 deg/s band, same exclusions (F8: contaminated by in-hand motion; not used for "
                 "the verdict)", "o15", "c15", "f15", "s15", "15")
    # ----------------------------------------------------------------------------------------------------- pooled
    def pooled_block(title, Ss, items):
        L.append(f"\n{title}")
        L.append(f"  {'bin':12s} " + " ".join(f"{lab:>40s}" for lab, _, _ in items))
        P = [pooled(Ss, key, kind, a.boot) for _, key, kind in items]
        for b in range(nb):
            if all(r[b] is None for r in P): continue
            cells = []
            for r in P:
                x = r[b]
                cells.append("" if x is None else f"{fm(x['med'], 3, x['thin'])} {fci(x['ci'])} {x['min']:.2f}..{x['max']:.2f} k{x['k']}")
            L.append(f"  {blab(EDGES, b):12s} " + " ".join(f"{c:>40s}" for c in cells))
        return P

    L.append("\nPooled cells: per bin the median over the sessions whose cell is valid [95 % CI: episodes resampled within each "
             "session, median of the sessions' draws] min..max over sessions, k = sessions; '!' if the thinner direction has < 30 "
             "frames summed over those sessions. The median of tau_open minus the median of tau_close need not equal 2 x median s.")
    PA40 = pooled_block("Table 4. unit A pooled, steady 40-100: s, f, tau_open, tau_close", A,
                        [("s", "s40", "40"), ("f", "f40", "40"), ("tau_open", "o40", "40"), ("tau_close", "c40", "40")])
    PAst = pooled_block("Table 5. unit A pooled, still: off (p90), F (p90), off50, p50 open, p50 close", A,
                        [("off", "off", "st"), ("F", "Fst", "st"), ("off50", "off50", "st"), ("p50 open", "p50o", "st"), ("p50 close", "p50c", "st")])
    PAd = pooled_block("Table 6. unit A pooled, still minus steady: off - s40 (cells valid in both)", A, [("off - s40", "dss", "dss")])
    PA15 = pooled_block("Table 7. FLAGGED unit A pooled, steady 15-40: s, f, tau_open, tau_close", A,
                        [("s", "s15", "15"), ("f", "f15", "15"), ("tau_open", "o15", "15"), ("tau_close", "c15", "15")])
    PB40 = pooled_block("Table 8. unit B pooled (newarm, operator-50ep), steady 40-100: s, f, tau_open, tau_close", Bu,
                        [("s", "s40", "40"), ("f", "f40", "40"), ("tau_open", "o40", "40"), ("tau_close", "c40", "40")])
    PBst = pooled_block("Table 9. unit B pooled, still: off, F, off50", Bu, [("off", "off", "st"), ("F", "Fst", "st"), ("off50", "off50", "st")])
    # ---------------------------------------------------------------------------------------------------- release
    L.append(f"\nTable 10. release confound: steady 40-100 opening frames within {RELEASE_S} s after the last |tau| > 1.5 Nm frame of "
             "the episode (post) vs later (late); s_late = (tau_open(late) - tau_close) / 2 [CI]")
    for S in sess:
        p, bt = S["pt"], S["bt"]
        L.append(f"  {S['label']:30s} share of steady opening frames that are post-release {S['post_share']:.2f}")
        for b in range(nb):
            n1, n2, nc = int(S["n"]["opost"][b]), int(S["n"]["olate"][b]), int(S["n"]["c40"][b])
            if not (n1 or n2): continue
            ci = tuple(np.nanpercentile(bt["slate"][:, b], [2.5, 97.5])) if a.boot and np.any(np.isfinite(bt["slate"][:, b])) else (np.nan, np.nan)
            ok = n2 >= CELL_N and nc >= CELL_N
            L.append(f"    {blab(EDGES, b):12s} post {n1:4d} {fm(p['opost'][b], 3, n1):>7s}  late {n2:4d} {fm(p['olate'][b], 3, n2):>7s}  "
                     f"close {nc:4d} {fm(p['c40'][b], 3, nc):>7s}  s_late " + (f"{fm(p['slate'][b], 3, min(n2, nc))} {fci(ci)}" if ok else "-")
                     + (f"  s(all) {fm(p['s40'][b], 3, min(S['n']['o40'][b], nc))}" if valid(S, '40')[b] else ""))
    L.append("\nTable 11a. NOT pre-registered, the mirror image for closing: closing frames within 0.5 s before the next |tau| > 1.5 "
             "frame of the episode (approach to a grasp) and closing frames not more than 6 deg wider than the episode's hold (band.py's "
             "'wider' rule; episodes without a hold count as not wider). Share of steady closing frames that are pre-contact / wider:")
    L.append("  " + "; ".join(f"{S['label']} {S['pre_share']:.2f} / {S['far_share']:.2f}" for S in sess))
    PAl = pooled_block("Table 11. unit A pooled: s from late opening frames only (pre-registered); s with closing frames > 6 deg wider "
                       "than the hold only, and s with closing frames not within 0.5 s before contact only (11a); s from all", A,
                       [("s_late", "slate", "late"), ("s_far", "sfar", "far"), ("s_npre", "snpre", "npre"), ("s all", "s40", "40")])
    # ---------------------------------------------------------------------------------------------------- gravity
    L.append("\nTable 12. gravity: |tau| at fixed session, bin and direction regressed on the MJCF gravity torque on the two finger "
             "hinges in the closing sense, g_close (Nm, MJCF masses and mounting, motor angle = finger hinge angle). Cell means "
             "removed; the regressor is multiplied by +1 for opening and -1 for closing frames, so gravity acting as modelled gives "
             "slope k = +1 (x the finger-to-motor torque ratio, 0.8 if the motor turns 1.25 x the finger) and no gravity gives 0. "
             "'effect' = slope x (p95 - p5 of the within-cell regressor). 3-D: the same on the 3 components of the base-frame vector "
             "U (g_close = gravity_base . U, any mounting); shown with the condition number of its normal matrix (the three "
             "components barely vary independently within a cell, so this fit is ill-conditioned and only printed for "
             "completeness). bound = 9.81 |U| (p50 / p99 / max over the session's frames) = the largest |g_close| that gravity "
             "can give at any mounting with MJCF masses and geometry. s_grav = mean g_close of the bin's steady frames = the s "
             "that gravity alone would give (MJCF mounting).")
    if all(S["z"].get("g_close") is not None for S in sess):
        def gfit(Ss, draw=None):
            sxy = sum(S["pt"]["g_sxy"] if draw is None else S["bt"]["g_sxy"][draw] for S in Ss)
            sxx = sum(S["pt"]["g_sxx"] if draw is None else S["bt"]["g_sxx"][draw] for S in Ss)
            xtx = sum(S["pt"]["g_xtx"] if draw is None else S["bt"]["g_xtx"][draw] for S in Ss)
            xty = sum(S["pt"]["g_xty"] if draw is None else S["bt"]["g_xty"][draw] for S in Ss)
            k = sxy / sxx if sxx > 0 else np.nan
            try: beta = np.linalg.solve(xtx, xty)
            except np.linalg.LinAlgError: beta = np.full(3, np.nan)
            return k, beta, float(np.linalg.cond(xtx))
        rows = [(S["label"], [S]) for S in sess] + [("unit A pooled", A), ("unit B pooled", Bu)]
        GA = None
        for lab, Ss in rows:
            k, beta, cond = gfit(Ss)
            if a.boot:
                dr = [gfit(Ss, i) for i in range(a.boot)]
                kci = tuple(np.nanpercentile([d[0] for d in dr], [2.5, 97.5]))
            else: kci = (np.nan, np.nan)
            spread = np.median([S["g_spread"] for S in Ss]); ng = sum(S["g_n"] for S in Ss)
            bound = (np.median([S["g_bound"][0] for S in Ss]), max(S["g_bound"][1] for S in Ss), max(S["g_bound"][2] for S in Ss))
            if lab == "unit A pooled": GA = (k, kci, spread, beta, bound, ng)
            L.append(f"  {lab:30s} n {ng:5d}  k {fm(k, 2, ng)} {fci(kci, 2)}  within-cell regressor p5-p95 {spread:.4f} Nm  effect "
                     f"{fm(k * spread, 4)} [{fm(kci[0] * spread, 4)},{fm(kci[1] * spread, 4)}] Nm  |  bound {bound[0]:.4f} / "
                     f"{bound[1]:.4f} / {bound[2]:.4f} Nm  |  3-D beta/9.81 = {np.round(beta / 9.81, 2).tolist()} (cond {cond:.0f}; "
                     "MJCF mounting predicts k x [0, 0, -1])")
        L.append("  s_grav per bin (MJCF, median over unit A sessions; range over sessions):")
        for b in range(nb):
            v = np.array([S["s_grav"][b] for S in A if np.isfinite(S["s_grav"][b])])
            if len(v): L.append(f"    {blab(EDGES, b):12s} {np.median(v):+.4f}  ({v.min():+.4f}..{v.max():+.4f}, k{len(v)})")
    else:
        GA = None; L.append("  no _fk.npz for every dataset: gravity test skipped")
    # ------------------------------------------------------------------------------------------- absolute vs relative
    L.append("\nTable 13. absolute angle vs distance from the session's open end (unit A sessions). Each session's valid cells, lined "
             "up by absolute bin or by relative bin (theta - range minimum); residual RMS of an additive least-squares model "
             "(one-way = bin only; two-way = bin + session level), dof-corrected. Smaller = that indexing lines the sessions up "
             "better. diff = rel - abs [95 % CI, joint resample]; > 0 favours absolute angle.")
    for lab, ka, kr, kia, kir, tw in (("steady s, one-way", "s40", "s40r", "40", "40r", False), ("steady s, two-way", "s40", "s40r", "40", "40r", True),
                                      ("steady f, two-way", "f40", "f40r", "40", "40r", True), ("steady tau_open, two-way", "o40", "o40r", "40", "40r", True),
                                      ("still p50 open (n >= 15), two-way", "p50o", "p50or", "po", "por", True)):
        pa_, pr_, na_, nr_, ci = abs_vs_rel(A, ka, kr, kia, kir, tw, a.boot)
        L.append(f"  {lab:36s} abs {fm(pa_, 4)} ({na_} cells)  rel {fm(pr_, 4)} ({nr_} cells)  diff {fm(pr_ - pa_, 4)} {fci(ci, 4)}")
    L.append("  per session open end (range minimum) and steady s by relative bin:")
    for S in A:
        cells = [f"{REL_EDGES[b]:g}-{REL_EDGES[b] + 5:g}: {fm(S['pt']['s40r'][b], 3, min(S['nr']['o40'][b], S['nr']['c40'][b]))}"
                 for b in range(len(REL_EDGES)) if valid(S, "40r")[b]]
        L.append(f"    {S['label']:30s} open end {S['range'][0]:6.1f}  " + "  ".join(cells))
    L.append("  per session still p50 open (n >= 15) by absolute bin:")
    for S in A:
        cells = [f"{EDGES[b]:g}: {S['pt']['p50o'][b]:.3f} ({S['n']['sto'][b]})" for b in range(nb) if valid(S, "po")[b]]
        L.append(f"    {S['label']:30s} open end {S['range'][0]:6.1f}  " + "  ".join(cells))
    # ------------------------------------------------------------------------------------------------- predictions
    bi = lambda e: int(round((e - EDGES[0]) / 5))
    b47, b27, b22, b42 = bi(-47), bi(-27), bi(-22), bi(-42)
    g = lambda P, b, key="med": (P[b][key] if P[b] is not None else np.nan)
    L.append("\nTable 14. pre-registered predictions, checked on Table 4 / 5 (unit A pooled) and Table 8 / 9 (unit B)")
    s40 = PA40[0]; f40 = PA40[1]; o40 = PA40[2]; c40 = PA40[3]; offA = PAst[0]
    smax = max(abs(r["med"]) for r in s40 if r is not None); offmax = max((abs(r["med"]) for r in offA if r is not None), default=np.nan)
    L.append(f"  s at [-47,-42) {fm(g(s40, b47))} {fci(s40[b47]['ci']) if s40[b47] else ''} k{s40[b47]['k'] if s40[b47] else 0}; "
             f"at [-27,-22) {fm(g(s40, b27))} {fci(s40[b27]['ci']) if s40[b27] else ''}; at [-22,-17) {fm(g(s40, b22))} "
             f"{fci(s40[b22]['ci']) if s40[b22] else ''}; max |s| over bins {smax:.3f}; still max |off| {fm(offmax)}")
    L.append(f"  f at [-47,-42) {fm(g(f40, b47))} {fci(f40[b47]['ci']) if f40[b47] else ''}; at [-27,-22) {fm(g(f40, b27))}; "
             f"at [-22,-17) {fm(g(f40, b22))}; tau_close at [-47,-42) {fm(g(c40, b47))} {fci(c40[b47]['ci']) if c40[b47] else ''}, "
             f"at [-27,-22) {fm(g(c40, b27))}, at [-22,-17) {fm(g(c40, b22))}")
    tri = lambda x, lo, hi: "undecidable (no valid cell)" if not np.isfinite(x) else ("met" if lo <= x <= hi else "not met")
    out05 = [EDGES[b] for b, r in enumerate(s40) if r is not None and (r["ci"][0] > 0.05 or r["ci"][1] < -0.05)]
    L.append(f"  context, adjacent bin [-42,-37): s {fm(g(s40, b42))} {fci(s40[b42]['ci']) if s40[b42] else ''} k{s40[b42]['k'] if s40[b42] else 0}, "
             f"f {fm(g(f40, b42))}, tau_close {fm(g(c40, b42))}; f range over valid bins "
             f"{min(r['med'] for r in f40 if r is not None):.3f}..{max(r['med'] for r in f40 if r is not None):.3f}")
    L.append(f"  (i) pure friction: every |s| < 0.05: steady point estimates {'met' if smax < 0.05 else 'not met'} (bins whose 95 % CI "
             f"lies wholly outside +-0.05: {out05}); still {'met' if offmax < 0.05 else 'not met'}; f rises toward open: "
             f"f[-47] - f[-22] = {fm(g(f40, b47) - g(f40, b22))}, f[-42] - f[-22] = {fm(g(f40, b42) - g(f40, b22))}")
    L.append(f"  (ii) pure spring: f[-47] in 0.6-0.7 {tri(g(f40, b47), 0.6, 0.7)}, f[-22] in 0.6-0.7 {tri(g(f40, b22), 0.6, 0.7)}; "
             f"s[-47] in 0.2-0.3 {tri(g(s40, b47), 0.2, 0.3)}; tau_close[-47] in 0.35-0.45 {tri(g(c40, b47), 0.35, 0.45)}")
    L.append(f"  (iii) author: s[-47] in 0.15-0.25 {tri(g(s40, b47), 0.15, 0.25)}; f[-47] in 0.75-0.85 {tri(g(f40, b47), 0.75, 0.85)}; "
             f"|s| <= 0.10 at [-27,-22) {tri(abs(g(s40, b27)), 0, 0.10)}, at [-22,-17) {tri(abs(g(s40, b22)), 0, 0.10)}")
    sB = PB40[0]
    L.append("  unit B: s per bin " + ", ".join(f"{EDGES[b]:g}: {fm(r['med'], 3, r['thin'])} {fci(r['ci'])}" for b, r in enumerate(sB) if r is not None)
             + "; f per bin " + ", ".join(f"{EDGES[b]:g}: {fm(r['med'], 3, r['thin'])}" for b, r in enumerate(PB40[1]) if r is not None))
    if GA: L.append(f"  gravity (unit A pooled): k {fm(GA[0], 2)} {fci(GA[1], 2)}, effect {fm(GA[0] * GA[2], 4)} Nm, any-mounting "
                    f"bound max {GA[4][2]:.4f} Nm")
    # ------------------------------------------------------------------------------------------------- sensitivity
    L.append("\nTable 15. sensitivity of the unit A pooled steady 40-100 s and f (point estimates, no resampling) to two mask choices "
             "that are not part of the pre-registration: open-end cut at the session's p0.5 + 3 (the d2 / F8 rule) instead of the "
             "range minimum + 3, and the 5-deg grid at multiples of 5 instead of the -2 offset")
    for lab, rule, edges in (("main (min + 3, offset grid)", "min", EDGES), ("open-end cut p0.5 + 3", "p05", EDGES),
                             ("grid at multiples of 5", "min", np.arange(-60.0, 5.0, 5.0))):
        Ss = []
        for j, S in enumerate(sess):
            if S["unit"] != "A": continue
            Ss.append(analyse_session(S["V"], S["z"], S["sel"], 0, 0, rule, edges))
        cells = []
        for b in range(len(edges)):
            use = [T for T in Ss if valid(T, "40")[b]]
            if not use: continue
            sv = np.median([T["pt"]["s40"][b] for T in use]); fv = np.median([T["pt"]["f40"][b] for T in use])
            cv = np.median([T["pt"]["c40"][b] for T in use])
            thin = min(sum(int(T["n"]["o40"][b]) for T in use), sum(int(T["n"]["c40"][b]) for T in use))
            cells.append(f"{edges[b]:g}: s {fm(sv, 3, thin)} f {fm(fv, 3, thin)} c {fm(cv, 3, thin)} k{len(use)}")
        L.append(f"  {lab:30s} " + "; ".join(cells))
    E10 = np.arange(-57.0, 3.0, 10.0)
    L.append(f"  10-deg bins (edges {E10[0]:g}, {E10[1]:g}, ...; [-47,-37) contains the predictions' [-47,-42)), with 95 % CI, same "
             "resampling (seed 8 + session number):")
    S10 = [dict(analyse_session(S["V"], S["z"], S["sel"], a.boot, 8 + j, "min", E10), label=S["label"], unit=S["unit"])
           for j, S in enumerate(sess)]
    Ss = [T for T in S10 if T["unit"] == "A"]; Ss_B = [T for T in S10 if T["unit"] == "B"]
    P10 = [pooled(Ss, key, "40", a.boot) for key in ("s40", "f40", "o40", "c40")]
    for b in range(len(E10)):
        if P10[0][b] is None: continue
        L.append(f"    {blab(E10, b):12s} " + "  ".join(f"{lab} {fm(r[b]['med'], 3, r[b]['thin'])} {fci(r[b]['ci'])} {r[b]['min']:.2f}..{r[b]['max']:.2f}"
                                                 for lab, r in zip(("s", "f", "tau_open", "tau_close"), P10)) + f"  k{P10[0][b]['k']}")
    k47, k27 = int(round((-47 - E10[0]) / 10)), int(round((-27 - E10[0]) / 10))
    g10 = lambda r, b: r[b]["med"] if r[b] is not None else np.nan
    L.append(f"  predictions re-checked on the 10-deg grid ([-47,-37) for the open end, [-27,-17) for the middle): "
             f"(ii) f[-47,-37) in 0.6-0.7 {tri(g10(P10[1], k47), 0.6, 0.7)}, f[-27,-17) in 0.6-0.7 {tri(g10(P10[1], k27), 0.6, 0.7)}, "
             f"s[-47,-37) in 0.2-0.3 {tri(g10(P10[0], k47), 0.2, 0.3)}, tau_close[-47,-37) in 0.35-0.45 {tri(g10(P10[3], k47), 0.35, 0.45)}; "
             f"(iii) s[-47,-37) in 0.15-0.25 {tri(g10(P10[0], k47), 0.15, 0.25)}, f[-47,-37) in 0.75-0.85 {tri(g10(P10[1], k47), 0.75, 0.85)}, "
             f"|s[-27,-17)| <= 0.10 {tri(abs(g10(P10[0], k27)), 0, 0.10)}; (i) |s| < 0.05 everywhere "
             f"{'met' if all(abs(r['med']) < 0.05 for r in P10[0] if r is not None) else 'not met'}")
    L.append("  10-deg bins, the confound splits (Table 11 / 11a): s_late, s_far, s_npre [CI] k")
    Q10 = [pooled(Ss, key, kind, a.boot) for key, kind in (("slate", "late"), ("sfar", "far"), ("snpre", "npre"))]
    for b in range(len(E10)):
        if all(r[b] is None for r in Q10): continue
        L.append(f"    {blab(E10, b):12s} " + "  ".join(("-" if r[b] is None else f"{lab} {fm(r[b]['med'], 3, r[b]['thin'])} {fci(r[b]['ci'])} k{r[b]['k']}")
                                                 for lab, r in zip(("s_late", "s_far", "s_npre"), Q10)))
    # ------------------------------------------------------------------------------------------------- within-session slopes
    L.append("\nTable 17. NOT pre-registered: within-session change with angle, free of which sessions happen to fill which bin. Per "
             "session, the least-squares slope of the cell values against the bin centre over that session's valid bins (>= 2 bins "
             "spanning >= 10 deg), in Nm per 10 deg TOWARD OPEN (> 0 = larger at the open side). Median over sessions [95 % CI: per "
             "draw the median of the sessions' slopes on their resampled cells] min..max, k sessions.")
    for glab, Sg, edges in (("A 5-deg", A, EDGES), ("A 10-deg", Ss, E10), ("B 5-deg", Bu, EDGES), ("B 10-deg", Ss_B, E10)):
        got = {}
        for key, kind, lab in (("s40", "40", "s"), ("f40", "40", "f"), ("o40", "40", "tau_open"), ("c40", "40", "tau_close"),
                               ("sfar", "far", "s_far"), ("off", "st", "still off"), ("Fst", "st", "still F"), ("p50o", "po", "still p50 open")):
            r = slopes(Sg, key, kind, edges, a.boot); got[lab] = r
            if r is None: L.append(f"  {glab:9s} {lab:16s} -  (no session with >= 2 valid bins)"); continue
            L.append(f"  {glab:9s} {lab:16s} {r['med']:+.3f} {fci(r['ci'])} {r['min']:+.2f}..{r['max']:+.2f} k{r['k']}   per session: "
                     + ", ".join(f"{l} {v:+.3f}" for l, v in r["per"]))
        if got["s"] is not None and got["f"] is not None:
            sh = got["s"]["med"] / (got["s"]["med"] + got["f"]["med"])
            d = got["s"]["dr"] / (got["s"]["dr"] + got["f"]["dr"])
            L.append(f"  {glab:9s} share of the steady opening torque's slope that is conservative, s / (s + f) = {sh:.2f} "
                     f"{fci(tuple(np.nanpercentile(d, [2.5, 97.5])) if a.boot else (np.nan, np.nan), 2)}")
    # ------------------------------------------------------------------------------------------------- dynamics
    L.append("\nTable 16. NOT pre-registered, diagnostic: is the direction difference a speed or acceleration difference between "
             "the directions? Within each (session, bin, direction) cell of the steady 40-100 frames, |tau| is regressed on |v| and on "
             "the acceleration along the motion (a > 0 = speeding up), pooled by summing the normal equations over sessions: "
             "b (Nm per 100 deg/s), J (Nm per 1000 deg/s^2). Per bin: median over valid sessions of the median |v| and a per "
             "direction, s, s_dyn = (b (|v|_open - |v|_close) + J (a_open - a_close)) / 2 per session, s - s_dyn [CI].")
    def dfit(Ss, draw=None):
        xtx = sum(S["pt"]["d_xtx"] if draw is None else S["bt"]["d_xtx"][draw] for S in Ss)
        xty = sum(S["pt"]["d_xty"] if draw is None else S["bt"]["d_xty"][draw] for S in Ss)
        try: return np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError: return np.full(2, np.nan)
    for lab, Ss in (("unit A pooled", A), ("unit B pooled", Bu)):
        be = dfit(Ss); dr = np.array([dfit(Ss, i) for i in range(a.boot)]) if a.boot else np.full((1, 2), np.nan)
        cb, cj = np.nanpercentile(dr[:, 0], [2.5, 97.5]), np.nanpercentile(dr[:, 1], [2.5, 97.5])
        L.append(f"  {lab:14s} n {sum(S['g_n'] for S in Ss):5d}  b {be[0] * 100:+.3f} [{cb[0] * 100:+.3f},{cb[1] * 100:+.3f}] Nm per 100 deg/s  "
                 f"J {be[1] * 1000:+.3f} [{cj[0] * 1000:+.3f},{cj[1] * 1000:+.3f}] Nm per 1000 deg/s^2")
        if lab == "unit A pooled": bA, drA = be, dr
    L.append(f"  {'bin':12s} {'|v| open':>9s} {'|v| close':>9s} {'a open':>7s} {'a close':>7s} {'s':>7s} {'s_dyn':>7s} {'s - s_dyn [CI]':>24s}  k")
    for b in range(nb):
        use = [S for S in A if valid(S, "40")[b]]
        if not use: continue
        sd = lambda S, bb, jj: (bb * (S["vo"][b] - S["vc"][b]) + jj * (S["ao"][b] - S["ac"][b])) / 2
        thin = min(sum(int(S["n"]["o40"][b]) for S in use), sum(int(S["n"]["c40"][b]) for S in use))
        corr = np.median([S["pt"]["s40"][b] - sd(S, bA[0], bA[1]) for S in use])
        ci = (np.nan, np.nan)
        if a.boot:
            d = [np.nanmedian([S["bt"]["s40"][i, b] - sd(S, drA[i, 0], drA[i, 1]) for S in use]) for i in range(a.boot)]
            ci = tuple(np.nanpercentile(d, [2.5, 97.5]))
        L.append(f"  {blab(EDGES, b):12s} {np.median([S['vo'][b] for S in use]):9.1f} {np.median([S['vc'][b] for S in use]):9.1f} "
                 f"{np.median([S['ao'][b] for S in use]):7.1f} {np.median([S['ac'][b] for S in use]):7.1f} "
                 f"{fm(np.median([S['pt']['s40'][b] for S in use]), 3, thin):>7s} {np.median([sd(S, bA[0], bA[1]) for S in use]):+7.3f} "
                 f"{fm(corr, 3, thin):>7s} {fci(ci):>16s}  k{len(use)}")
    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh: fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export"); e.add_argument("roots", nargs="+"); e.add_argument("--out", required=True)
    k = sp.add_parser("fk"); k.add_argument("npzdir"); k.add_argument("--mjcf", required=True)
    k.add_argument("--side", default="auto", choices=("auto", "left", "right"))
    n = sp.add_parser("analyze"); n.add_argument("npzdir"); n.add_argument("--hf-cache", required=True)
    n.add_argument("--out", required=True); n.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"export": cmd_export, "fk": cmd_fk, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
