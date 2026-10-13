"""Floor card for one OpenArm 2.0 gripper unit, read from its own torque-logged LeRobot v3 dataset.

The gripper (DM-J4310, MIT position mode, no feedforward, no force sensor) applies tau = kp (q_cmd - q) - kd dq. Every
grasp force below the gripper's own friction is invisible in that torque: the friction is the floor. It differs 4-6x in level (3.7x in within-episode scatter)
between units of the same design (floor_ja.md F2-8), so it should be measured and recorded per unit. Sections:
 1. data: frames, episodes, fps, the columns used, the gripper's range of motion (0 = closed, negative = open).
 2. scale: smallest non-zero step of torque / position / velocity; expect 20/4095 Nm (12 bit on +-10 Nm), 25/65535 rad
    (16 bit on +-12.5 rad) and 60/4095 rad/s (12 bit on +-30 rad/s). A different step means a different decode.
 3. kp_eff: tau regressed on (q_cmd - q)[rad] over still frames (|v| < 1 deg/s, frame-to-frame |dq| < 0.03 deg, both
    neighbours still; stiction.py / F2-2). At rest tau = kp (q_cmd - q) exactly, so the slope checks the torque scale
    against the configured kp.
 4. static band ("push-still"; band.py / F2-5): strict stillness (position within 0.03 deg over 7 frames of one episode
    and |v| < 1 deg/s), mid-range (3 deg inside the p0.5 / p99.5 ends), motor pulling but shaft not moving: static
    friction >= |tau|. Opening = tau < 0 (only friction resists). Closing = tau > 0 with the jaw more than 6 deg wider
    than that episode's hold angle (median position of strict-still frames with tau > 1.5 Nm), i.e. not on the object.
 5. moving friction (steady_spread.py): |v| > 15 deg/s, |acc| < 150 deg/s^2 (central difference), mid-range, both
    neighbours in the same episode; per direction and speed band: n, |tau| p10/p50/p90 and the scatter of |tau| about
    its per-5-deg-bin median (half of p16..p84, and std). For a hand that keeps the motor moving, the floor is this
    scatter, not the static band.
 6. hold: |v| < 2 deg/s and tau > 1.5 Nm for >= 5 consecutive frames of one episode (F2-3 / padcal.py).
 7. pad force F = tau / (2 r): one motor drives both fingers through a 1:1 coupling, r = lever from the finger hinge to
    the pad (official MJCF: 0.028..0.097 m; default 0.06..0.097). If a single finger carries the torque, F = tau / r (x2).
    Motor-to-hinge ratio is assumed 1. These assumptions move the N values by up to ~4x (lever 1.6 x coupling 2 x ratio 1.25); the Nm values do not depend on them.
Sign convention: tau > 0 and v > 0 close the jaw. If the dataset's holds have negative torque, all four gripper
channels are flipped and the card says so.
usage: python3 -I unit_floor.py ROOT [ROOT ...] [--r 0.06 0.097] [--gripper-col 21]"""
import argparse, glob, json, os, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa

ap = argparse.ArgumentParser(description="Print a friction-floor card for each OpenArm LeRobot v3 dataset root.")
ap.add_argument("roots", nargs="+")
ap.add_argument("--r", nargs=2, type=float, default=[0.06, 0.097], help="pad lever range [m]")
ap.add_argument("--gripper-col", type=int, default=21, help="state index of gripper position; +1 velocity, +2 torque")
args = ap.parse_args()
r_lo, r_hi = sorted(args.r)
BANDS = ((15, 40), (40, 100), (100, 400))

def pct(x, q): return np.percentile(x, q) if len(x) else np.nan

def smallest_step(x):
    d = np.diff(np.unique(np.round(x, 7)))
    d = d[d > 1e-7]
    return d.min() if len(d) else np.nan

def card(root):
    info = json.load(open(os.path.join(root, "meta/info.json")))
    fps = float(info.get("fps", 30)); feats = info["features"]
    files = sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))
    tabs = []
    for f in files:
        try: tabs.append(pq.read_table(f, columns=["observation.state", "action", "episode_index"]))
        except Exception as e: print(f"  skip unreadable {os.path.basename(f)} ({type(e).__name__})")
    tab = pa.concat_tables(tabs)
    S = np.array(tab.column("observation.state").to_pylist(), float); A = np.array(tab.column("action").to_pylist(), float)
    ep = np.array(tab.column("episode_index").to_pylist())
    g = args.gripper_col
    sn = feats["observation.state"].get("names") or []; an = feats["action"].get("names") or []
    # gripper command: the action entry with the same name as the gripper position (band.py / stiction.py use action[7])
    ga = an.index(sn[g]) if g < len(sn) and sn[g] in an else 7
    pos, vel, tau, act = S[:, g].copy(), S[:, g + 1].copy(), S[:, g + 2].copy(), A[:, ga].copy()
    n = len(pos)
    loaded = (np.abs(tau) > 1.5) & (np.abs(vel) < 2)
    flipped = loaded.any() and np.median(tau[loaded]) < 0
    if flipped: pos, vel, tau, act = -pos, -vel, -tau, -act
    lo, hi = np.percentile(pos, 0.5), np.percentile(pos, 99.5)
    nm = lambda i, names: names[i] if i < len(names) else "?"
    print(f"\n==== floor card: {os.path.basename(os.path.normpath(root))}")
    print(f"1. data   frames {n}, episodes {len(np.unique(ep))}, fps {fps:g}, files {len(tabs)}/{len(files)}")
    print(f"   columns state[{g}] {nm(g, sn)} (deg), state[{g+1}] {nm(g+1, sn)} (deg/s), state[{g+2}] {nm(g+2, sn)} (Nm), "
          f"action[{ga}] {nm(ga, an)} (command, deg){'  [signs flipped: holds had negative torque]' if flipped else ''}")
    print(f"   range  position {pos.min():.2f} .. {pos.max():.2f} deg (0 = closed, negative = open); "
          f"p0.5 / p99.5 = {lo:.1f} / {hi:.1f}, mid-range guard 3 deg inside")
    # 2. scale
    # float32 storage shifts the smallest gap by ~1 ulp, hence a 0.5 % tolerance
    chk = lambda x, ref: f"{x:.6f} (expect {ref:.6f}, {'ok' if abs(x / ref - 1) < 0.005 else 'CHECK'})"
    print(f"2. scale  smallest step: torque {chk(smallest_step(tau), 20/4095)} Nm, position "
          f"{chk(smallest_step(pos), np.degrees(25/65535))} deg, velocity {chk(smallest_step(vel), np.degrees(60/4095))} deg/s")
    # 3. kp_eff (stiction.py still_mask, then both neighbours still)
    s = (np.abs(vel) < 1.0) & (np.abs(np.r_[0, np.diff(pos)]) < 0.03); s[1:] &= s[:-1]; s[:-1] &= s[1:]
    e = np.radians(act - pos)[s]; t = tau[s]
    X = np.stack([e, np.ones_like(e)], 1); c, *_ = np.linalg.lstsq(X, t, rcond=None)
    r2 = 1 - np.sum((t - X @ c) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-12)
    print(f"3. kp_eff {c[0]:.2f} Nm/rad, intercept {c[1]:+.3f} Nm, R2 {r2:.3f}, n {s.sum()} still frames")
    # 4. static band (band.py)
    W = 3; still = np.abs(vel) < 1.0
    win = np.lib.stride_tricks.sliding_window_view(pos, 2 * W + 1)
    rng = np.zeros(n, bool); rng[W:n - W] = ((win.max(1) - win.min(1)) < 0.03) & (ep[:-2 * W] == ep[2 * W:])
    still &= rng
    mid = (pos > lo + 3) & (pos < hi - 3)
    hold_ep = {}
    for k in np.unique(ep):
        m = (ep == k) & still & (tau > 1.5)
        if m.sum() > 5: hold_ep[k] = np.median(pos[m])
    hold_at = np.array([hold_ep.get(k, np.nan) for k in ep])
    with np.errstate(invalid="ignore"): wider = pos < hold_at - 6
    static = {"closing": still & mid & (tau > 0) & wider, "opening": still & mid & (tau < 0)}
    print(f"4. static band (push-still): strict-still frames {still.sum()}, episodes with a hold {len(hold_ep)}; "
          f"5-deg bins shown when n >= 15")
    for d, m in static.items():
        x = np.abs(tau[m])
        print(f"   {d:8s} n {m.sum():6d}  |tau| p50 {pct(x, 50):.3f}  p90 {pct(x, 90):.3f} Nm")
        for b0 in np.arange(np.floor(lo), np.ceil(hi), 5):
            mb = m & (pos >= b0) & (pos < b0 + 5)
            if mb.sum() >= 15:
                xb = np.abs(tau[mb])
                print(f"     [{b0:6.1f},{b0+5:6.1f}) n {mb.sum():5d}  p50 {pct(xb, 50):.3f}  p90 {pct(xb, 90):.3f}")
    # 5. moving friction (steady_spread.py, exact criteria)
    acc = np.gradient(vel) * fps
    same = np.r_[True, ep[1:] == ep[:-1]] & np.r_[ep[:-1] == ep[1:], True]
    steady = (np.abs(vel) > 15) & (np.abs(acc) < 150) & mid & same
    print(f"5. moving friction: steady frames {steady.sum()} (|v| > 15 deg/s, |acc| < 150 deg/s^2)")
    print(f"   {'direction':9s} {'band dps':>9s} {'n':>5s} {'p10':>6s} {'p50':>6s} {'p90':>6s} | about 5-deg-bin median: "
          f"half(p16..p84)   std  (n)")
    mov = {}
    for d, dm in (("closing", vel > 0), ("opening", vel < 0)):
        for b0, b1 in BANDS:
            m = steady & dm & (np.abs(vel) >= b0) & (np.abs(vel) < b1)
            if m.sum() < 30: print(f"   {d:9s} {b0:4d}-{b1:<4d} {m.sum():5d}  (too few)"); continue
            t = np.abs(tau[m]); p = pos[m]; res = np.full(len(t), np.nan)
            for a0 in np.arange(np.floor(lo), np.ceil(hi), 5):
                k = (p >= a0) & (p < a0 + 5)
                if k.sum() >= 10: res[k] = t[k] - np.median(t[k])
            rr = res[np.isfinite(res)]
            half = (pct(rr, 84) - pct(rr, 16)) / 2
            mov[(d, b0)] = half
            print(f"   {d:9s} {b0:4d}-{b1:<4d} {m.sum():5d} {pct(t, 10):6.2f} {pct(t, 50):6.2f} {pct(t, 90):6.2f} | "
                  f"{half:22.3f} {np.std(rr) if len(rr) else np.nan:6.3f}  ({len(rr)})")
    # 6. hold: runs of >= 5 consecutive frames in one episode
    hm = (np.abs(vel) < 2) & (tau > 1.5)
    start = hm & ~(np.r_[False, hm[:-1]] & np.r_[False, ep[1:] == ep[:-1]])
    rid = np.cumsum(start); cnt = np.bincount(rid[hm], minlength=rid.max() + 1)
    hold = hm & (cnt[rid] >= 5)
    th = tau[hold]
    print(f"6. hold   frames {hold.sum()} in {len(np.unique(ep[hold]))} episodes: |tau| p50 {pct(th, 50):.2f}  p90 "
          f"{pct(th, 90):.2f} Nm, at position p50 {pct(pos[hold], 50):.1f} deg")
    # 7. pad force
    q_t = smallest_step(tau); st90 = pct(np.abs(tau[static["closing"]]), 90)
    ms = mov.get(("closing", 40), np.nan); h50 = pct(th, 50)
    print(f"7. pad force F = tau/(2r), r = {r_lo:.3f}..{r_hi:.3f} m (one motor, both fingers, 1:1 coupling; "
          f"x2 if a single finger carries the torque: F = tau/r)")
    for lab, v in (("quantization", q_t), ("static band p90 (closing)", st90),
                   ("moving scatter (closing, 40-100 dps)", ms), ("hold p50", h50)):
        if not np.isfinite(v): print(f"   {lab:38s}      n/a"); continue
        print(f"   {lab:38s} {v:7.4f} Nm -> {v/(2*r_hi):7.3f} .. {v/(2*r_lo):7.3f} N  (single finger {v/r_hi:.3f} .. {v/r_lo:.3f} N)")
    f = lambda v, dg, why: f"{v:.{dg}f} Nm" if np.isfinite(v) else f"n/a ({why})"
    print(f"unit floor (closing): static p90 {f(st90, 2, 'no push-still frames')}, moving scatter "
          f"{f(ms, 3, 'closing 40-100 dps has < 30 steady frames')}, hold {f(h50, 2, 'no holds')}")

for root in args.roots:
    card(root)
