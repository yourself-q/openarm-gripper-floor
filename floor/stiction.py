"""Static friction band of the OpenArm 2.0 gripper, and the effective MIT kp of every joint, read from a
torque-logged LeRobot dataset (openarm_follower layout: state = pos, vel, torque per joint; action = positions).

Why this works: in MIT position mode with no feedforward the motor's current loop applies tau = kp (q_cmd - q)
+ kd (0 - dq) whether or not the shaft moves. At rest, tau = kp (q_cmd - q) exactly, so (a) regressing tau on the
error over still frames gives kp_eff per joint, which also checks the torque decode scale against the configured
gains; and (b) a still gripper away from its end stops, with the motor pulling toward OPEN (nothing but friction can
resist that) or pushing toward CLOSED at a jaw wider than the object, is held by static friction >= |tau|. The
torque on the last still frame before motion starts is the breakaway torque (static friction itself, if the command
ramps slowly). Steady free motion gives Coulomb + viscous friction directly.
usage: python3 -I stiction.py ROOT [--hold-margin-deg 6]"""
import glob, json, os, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa

root = sys.argv[1]
hm = float(sys.argv[sys.argv.index("--hold-margin-deg") + 1]) if "--hold-margin-deg" in sys.argv else 6.0
info = json.load(open(os.path.join(root, "meta/info.json"))); names = info["features"]["observation.state"]["names"]
files = sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True))
tabs = []
for f in files:
    try: tabs.append(pq.read_table(f, columns=["observation.state", "action", "episode_index", "timestamp"]))
    except Exception as e: print("skip", os.path.basename(f), type(e).__name__)
tab = pa.concat_tables(tabs)
S = np.array(tab.column("observation.state").to_pylist(), float); A = np.array(tab.column("action").to_pylist(), float)
ep = np.array(tab.column("episode_index").to_pylist()); ts = np.array(tab.column("timestamp").to_pylist(), float)
J = [n.split(".")[0] for n in names[::3]]
print(f"{root}: frames {len(S)}, episodes {len(np.unique(ep))}, files {len(tabs)}/{len(files)}")
def still_mask(pos, vel):
    s = np.abs(vel) < 1.0                       # below ~1 LSB (0.84 deg/s)
    s &= np.abs(np.r_[0, np.diff(pos)]) < 0.03  # no position change over the frame (LSB 0.022 deg)
    return s
# (a) kp_eff per joint
print("joint    n_still  kp_eff[Nm/rad]  icpt    R2   (LeRobot defaults 240,240,240,240,24,31,25,25)")
for j in range(8):
    pos, vel, tau, act = S[:, 3*j], S[:, 3*j+1], S[:, 3*j+2], A[:, j]
    s = still_mask(pos, vel); s[1:] &= s[:-1]; s[:-1] &= s[1:]
    e = np.radians(act - pos)[s]; t = tau[s]
    X = np.stack([e, np.ones_like(e)], 1); c, *_ = np.linalg.lstsq(X, t, rcond=None)
    r2 = 1 - np.sum((t - X @ c)**2) / max(np.sum((t - t.mean())**2), 1e-12)
    print(f"{J[j]:8s} {s.sum():7d}   {c[0]:8.2f}      {c[1]:7.3f}  {r2:.3f}")
# (b) gripper
g = 7; pos, vel, tau, act = S[:, 3*g], S[:, 3*g+1], S[:, 3*g+2], A[:, g]
s0 = still_mask(pos, vel)
lo, hi = np.percentile(pos, 0.5), np.percentile(pos, 99.5)
mid = (pos > lo + 3) & (pos < hi - 3)
# hold plateau per episode: still frames with tau > 1.5 (closing against the object)
hold_pos = {}
for e in np.unique(ep):
    m = (ep == e) & s0 & (tau > 1.5)
    if m.sum() > 5: hold_pos[e] = np.median(pos[m])
hp = np.array(list(hold_pos.values()))
print(f"gripper pos {pos.min():.2f}..{pos.max():.2f} deg (0 = closed, negative = open); end-stop guard 3 deg inside "
      f"[{lo:.1f}, {hi:.1f}]; hold plateau in {len(hp)} episodes at pos median {np.median(hp):.1f} (p10 {np.percentile(hp,10):.1f}, "
      f"p90 {np.percentile(hp,90):.1f}), tau median {np.median(tau[s0 & (tau > 1.5)]):.2f} Nm")
open_pull = s0 & mid & (tau < -0.0)                      # motor pulls toward open: only friction resists
wider = np.array([pos[i] < hold_pos.get(ep[i], np.nan) - hm for i in range(len(pos))])
close_free = s0 & mid & (tau > 0) & wider                # pushes toward closed with the jaw wider than the object
def q(x): return np.percentile(x, [10, 25, 50, 75, 90, 99])
print("still-under-torque frames (static friction >= |tau|):")
for lab, m in (("pull toward open", open_pull), (f"push toward closed, jaw > hold + {hm:.0f} deg", close_free)):
    x = np.abs(tau[m]); print(f"  {lab:42s} n {m.sum():6d}  |tau| p10/25/50/75/90/99 = {np.round(q(x), 3)}")
# dwell: how long such states last (frames in runs) — rules out slow creep hidden by the velocity LSB
for lab, m in (("pull toward open", open_pull), ("push toward closed (free jaw)", close_free)):
    runs, k = [], 0
    for i in range(len(m)):
        if m[i] and (i == 0 or ep[i] == ep[i-1]): k += 1
        else:
            if k: runs.append(k)
            k = 0
    if k: runs.append(k)
    runs = np.array(runs)
    big = runs >= 15
    print(f"  dwell {lab}: {len(runs)} runs; >= 0.5 s: {big.sum()}; longest {runs.max()/30:.1f} s")
# position dependence of the band (open-pull frames), 5-deg bins
print("  |tau| median of pull-toward-open still frames by position bin:")
for b0 in np.arange(np.floor(lo), np.ceil(hi), 5):
    m = open_pull & (pos >= b0) & (pos < b0 + 5)
    if m.sum() > 30: print(f"    [{b0:6.1f},{b0+5:6.1f}) n {m.sum():5d}  median {np.median(np.abs(tau[m])):.3f}  p90 {np.percentile(np.abs(tau[m]),90):.3f}")
# breakaway: last still frame before >= 3 frames of motion, mid-range
moving = np.abs(vel) >= 1.7
idx = [i for i in range(2, len(vel) - 3) if s0[i] and s0[i-1] and moving[i+1] and moving[i+2] and moving[i+3]
       and ep[i] == ep[i+3] and mid[i]]
idx = np.array(idx)
if len(idx):
    d = np.sign(vel[idx + 2]); tb = tau[idx] * d
    for sgn, lab in ((1, "closing"), (-1, "opening")):
        m = d == sgn
        if m.sum():
            v = tb[m]; print(f"breakaway {lab:8s} n {m.sum():4d}  tau*dir p10/25/50/75/90 = {np.round(np.percentile(v,[10,25,50,75,90]),3)}  "
                             f"share with torque still opposing the motion {np.mean(v < 0):.2f}")
# steady free motion
acc = np.r_[0, np.diff(vel)] * 30.0
steady = (np.abs(vel) > 20) & (np.abs(acc) < 100) & mid
for sgn, lab in ((1, "closing"), (-1, "opening")):
    m = steady & (np.sign(vel) == sgn) & (wider | (sgn < 0))
    if m.sum() > 20:
        vv, tt = np.abs(np.radians(vel[m])), tau[m] * sgn
        print(f"steady {lab:8s} n {m.sum():4d}  |v| {np.degrees(vv.min()):.0f}..{np.degrees(vv.max()):.0f} deg/s  tau*dir median {np.median(tt):.3f}  "
              f"p25 {np.percentile(tt,25):.3f} p75 {np.percentile(tt,75):.3f}")
