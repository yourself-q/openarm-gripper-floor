"""First look at a gripper channel of a torque-logged LeRobot OpenArm dataset: layout, units, torque quantization
step, torque at rest, torque vs velocity in free motion (hysteresis = friction band), hold plateau.
usage: python3 -I torque_look.py ROOT [GRIPPER_NAME_SUBSTRING]   (ROOT holds meta/info.json and data/)"""
import glob, json, os, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa

root = sys.argv[1]; want = sys.argv[2] if len(sys.argv) > 2 else "gripper"
info = json.load(open(os.path.join(root, "meta/info.json")))
feats = info["features"]
files = sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))
tabs = []
for f in files:
    try: tabs.append(pq.read_table(f))
    except Exception as e: print("skip", f, type(e).__name__)
tab = pa.concat_tables(tabs)
print("rows", tab.num_rows, "files", len(tabs), "of", len(files))
def col(name):
    c = tab.column(name)
    if pa.types.is_fixed_size_list(c.type) or pa.types.is_list(c.type):
        return np.array(c.to_pylist(), dtype=float)
    return np.array(c.to_pylist())
# locate gripper pos / vel / torque wherever they live
loc = {}
for fname, fd in feats.items():
    for i, n in enumerate(fd.get("names") or []):
        n = str(n).lower()
        if want in n and fname.startswith("observation"):
            for key, tags in (("pos", (".pos",)), ("vel", (".vel",)), ("tau", (".torque", ".tau", ".effort"))):
                if any(n.endswith(t) for t in tags): loc[key] = (fname, i, n)
    for i, n in enumerate(fd.get("names") or []):
        if want in str(n).lower() and fname == "action": loc["act"] = (fname, i, n)
print("located:", loc)
cache = {}
def get(key):
    fname, i, _ = loc[key]
    if fname not in cache: cache[fname] = col(fname)
    return cache[fname][:, i]
pos, vel, tau, act = get("pos"), get("vel"), get("tau"), get("act")
ep = col("episode_index"); ts = col("timestamp")
print(f"pos range {pos.min():.3f}..{pos.max():.3f}  vel range {vel.min():.3f}..{vel.max():.3f}  "
      f"tau range {tau.min():.4f}..{tau.max():.4f}  action range {act.min():.3f}..{act.max():.3f}")
def step(x, eps=1e-7):
    u = np.unique(np.round(x, 7)); d = np.diff(u); d = d[d > eps]
    return len(u), (d.min() if len(d) else np.nan), np.round(np.sort(d)[:6], 6)
nu, s, ss = step(tau); print(f"torque: unique {nu}, smallest gap {s:.6f}, smallest gaps {ss}; DM4310 LSB 20/4095 = {20/4095:.6f}, 14/4095 = {14/4095:.6f}")
nu, s, ss = step(pos); print(f"pos: unique {nu}, smallest gap {s:.6f}; 16-bit +-12.5 rad = {25/65535:.6f} rad = {np.degrees(25/65535):.6f} deg")
nu, s, ss = step(vel); print(f"vel: unique {nu}, smallest gap {s:.6f}; 12-bit +-30 rad/s = {60/4095:.6f} rad/s = {np.degrees(60/4095):.4f} deg/s")
hold_is_min = np.median(tau[pos < np.percentile(pos, 5)]) > np.median(tau[pos > np.percentile(pos, 95)])
print("hold (large torque) at", "min pos" if hold_is_min else "max pos")
print("per-episode: pos min..max, median tau at the closed 10 %, share tau>7, share |tau|>3, action min..max")
for e in np.unique(ep)[:10]:
    m = ep == e; p, t, a = pos[m], tau[m], act[m]
    close = p < np.percentile(p, 10) if hold_is_min else p > np.percentile(p, 90)
    print(f"  ep {int(e):3d}: pos {p.min():8.3f}..{p.max():8.3f}  tau@closed {np.median(t[close]):7.3f}  "
          f"tau>7 {np.mean(t>7):.3f}  |tau|>3 {np.mean(np.abs(t)>3):.3f}  act {a.min():8.3f}..{a.max():8.3f}")
print(f"all: share tau>7 {np.mean(tau>7):.3f}; |tau|>3 {np.mean(np.abs(tau)>3):.3f}; |tau|<0.1 {np.mean(np.abs(tau)<0.1):.3f}")
span = pos.max() - pos.min(); margin = 0.15 * span
plateau = np.percentile(pos, 5) if hold_is_min else np.percentile(pos, 95)
free = (pos > plateau + margin) if hold_is_min else (pos < plateau - margin)
print(f"free frames {free.sum()} of {len(pos)} (beyond plateau {plateau:.2f} by {margin:.2f})")
vmax = np.percentile(np.abs(vel[free]), 99)
edges = np.concatenate([-np.geomspace(vmax, vmax/200, 8), [0], np.geomspace(vmax/200, vmax, 8)])
print("  vel bin                    n   tau median   tau mean    tau sd")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = free & (vel >= lo) & (vel < hi)
    if m.sum() > 20:
        print(f"  [{lo:9.3f},{hi:9.3f})  {m.sum():6d}  {np.median(tau[m]):10.4f}  {tau[m].mean():9.4f}  {tau[m].std():8.4f}")
still = free & (np.abs(vel) < vmax / 200)
if still.sum() > 10:
    print(f"at rest and free: n {still.sum()}, tau mean {tau[still].mean():.4f}, sd {tau[still].std():.4f}, "
          f"p5 {np.percentile(tau[still],5):.4f}, p95 {np.percentile(tau[still],95):.4f}")
m = free & (np.abs(vel) > vmax / 50)
X = np.stack([np.ones(m.sum()), vel[m], np.sign(vel[m])], 1)
coef, *_ = np.linalg.lstsq(X, tau[m], rcond=None)
print(f"free-motion fit tau = Fo + Fv v + Fc sign(v) (n {m.sum()}): Fo {coef[0]:.4f}  Fv {coef[1]:.5f} (per vel unit)  "
      f"Fc {coef[2]:.4f}   | Enactic gripper Fc 0.0512, Fv 0.084 Nm s/rad, Fo -0.050")
np.savez(os.path.join(root, f"_{want}.npz"), pos=pos, vel=vel, tau=tau, act=act, ep=ep, ts=ts)
