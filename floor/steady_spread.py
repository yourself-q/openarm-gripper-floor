"""Friction while the gripper is moving: level and scatter, per speed band and direction. The question is what a hand
that keeps the motor moving (kinetic friction, no stiction) can resolve: the floor is then the scatter of the friction
torque, not the static band. Steady motion: |v| > 15 deg/s and |acc| < 150 deg/s^2 (as band.py), away from both ends
of the range. Per direction and speed band: n, tau p10/p50/p90, and the scatter of tau around its per-5-deg-bin median
(p16..p84 half-width and std) so that angle dependence is removed.

usage: python3 -I steady_spread.py ROOT [ROOT ...]"""
import glob, os, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa

for root in sys.argv[1:]:
    tabs = []
    for f in sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True)):
        try: tabs.append(pq.read_table(f, columns=["observation.state", "episode_index"]))
        except Exception: pass
    tab = pa.concat_tables(tabs)
    S = np.array(tab.column("observation.state").to_pylist(), float); ep = np.array(tab.column("episode_index").to_pylist())
    pos, vel, tau = S[:, 21], S[:, 22], S[:, 23]
    acc = np.gradient(vel) * 30.0; same = np.r_[True, ep[1:] == ep[:-1]] & np.r_[ep[:-1] == ep[1:], True]
    lo, hi = np.percentile(pos, 0.5), np.percentile(pos, 99.5); mid = (pos > lo + 3) & (pos < hi - 3) & same
    steady = (np.abs(vel) > 15) & (np.abs(acc) < 150) & mid
    print(f"\n{os.path.basename(root)}: frames {len(pos)}, steady {steady.sum()}")
    print(f"  {'direction':8s} {'speed band':>12s} {'n':>6s} {'p10':>6s} {'p50':>6s} {'p90':>6s} | scatter about angle-bin median: {'half(p16..p84)':>14s} {'std':>6s}")
    for dname, dm in (("closing", vel > 0), ("opening", vel < 0)):
        for b0, b1 in ((15, 40), (40, 100), (100, 400)):
            m = steady & dm & (np.abs(vel) >= b0) & (np.abs(vel) < b1)
            if m.sum() < 30: print(f"  {dname:8s} {b0:4d}-{b1:<4d} dps {m.sum():6d}  (too few)"); continue
            t = np.abs(tau[m]); p = pos[m]
            res = np.full(len(t), np.nan)
            for a0 in np.arange(np.floor(lo), np.ceil(hi), 5):
                k = (p >= a0) & (p < a0 + 5)
                if k.sum() >= 10: res[k] = t[k] - np.median(t[k])
            r = res[np.isfinite(res)]
            print(f"  {dname:8s} {b0:4d}-{b1:<4d} dps {m.sum():6d} {np.percentile(t,10):6.2f} {np.percentile(t,50):6.2f} {np.percentile(t,90):6.2f} | "
                  f"{(np.percentile(r,84)-np.percentile(r,16))/2:14.3f} {np.std(r):6.3f}  (n {len(r)})")
