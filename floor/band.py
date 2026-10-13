"""Friction band vs spring test for the OpenArm 2.0 gripper from torque-logged teleop data. Strict stillness: the
position stays within 0.03 deg over a 7-frame window (creep < 0.13 deg/s) and |vel| < 1 deg/s. At a still motor the
applied torque tau = kp (q_cmd - q) is balanced by whatever resists: static friction (a BAND: both signs occur at the
same position, up to +-Fs) or a spring / soft stop (a FUNCTION of position: one sign per position). Per 5-deg bin of
motor angle we print the 90th percentile of |tau| for each sign; half the sum is the friction estimate, half the
difference the spring-like offset. Steady free motion per bin and direction gives Coulomb friction the same way.
usage: python3 -I band.py ROOT [ROOT ...]"""
import glob, json, os, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa

for root in sys.argv[1:]:
    info = json.load(open(os.path.join(root, "meta/info.json")))
    files = sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True))
    tabs = []
    for f in files:
        try: tabs.append(pq.read_table(f, columns=["observation.state", "action", "episode_index"]))
        except Exception: pass
    tab = pa.concat_tables(tabs)
    S = np.array(tab.column("observation.state").to_pylist(), float); A = np.array(tab.column("action").to_pylist(), float)
    ep = np.array(tab.column("episode_index").to_pylist())
    pos, vel, tau, act = S[:, 21], S[:, 22], S[:, 23], A[:, 7]
    n = len(pos); W = 3
    still = np.abs(vel) < 1.0
    rng = np.zeros(n, bool)
    for i in range(W, n - W):
        seg = pos[i-W:i+W+1]
        rng[i] = (seg.max() - seg.min()) < 0.03 and (ep[i-W] == ep[i+W])
    still &= rng
    lo, hi = np.percentile(pos, 0.5), np.percentile(pos, 99.5)
    mid = (pos > lo + 3) & (pos < hi - 3)
    hold = {}
    for e in np.unique(ep):
        m = (ep == e) & still & (tau > 1.5)
        if m.sum() > 5: hold[e] = np.median(pos[m])
    wider = np.array([pos[i] < hold.get(ep[i], np.nan) - 6 for i in range(n)])
    op = still & mid & (tau < 0); cl = still & mid & (tau > 0) & wider
    print(f"\n{os.path.basename(root)}: frames {n}, strict-still {still.sum()}, pull-open {op.sum()}, push-closed(free) {cl.sum()}, "
          f"hold at {np.median(list(hold.values())):.1f} deg")
    print("  motor angle bin   n_open  p50_open  p90_open   n_close  p50_close  p90_close   Fs~(p90o+p90c)/2  spring~(p90o-p90c)/2")
    for b0 in np.arange(np.floor(lo), np.ceil(hi), 5):
        mo = op & (pos >= b0) & (pos < b0 + 5); mc = cl & (pos >= b0) & (pos < b0 + 5)
        if mo.sum() >= 20 or mc.sum() >= 20:
            po = np.percentile(-tau[mo], [50, 90]) if mo.sum() >= 20 else (np.nan, np.nan)
            pc = np.percentile(tau[mc], [50, 90]) if mc.sum() >= 20 else (np.nan, np.nan)
            print(f"  [{b0:6.1f},{b0+5:6.1f})  {mo.sum():6d}  {po[0]:8.3f}  {po[1]:8.3f}   {mc.sum():7d}  {pc[0]:9.3f}  {pc[1]:9.3f}"
                  f"   {(po[1]+pc[1])/2:8.3f}          {(po[1]-pc[1])/2:8.3f}")
    # dwell of strict-still-under-torque states
    for lab, m in (("pull-open", op), ("push-closed(free)", cl)):
        runs, k = [], 0
        for i in range(n):
            if m[i] and (i == 0 or ep[i] == ep[i-1]): k += 1
            else:
                if k: runs.append(k)
                k = 0
        if k: runs.append(k)
        runs = np.array(runs) if runs else np.zeros(1, int)
        print(f"  dwell {lab}: runs {len(runs)}, >= 0.5 s {np.sum(runs >= 15)}, >= 1 s {np.sum(runs >= 30)}, longest {runs.max()/30:.2f} s; "
              f"|tau| median in runs >= 0.5 s: {np.median(np.abs(tau[m])) if m.sum() else np.nan:.3f}")
    # steady motion per direction and bin
    acc = np.r_[0, np.diff(vel)] * 30.0
    steady = (np.abs(vel) > 15) & (np.abs(acc) < 150) & mid
    print("  steady motion (|v| > 15 deg/s, |a| < 150 deg/s^2): bin, n_close, tau_close median, n_open, tau_open(signed toward motion) median")
    for b0 in np.arange(np.floor(lo), np.ceil(hi), 10):
        mc = steady & (vel > 0) & wider & (pos >= b0) & (pos < b0 + 10); mo = steady & (vel < 0) & (pos >= b0) & (pos < b0 + 10)
        if mc.sum() >= 10 or mo.sum() >= 10:
            print(f"    [{b0:6.1f},{b0+10:6.1f})  {mc.sum():5d}  {np.median(tau[mc]) if mc.sum()>=10 else np.nan:7.3f}   {mo.sum():5d}  {np.median(-tau[mo]) if mo.sum()>=10 else np.nan:7.3f}")
