"""Rigid-cube hold baseline for the hold-creep channel: during teleop holds (|v| < 2 deg/s, tau > 1.5 Nm), how much does
the gripper encoder q move, and does it move only when the leader command a moves? usage: python3 -I hold_quiet.py ROOT..."""
import glob, os, sys, numpy as np, pyarrow.parquet as pq, pyarrow as pa
LSB = 0.021854
for root in sys.argv[1:]:
    tab = pa.concat_tables([pq.read_table(f, columns=["observation.state", "action", "episode_index"]) for f in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))])
    S = np.array(tab.column("observation.state").to_pylist(), float); A = np.array(tab.column("action").to_pylist(), float)
    ep = np.array(tab.column("episode_index").to_pylist()); q, v, tau, a = S[:, 21], S[:, 22], S[:, 23], A[:, 7]
    # unit check: kp_eff from still frames, tau = kp (a - q), both in deg -> rad
    still = (np.abs(v) < 2) & (np.abs(tau) > 0.5)
    kp = np.polyfit(np.radians(a[still] - q[still]), tau[still], 1)[0]
    hold = (np.abs(v) < 2) & (tau > 1.5)
    pair = hold[:-1] & hold[1:] & (ep[:-1] == ep[1:])
    dq, da = np.diff(q)[pair], np.diff(a)[pair]
    steady = np.abs(da) < LSB; busy = np.abs(da) >= 5 * LSB
    print(f"{os.path.basename(root)}: kp_eff {kp:.2f}; hold frame pairs n {pair.sum()} ({hold.sum()/30:.0f} s of hold)")
    print(f"  |dq| >= 1 LSB: all {np.mean(np.abs(dq) >= LSB):.3f}; command steady (<1 LSB, n {steady.sum()}) {np.mean(np.abs(dq[steady]) >= LSB):.3f}; "
          f"command busy (>=5 LSB, n {busy.sum()}) {np.mean(np.abs(dq[busy]) >= LSB) if busy.any() else float('nan'):.3f}")
    print(f"  dq sign when it moves (steady command): closing {np.mean(dq[steady & (np.abs(dq) >= LSB)] < 0):.2f}")
    # per-episode longest contiguous hold run >= 1 s: net q drift, net command drift, duration
    runs = []
    for e in np.unique(ep):
        i = np.nonzero(hold & (ep == e))[0]
        if len(i) < 30: continue
        br = np.nonzero(np.diff(i) > 1)[0]; starts = np.r_[0, br + 1]; ends = np.r_[br, len(i) - 1]
        k = np.argmax(ends - starts); s, t = i[starts[k]], i[ends[k]]
        if t - s < 30: continue
        runs.append((e, (t - s) / 30, q[t] - q[s], a[t] - a[s], tau[t] - tau[s], np.ptp(q[s:t + 1])))
    R = np.array(runs)
    if len(R):
        p = lambda x: np.percentile(x, [16, 50, 84])
        print(f"  longest hold run per episode (n {len(R)}): duration p16/50/84 {np.round(p(R[:,1]),2)} s; net dq {np.round(p(R[:,2]),3)} deg; "
              f"net d_cmd {np.round(p(R[:,3]),3)} deg; net d_tau {np.round(p(R[:,4]),3)} Nm; q range within run {np.round(p(R[:,5]),3)} deg")
        qs = R[np.abs(R[:, 3]) < 0.1]
        if len(qs): print(f"  runs with |net d_cmd| < 0.1 deg (n {len(qs)}): net dq {np.round(p(qs[:,2]),3)} deg, q range {np.round(p(qs[:,5]),3)} deg, duration p50 {np.median(qs[:,1]):.2f} s")
