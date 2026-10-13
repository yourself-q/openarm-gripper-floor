"""D2 unit card: the friction floor of one OpenArm 2.0 gripper, from a CSV of tools/openarm_floor_logger.py, or from a
torque-logged LeRobot v3 dataset root (gtgando/openarm-pick-place-*: same column mapping as unit_floor.py), so the unit A
and unit B cards can be recomputed with this script as a check.

Sign convention as unit_floor.py: tau > 0 and v > 0 close the jaw; 0 deg = closed, negative = open.
Generic block (both inputs; the CSV is first sampled at 30 Hz so every window means what it means in unit_floor.py):
  scale steps, kp_eff, push-still band (unit_floor.py section 4), moving friction bands (section 5), within-stroke scatter
  s (within_stroke.py: closing 40-100 deg/s, residual about the 5-deg-bin median, DOF-corrected pooled std; a group is an
  episode in a dataset and one stroke in a D2 CSV).
D2 block (CSV only, full rate, uses the phase labels):
  torque law fit tau = a (q_cmd - q) + b v + c on P1 (expect a = kp sent, b = -kd sent: the motor reports its MIT torque,
  which includes -kd v while moving); kinetic friction per commanded speed and direction (trimmed p10..p90 mean of the
  tick-mean applied torque times dir, see tick_parts(), in the steady part of each stroke: >= 2 deg of travel and
  >= min(0.5 s, 3 deg / speed) after the stroke starts) and per 5-deg bin;
  P2 creep starts: tau_s = peak of kp (q_cmd - q) before 0.5 deg of travel (at rest the logged torque is exactly that,
  without the torque noise), tau_k = trimmed mean after 1 deg, margin tau_s - tau_k; ripple: per-encoder-step mean torque on
  the P2 runs (from 0.1 deg of travel on), linear detrend, autocorrelation, dominant period = highest peak after the first
  zero crossing, run-to-run correlation of the two runs over the same span (position-locked ripple correlates, noise does
  not); P3 hold-quiet std of tau and q after 2 s of settling; P1 vs P5 drift bracket (kinetic friction at 1 deg/s on the
  same span, with motor temperatures); P4 lever calibration r_eff = [(tau_c + tau_o)_with - (tau_c + tau_o)_without] /
  (2 m g) per finger (signed: friction asymmetry and finger weight cancel) and the coupling verdict, F = tau / (r1 + r2).
Reference columns A / B are this script's own output on gtgando/openarm-pick-place-100ep-v2 (unit A) and -newarm (B).
usage: python d2_analyze.py RUN.csv [--json OUT]        python3 -I d2_analyze.py DATASET_ROOT [ROOT ...]
(numpy only for the CSV; pyarrow is imported only for dataset roots.)"""
import argparse, csv, glob, itertools, json, math, os, re, sys
import numpy as np

Q_LSB, V_LSB, T_LSB = math.degrees(25 / 65535), math.degrees(60 / 4095), 20 / 4095
G = 9.81
TAG = re.compile(r"^(dither|press|release)([0-9.]+)@([0-9.]+)$")   # P6 / P7 run tags: kind, amplitude [deg], frequency [Hz]
BANDS = ((15, 40), (40, 100), (100, 400))
REF = {  # this script's output on the two dataset roots (identical to unit_floor.py / within_stroke.py on the same data);
    # None = fewer than 30 steady frames in that band (B opening 40-100 deg/s has 24)
    "A": {"push_close_p50": 0.286, "push_close_p90": 0.545, "push_open_p50": 0.852, "push_open_p90": 1.292,
          "move_close_40": 0.696, "move_open_40": 0.808, "s_pooled3": 0.0660, "range": "-55.7..0.3"},
    "B": {"push_close_p50": 0.046, "push_close_p90": 0.136, "push_open_p50": 0.076, "push_open_p90": 0.129,
          "move_close_40": 0.203, "move_open_40": None, "s_pooled3": 0.0187, "range": "-45.1..0.5"}}


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def smallest_step(x):
    d = np.diff(np.unique(np.round(x[np.isfinite(x)], 7)))
    d = d[d > 1e-7]
    return float(d.min()) if len(d) else float("nan")


# ----------------------------------------------------------------------------------------------------------------- input
def load_csv(path):
    meta, cur, rows, cols = {}, None, [], None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if ": " in line:
                    k, v = line[2:].rstrip("\n").split(": ", 1)
                    if k == "run": cur = meta.setdefault(int(v), {})
                    if cur is not None: cur[k] = v
            elif cols is None:
                cols = line.strip().split(",")
            elif line.strip():
                rows.append(line.rstrip("\n").split(","))
    C = list(zip(*rows))
    D = {}
    for i, k in enumerate(cols):
        D[k] = np.array(C[i]) if k in ("phase", "tag") else np.array(C[i], dtype=float)
    D["run"] = D["run"].astype(int); D["seg"] = D["seg"].astype(int)
    m0 = meta.get(0, {})
    if int(float(m0.get("closing_sign", 1))) < 0:                        # make closing positive
        for k in ("q", "v", "tau", "q_cmd"): D[k] = -D[k]
        for r in meta.values():
            for k in ("p0_stops_closed_open", "working_range"):
                if k in r: r[k] = " ".join(str(-float(x)) for x in r[k].split())
    return D, meta


def load_dataset(root, gcol):
    import pyarrow as pa, pyarrow.parquet as pq                          # only needed for dataset roots
    info = json.load(open(os.path.join(root, "meta/info.json")))
    fps, feats = float(info.get("fps", 30)), info["features"]
    tabs = []
    for f in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet"))):
        try: tabs.append(pq.read_table(f, columns=["observation.state", "action", "episode_index"]))
        except Exception as e: print(f"  skip unreadable {os.path.basename(f)} ({type(e).__name__})")
    tab = pa.concat_tables(tabs)
    S = np.array(tab.column("observation.state").to_pylist(), float); A = np.array(tab.column("action").to_pylist(), float)
    ep = np.array(tab.column("episode_index").to_pylist())
    sn = feats["observation.state"].get("names") or []; an = feats["action"].get("names") or []
    ga = an.index(sn[gcol]) if gcol < len(sn) and sn[gcol] in an else 7     # unit_floor.py mapping
    pos, vel, tau, act = S[:, gcol].copy(), S[:, gcol + 1].copy(), S[:, gcol + 2].copy(), A[:, ga].copy()
    loaded = (np.abs(tau) > 1.5) & (np.abs(vel) < 2)
    flipped = bool(loaded.any() and np.median(tau[loaded]) < 0)
    if flipped: pos, vel, tau, act = -pos, -vel, -tau, -act
    return dict(pos=pos, vel=vel, tau=tau, act=act, ep=ep, grp=ep, fps=fps, flipped=flipped,
                note=f"{len(np.unique(ep))} episodes, {len(pos)} frames at {fps:g} fps")


def view30(D):
    """Main run (run 0) sampled at 30 Hz, the frame rate the generic definitions were written for."""
    m = D["run"] == 0; t = D["t"][m]
    idx = np.unique(np.searchsorted(t, np.arange(t[0], t[-1], 1 / 30.0)))
    pick = lambda k: D[k][m][idx]
    return dict(pos=pick("q"), vel=pick("v"), tau=pick("tau"), act=pick("q_cmd"), ep=np.zeros(len(idx), int), kp=pick("kp"),
                grp=pick("seg"), phase=pick("phase"), tag=pick("tag"), fps=30.0, flipped=False,
                note=f"{len(idx)} frames of run 0 sampled at 30 Hz")


# ------------------------------------------------------------------------------------------------- generic (unit_floor)
def generic(V, out):
    pos, vel, tau, act, ep, fps = V["pos"], V["vel"], V["tau"], V["act"], V["ep"], V["fps"]
    n = len(pos); lo, hi = np.percentile(pos, 0.5), np.percentile(pos, 99.5)
    G_ = out.setdefault("generic", {})
    G_["range"] = [float(pos.min()), float(pos.max())]
    G_["steps"] = {"tau": smallest_step(tau), "q": smallest_step(pos), "v": smallest_step(vel)}
    s = (np.abs(vel) < 1.0) & (np.abs(np.r_[0, np.diff(pos)]) < 0.03); s[1:] &= s[:-1]; s[:-1] &= s[1:]
    if "kp" in V: s &= V["kp"] == np.median(V["kp"])                     # D2 CSV: leave out the low-gain P0 rows
    e = np.radians(act - pos)[s]; t = tau[s]
    if len(e) > 10:
        X = np.stack([e, np.ones_like(e)], 1); c, *_ = np.linalg.lstsq(X, t, rcond=None)
        G_["kp_eff"] = {"slope": float(c[0]), "intercept": float(c[1]), "n": int(s.sum())}
    # push-still (band.py / unit_floor.py section 4)
    W = 3; still = np.abs(vel) < 1.0
    win = np.lib.stride_tricks.sliding_window_view(pos, 2 * W + 1)
    rng = np.zeros(n, bool); rng[W:n - W] = ((win.max(1) - win.min(1)) < 0.03) & (ep[:-2 * W] == ep[2 * W:])
    still &= rng; mid = (pos > lo + 3) & (pos < hi - 3)
    hold_ep = {}
    for k in np.unique(ep):
        mk = (ep == k) & still & (tau > 1.5)
        if mk.sum() > 5: hold_ep[k] = np.median(pos[mk])
    hold_at = np.array([hold_ep.get(k, np.nan) for k in ep])
    with np.errstate(invalid="ignore"): wider = pos < hold_at - 6
    static = {"closing": still & mid & (tau > 0) & wider, "opening": still & mid & (tau < 0)}
    subsets = {"all": np.ones(n, bool)}
    if "phase" in V: subsets["P2 ramps"] = (V["phase"] == "P2") & (V["tag"] == "creep")
    G_["push_still"] = {}
    for sub, sm in subsets.items():
        for d, m in static.items():
            x = np.abs(tau[m & sm]); bins = {}
            for b0 in np.arange(np.floor(lo), np.ceil(hi), 5):
                xb = np.abs(tau[m & sm & (pos >= b0) & (pos < b0 + 5)])
                if len(xb) >= 15: bins[f"{b0:g}"] = [len(xb), pct(xb, 50), pct(xb, 90)]
            G_["push_still"][f"{sub}/{d}"] = {"n": int(len(x)), "p50": pct(x, 50), "p90": pct(x, 90), "bins": bins}
    # moving friction (steady_spread.py / unit_floor.py section 5)
    acc = np.gradient(vel) * fps
    same = np.r_[True, ep[1:] == ep[:-1]] & np.r_[ep[:-1] == ep[1:], True]
    steady = (np.abs(vel) > 15) & (np.abs(acc) < 150) & mid & same
    G_["moving"] = {}
    for d, dm in (("closing", vel > 0), ("opening", vel < 0)):
        for b0, b1 in BANDS:
            m = steady & dm & (np.abs(vel) >= b0) & (np.abs(vel) < b1)
            if m.sum() < 30: G_["moving"][f"{d}/{b0}"] = {"n": int(m.sum())}; continue
            t = np.abs(tau[m]); p = pos[m]; res = np.full(len(t), np.nan)
            for a0 in np.arange(np.floor(lo), np.ceil(hi), 5):
                k = (p >= a0) & (p < a0 + 5)
                if k.sum() >= 10: res[k] = t[k] - np.median(t[k])
            rr = res[np.isfinite(res)]
            G_["moving"][f"{d}/{b0}"] = {"n": int(m.sum()), "p10": pct(t, 10), "p50": pct(t, 50), "p90": pct(t, 90),
                                         "half": (pct(rr, 84) - pct(rr, 16)) / 2}
    # within-stroke scatter (within_stroke.py; group = episode, or one stroke in a D2 CSV)
    grp = V["grp"]; lo3, hi3 = lo + 3, hi - 3
    acc2 = np.zeros_like(vel); acc2[1:-1] = (vel[2:] - vel[:-2]) / 2 * fps
    same2 = np.zeros(n, bool); same2[1:-1] = (grp[2:] == grp[1:-1]) & (grp[:-2] == grp[1:-1])
    m = same2 & (vel > 40) & (vel <= 100) & (np.abs(acc2) < 150) & (pos > lo3) & (pos < hi3)
    W_ = {"n": int(m.sum())}
    if m.sum() >= 5:
        b = np.floor(pos[m] / 5) * 5; t = np.abs(tau[m]); e = grp[m]
        res = t - np.array([np.median(t[b == bb]) for bb in b])
        eps, cnt = np.unique(e, return_counts=True)
        for kmin in (2, 3):
            ss, dof, pairs = 0.0, 0, []
            for ee, c in zip(eps, cnt):
                if c < kmin: continue
                r = res[e == ee]; ss += np.sum((r - r.mean()) ** 2); dof += len(r) - 1
                pairs += [abs(x - y) for x, y in itertools.combinations(r, 2)]
            W_[f"pooled{kmin}"] = math.sqrt(ss / dof) if dof else float("nan"); W_[f"dof{kmin}"] = dof
            W_[f"robust{kmin}"] = float(np.median(pairs) / 0.954) if pairs else float("nan")
        W_["groups"] = int(len(eps))
    G_["within"] = W_


# ---------------------------------------------------------------------------------------------------------- D2 (CSV only)
def strokes(D, phase, run=0):
    """Steady part of each stroke: dict seg -> (index array, speed, dir)."""
    out = {}
    m0 = (D["run"] == run) & (D["phase"] == phase) & (D["tag"] == "stroke")
    for s in np.unique(D["seg"][m0]):
        i = np.where(m0 & (D["seg"] == s))[0]
        d, v = int(D["dir"][i[0]]), float(D["vcmd"][i[0]])
        moved = (D["q"][i] - D["q"][i[0]]) * d; dt = D["t"][i] - D["t"][i[0]]
        k = i[(moved >= 2.0) & (dt >= min(0.5, 3.0 / v))]
        out[int(s)] = (k, v, d)
    return out


def tmean(x):
    """Mean of the central 80 % (p10..p90): robust to slips and spikes, and not tied to the 12-bit torque grid as a median is."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return float("nan")
    lo, hi = np.percentile(x, [10, 90]); k = x[(x >= lo) & (x <= hi)]
    return float(k.mean()) if len(k) else float(np.median(x))


def tick_parts(D, kp):
    """Mean applied torque over each tick from commands and positions only. The command sent at tick k acts until tick k+1, so
    tau_bar = kp (q_cmd_k - (q_k + q_k+1) / 2) - kd (q_k+1 - q_k) / dt. Returned as the kp part and the velocity (deg/s); the
    kd part is applied to a mean velocity so the 1-LSB position steps do not bias a trimmed mean. A reply frame instead shows
    one instant of the zero-order-hold staircase (in the mock about kp v dt / 2 below the mean: 0.035 Nm at 50 deg/s, 200 Hz)."""
    q, t = D["q"], D["t"]; n = len(q)
    ok = (D["run"][1:] == D["run"][:-1]) & (np.diff(t) > 0)
    fr = np.nan_to_num(D["fresh"], nan=1.0) > 0; ok &= fr[1:] & fr[:-1]
    lag, vel = np.full(n, np.nan), np.full(n, np.nan)
    lag[:-1] = np.where(ok, kp * np.radians(D["q_cmd"][:-1] - (q[:-1] + q[1:]) / 2), np.nan)
    vel[:-1] = np.where(ok, np.diff(q) / np.where(ok, np.diff(t), 1.0), np.nan)
    return lag, vel


def friction(lag, vel, k, d, kd):
    """Kinetic friction [Nm] on the samples k moving in direction d (+1 closing): trimmed tick-mean torque times d."""
    return tmean(lag[k] * d) - kd * math.radians(float(np.nanmean(vel[k]))) * d


def d2(D, meta, out, scale=None):
    m0 = meta[0]; R = out.setdefault("d2", {}); main = D["run"] == 0
    kp, kd = float(m0["kp_sent"]), float(m0["kd_sent"])
    t = D["t"][main]; dts = np.diff(t)
    R["run"] = {"duration_s": float(t[-1] - t[0]), "rate_hz": float((len(t) - 1) / (t[-1] - t[0])),
                "dt_p99_ms": pct(dts, 99) * 1e3, "fresh": float(np.nanmean(D["fresh"][main])),
                "t_mos": [float(np.nanmin(D["t_mos"][main])), float(np.nanmax(D["t_mos"][main]))],
                "t_rotor": [float(np.nanmin(D["t_rotor"][main])), float(np.nanmax(D["t_rotor"][main]))],
                "stops": [float(x) for x in m0.get("p0_stops_closed_open", "nan nan").split()],
                "end": m0.get("end", "missing (run did not finish?)")}
    qcp = np.r_[np.nan, D["q_cmd"][:-1]]                                   # command in force before this tick's command
    qcp[np.r_[True, D["run"][1:] != D["run"][:-1]]] = np.nan
    # torque law: which command (this tick's or the previous one) the reported torque belongs to, and its kp / kd
    m = main & (D["phase"] == "P1") & (D["fresh"] == 1) & np.isfinite(qcp)
    fits = {}
    for name, qc in (("previous", qcp), ("current", D["q_cmd"])):
        X = np.stack([np.radians(qc[m] - D["q"][m]), np.radians(D["v"][m]), np.ones(m.sum())], 1)
        if m.sum() > 100:
            c, *_ = np.linalg.lstsq(X, D["tau"][m], rcond=None)
            fits[name] = {"a": float(c[0]), "b": float(c[1]), "c": float(c[2]), "rms": float((D["tau"][m] - X @ c).std())}
    if fits:
        best = min(fits, key=lambda k: fits[k]["rms"])
        R["law"] = dict(fits[best], command=best, rms_other=fits["current" if best == "previous" else "previous"]["rms"],
                        kp_sent=kp, kd_sent=kd)
        if best == "current": qcp = D["q_cmd"].copy()
        rest = main & (np.abs(D["v"]) < 0.5) & (D["phase"] == "P3") & np.isfinite(qcp)
        e = np.radians(qcp[rest] - D["q"][rest]); big = np.abs(e * kp) > 0.1
        if big.sum() > 20: R["law"]["rest_ratio"] = med(D["tau"][rest][big] / (kp * e[big]))
    lag, vel = tick_parts(D, kp)
    # kinetic friction per commanded speed / direction and per 5-deg bin
    K, bins = {}, {}
    for s, (k, v, d) in strokes(D, "P1").items():
        K.setdefault(f"{v:g}/{'closing' if d > 0 else 'opening'}", []).append((k, d))
    for key, ks in K.items():
        k = np.concatenate([x for x, _ in ks]); d = ks[0][1]; y = D["tau"][k] * d
        b = np.floor(D["q"][k] / 5) * 5
        res = y - np.array([np.median(y[b == bb]) for bb in b]) if len(y) else y
        ss, dof = 0.0, 0
        for kk, _ in ks:
            r = res[np.isin(k, kk)]
            if len(r) > 1: ss += np.sum((r - r.mean()) ** 2); dof += len(r) - 1
        K[key] = {"n": int(len(y)), "friction": friction(lag, vel, k, d, kd), "reading": tmean(y), "p10": pct(y, 10),
                  "p90": pct(y, 90), "strokes": len(ks), "s_within": math.sqrt(ss / dof) if dof else float("nan")}
        for bb in np.unique(b):
            kb = k[b == bb]
            if len(kb) >= 20: bins.setdefault(f"{bb:g}", {})[key] = friction(lag, vel, kb, d, kd)
    R["kinetic"], R["kinetic_bins"] = K, bins
    # P2 creep starts and ripple
    C, prof = [], {1: [], -1: []}
    seg_m = main & (D["phase"] == "P2") & (D["tag"] == "creep")
    for s in np.unique(D["seg"][seg_m]):
        i = np.where(seg_m & (D["seg"] == s))[0]; d = int(D["dir"][i[0]])
        dw = np.where(main & (D["seg"] == s - 1))[0]
        q0 = med(D["q"][dw]) if len(dw) else D["q"][i[0]]
        mv = (D["q"][i] - q0) * d
        qs = np.array([np.median(D["q"][i][max(0, j - 10):j + 11]) for j in range(len(i))])   # 0.1 s running median
        lg = kp * np.radians(qcp[i] - qs) * d                          # = the torque while the jaw is at rest
        pre = mv < 0.5
        row = {"seg": int(s), "dir": d, "q0": q0, "travel": float(mv.max())}
        if (~pre).any():
            row["tau_s"] = float(np.nanmax(lg[pre]))
            post = mv >= 1.0
            if post.sum() > 20:
                row["tau_k"] = tmean(D["tau"][i][post] * d)
                row["tau_k_kp"] = tmean(lg[post]) - kd * math.radians(float(np.mean(D["v"][i][post]))) * d
                row["margin"] = row["tau_s"] - row["tau_k_kp"]
        else:
            row["tau_s_above"] = float(np.nanmax(lg))                    # no breakaway within the 5-deg ramp
        C.append(row)
        c = i[mv >= 0.1]
        if len(c) > 50:
            st = np.where(np.isfinite(D["q_raw"][c]), D["q_raw"][c], np.floor(D["q"][c] / Q_LSB)).astype(int)
            u = np.arange(st.min(), st.max() + 1); x = np.full(len(u), np.nan); nv = np.full(len(u), np.nan)
            y = D["tau"][c] * d
            for j, uu in enumerate(u):
                yy = y[st == uu]
                if len(yy): x[j] = yy.mean(); nv[j] = yy.var(ddof=1) / len(yy) if len(yy) > 1 else np.nan
            f = np.isfinite(x); row["steps_visited"] = float(f.mean())
            if f.sum() > 20:
                p = np.polyfit(u[f], x[f], 1); x = x - np.polyval(p, u)
                prof[d].append((u, x, nv))
    R["creep"] = C
    for d in (1, -1):
        rows = [r for r in C if r["dir"] == d and "margin" in r]
        R[f"creep_{'closing' if d > 0 else 'opening'}"] = {k: med([r[k] for r in rows]) for k in ("tau_s", "tau_k", "margin")} if rows else {}
    R["ripple"] = ripple(prof)
    # P3 hold quiet
    H = {}
    for tag in ("hold_mid", "hold_open"):
        i = np.where(main & (D["phase"] == "P3") & (D["tag"] == tag))[0]
        if len(i) < 10: continue
        i = i[D["t"][i] >= D["t"][i[0]] + 2.0]
        tt = D["t"][i] - D["t"][i[0]]; slope = np.polyfit(tt, D["tau"][i], 1)[0] * 60 if len(i) > 10 else np.nan
        H[tag] = {"tau_mean": float(D["tau"][i].mean()), "tau_std": float(D["tau"][i].std()), "q_std": float(D["q"][i].std()),
                  "q_std_lsb": float(D["q"][i].std() / Q_LSB), "q_distinct": int(len(np.unique(D["q"][i]))),
                  "tau_slope_per_min": float(slope), "q_mean": float(D["q"][i].mean())}
    R["hold"] = H
    # P1 vs P5 drift bracket at 1 deg/s (same span by construction)
    Dr = {}
    for ph in ("P1", "P5"):
        for s, (k, v, d) in strokes(D, ph).items():
            if v != 1.0 or len(k) < 20: continue
            Dr.setdefault(ph, {}).setdefault(d, []).append(friction(lag, vel, k, d, kd))
            Dr.setdefault(ph + "_T", []).append((float(np.nanmean(D["t_mos"][k])), float(np.nanmean(D["t_rotor"][k])),
                                                 float(D["t"][k].mean())))
    if "P1" in Dr and "P5" in Dr:
        R["drift"] = {}
        for d in (1, -1):
            a, b = Dr["P1"].get(d, []), Dr["P5"].get(d, [])
            if a and b:
                R["drift"]["closing" if d > 0 else "opening"] = {"P1": float(np.mean(a)), "P5": float(np.mean(b)),
                    "P1_strokes": a, "P5_strokes": b, "change": float(np.mean(b) - np.mean(a)),
                    "change_pct": float(100 * (np.mean(b) / np.mean(a) - 1))}
        for ph in ("P1", "P5"):
            T = np.array(Dr[ph + "_T"]); R["drift"][ph + "_temp"] = {"t_mos": float(T[:, 0].mean()), "t_rotor": float(T[:, 1].mean()),
                                                                       "t_s": float(T[:, 2].mean())}
    R["p4"] = p4(D, meta, kp, kd)
    R["dither"] = dither_block(D, meta, kp, kd, scale)


def ripple(prof):
    """Average autocorrelation of the detrended per-step torque profiles; dominant period after the first zero crossing."""
    out = {}
    allp = prof[1] + prof[-1]
    if not allp: return out
    K = min(150, min(len(u) for u, _, _ in allp) // 2)
    num, den, var_s, var_n, nst = np.zeros(K + 1), np.zeros(K + 1), 0.0, [], 0
    for u, x, nv in allp:
        f = np.isfinite(x); xs = np.where(f, x, 0.0)
        var_s += np.sum(xs ** 2); nst += f.sum(); var_n += list(nv[np.isfinite(nv)])
        for k in range(K + 1):
            num[k] += np.sum(xs[:len(xs) - k] * xs[k:]); den[k] += np.sum(f[:len(f) - k] & f[k:])
    acf = (num / np.maximum(den, 1)) / (var_s / max(nst, 1))
    out["acf"] = [float(a) for a in acf]
    z = np.where(acf[1:] < 0)[0]
    if len(z):
        k0 = z[0] + 1; cand = [k for k in range(k0 + 1, K) if acf[k] >= acf[k - 1] and acf[k] >= acf[k + 1]]
        if cand:
            kb = max(cand, key=lambda k: acf[k])
            out.update(period_steps=int(kb), period_deg=kb * Q_LSB, acf_peak=float(acf[kb]))
    sig2 = var_s / max(nst, 1); noise2 = float(np.mean(var_n)) if var_n else 0.0
    out["rms"] = math.sqrt(max(0.0, sig2 - noise2)); out["noise_rms_per_step"] = math.sqrt(noise2)
    out["max_period_steps"] = K
    rr = []
    for d in (1, -1):
        for (u1, x1, _), (u2, x2, _) in itertools.combinations(prof[d], 2):
            common = np.intersect1d(u1, u2)
            a, b = x1[np.isin(u1, common)], x2[np.isin(u2, common)]; f = np.isfinite(a) & np.isfinite(b)
            if f.sum() > 20: rr.append(float(np.corrcoef(a[f], b[f])[0, 1]))
    out["run_to_run_r"] = rr
    return out


def p4(D, meta, kp, kd):
    """Lever calibration per finger. Signed applied torque per stroke over the position window that all strokes of that run
    share (friction that varies with position then cancels too), round-trip sums S = tau_close + tau_open, and
    r_eff = (S_with - S_without) / (2 m g). Squeezing an object, both pads push with the same F, so tau = F (r1 + r2)."""
    res = {}; lag, vel = tick_parts(D, kp)
    for run, mm in meta.items():
        if run == 0 or mm.get("phase") != "P4": continue
        f, mass = int(mm["finger"]), float(mm["bottle_kg"])
        st = [(s, k, d) for s, (k, v, d) in strokes(D, "P4", run).items() if len(k) > 20]
        lo = max(D["q"][k].min() for s, k, d in st); hi = min(D["q"][k].max() for s, k, d in st)
        sums = {0: [], 1: []}
        for s, k, d in st:
            k = k[(D["q"][k] >= lo) & (D["q"][k] <= hi)]
            if len(k) > 20: sums[int(D["bottle"][k[0]])].append((s, d, d * friction(lag, vel, k, d, kd)))
        rt = {}
        for b in (0, 1):                                                   # a closing stroke and the opening stroke after it
            L = sorted(sums[b]); rt[b] = [L[j][2] + L[j + 1][2] for j in range(len(L) - 1) if L[j][1] > 0 > L[j + 1][1]]
        if len(rt[0]) >= 2 and len(rt[1]) >= 2:
            dS = np.mean(rt[1]) - np.mean(rt[0])
            se = math.sqrt(np.var(rt[1], ddof=1) / len(rt[1]) + np.var(rt[0], ddof=1) / len(rt[0]))
            res[f] = {"r_eff_m": float(dS / (2 * mass * G)), "se_m": float(se / (2 * mass * G)), "mass_kg": mass,
                      "window_deg": [float(lo), float(hi)], "sum_without": [float(x) for x in rt[0]], "sum_with": [float(x) for x in rt[1]]}
    if res:
        if len(res) < 2: res["verdict"] = "one finger measured: run P4 on the other finger for the coupling verdict"; return res
        r1, r2 = res[1]["r_eff_m"], res[2]["r_eff_m"]; tol = max(0.005, 2 * math.hypot(res[1]["se_m"], res[2]["se_m"]))
        if min(r1, r2) < -0.005 or max(r1, r2) > 0.09: v = "implausible lever: repeat P4 (bottle on the pad? wrist turned so the finger closes upward?)"
        elif min(r1, r2) < 0.010: v = f"one finger carries the torque: pad force F = tau / r = tau / {max(r1, r2):.4f} m"
        else:
            v = (f"both fingers load the motor: pad force F = tau / (r1 + r2) = tau / {r1 + r2:.4f} m; levers "
                 + ("agree" if abs(r1 - r2) <= tol else "differ") + f" ({1e3 * abs(r1 - r2):.1f} mm apart, tolerance {1e3 * tol:.1f} mm)")
        if not all(0.045 <= r <= 0.075 or r < 0.010 for r in (r1, r2)): v += "; outside the expected 45-75 mm"
        res["verdict"] = v
    return res


def cycle_means(D, j, per, lag, vel, kd, obj=None):
    """Non-overlapping windows of `per` frames over the index array j: per window the mean time, the mean applied torque
    (tick-mean kp lag minus kd times the mean velocity, closing-positive), mean q, mean base command, the jaw's dither
    amplitude on the encoder (std * sqrt 2) and, when given, the mean object torque. A dither of one period per window
    averages out of the command, so what remains is the torque the dither leaves on average."""
    rows = []
    for w in range(len(j) // per):
        k = j[w * per:(w + 1) * per]
        lg = lag[k]; ok = np.isfinite(lg)
        if ok.sum() < 0.8 * per: continue
        tau = float(np.nanmean(lg)) - kd * math.radians(float(np.nanmean(vel[k])))
        rows.append({"t": float(D["t"][k].mean()), "tau": tau, "q": float(D["q"][k].mean()), "q_cmd": float(D["q_cmd"][k].mean()),
                     "q_amp": float(D["q"][k].std() * math.sqrt(2)), "obj": float(np.nanmean(obj[k])) if obj is not None else float("nan")})
    return rows


def dither_block(D, meta, kp, kd, scale=None):
    """P6 (unloaded dithered creep) and P7 (dithered press on a scale), one row per run tag (amplitude a, frequency f).
    P6: offset = trimmed mean over windows of the cycle-mean torque times the direction, from 1 deg of base travel on (the
    start transient is left out, as in P2); scatter = residual std of that series about its linear trend in position
    (dof n - 2); q_amp = median jaw dither amplitude; moved = share of windows with q_amp > 0.05 deg, i.e. the jaw actually
    reversing. If the friction were all between the pad and the encoder, the dither could not touch it and these rows would
    equal the a = 0 row. P7: object torque (mock column 'obj', or the scale CSV converted with the lever) against the
    cycle-mean applied torque, press and the release after it as one loop. f_left = half the loop's width in applied
    torque at a given object torque (median over 9 levels inside the common range): the friction still hidden between the
    applied torque and the pad at this amplitude (first-crossing times, so a stuck staircase counts as its full band).
    centre = applied torque minus object torque at the loop's middle: the zero / lever error (0 if lever and zero are
    right). The mock predicts f_left = max(0, f_s - kp a) + nothing for the pad side: a force swing resets Coulomb
    friction at the pad as well, so the loop does not separate motor-side from pad-side friction; q_amp (the encoder's
    dither amplitude while the pad is blocked) does, since the motor side slips against the wind-up only once kp a
    exceeds the friction upstream of the encoder. transmission (JSON only) = slope of object torque against applied
    torque over the press windows in contact."""
    res = {"P6": [], "P7": []}; lag, vel = tick_parts(D, kp); rate = 1.0 / float(np.median(np.diff(D["t"][D["run"] == 0])))
    obj = D.get("obj")
    if scale is not None:
        ts, nm = scale; obj = np.interp(D["t"], ts, nm)
    P7 = []
    for run in sorted(meta):
        for tag in set(D["tag"][D["run"] == run]):
            m = TAG.match(tag)
            if not m: continue
            kind, amp, f = m.group(1), float(m.group(2)), float(m.group(3))
            per = max(1, round(rate / f)) if f > 0 else round(0.5 * rate)
            for seg in np.unique(D["seg"][(D["run"] == run) & (D["tag"] == tag)]):
                j = np.where((D["run"] == run) & (D["seg"] == seg) & (D["tag"] == tag))[0]      # the motion, not rise / taper
                d = int(D["dir"][j[0]]); W = cycle_means(D, j, per, lag, vel, kd, obj)
                if len(W) < 4: continue
                if kind == "dither":
                    Wt = [w for w in W if (w["q_cmd"] - W[0]["q_cmd"]) * d >= 1.0]
                    if len(Wt) < 4: continue
                    y = np.array([w["tau"] * d for w in Wt]); x = np.array([w["q"] for w in Wt])
                    M = np.vstack([x, np.ones_like(x)]).T; c, *_ = np.linalg.lstsq(M, y, rcond=None)
                    qa = np.array([w["q_amp"] for w in Wt])
                    res["P6"].append({"run": int(run), "seg": int(seg), "dir": "closing" if d > 0 else "opening", "amp_deg": amp, "hz": f,
                                      "offset": float(tmean(y)), "scatter": float(math.sqrt(((y - M @ c) ** 2).sum() / (len(y) - 2))),
                                      "q_amp": float(np.median(qa)), "moved": float((qa > 0.05).mean()), "n": len(Wt),
                                      "t_s": float(np.mean([w["t"] for w in Wt]))})
                else:
                    P7.append((int(run), int(seg), kind, amp, f, W))
    P7.sort(key=lambda e: e[:2])
    for i, (run, seg, kind, amp, f, P) in enumerate(P7):
        if kind != "press" or i + 1 >= len(P7) or P7[i + 1][2] != "release" or P7[i + 1][0] != run or P7[i + 1][3] != amp: continue
        Rl = P7[i + 1][5]
        row = {"run": run, "seg": seg, "amp_deg": amp, "hz": f, "n": len(P) + len(Rl),
               "q_amp": float(np.median([w["q_amp"] for w in P])), "tau_max": float(max(w["tau"] for w in P))}
        po, pt = np.array([w["obj"] for w in P]), np.array([w["tau"] for w in P])
        ro, rt = np.array([w["obj"] for w in Rl]), np.array([w["tau"] for w in Rl])
        if np.isfinite(po).all() and np.isfinite(ro).all():
            pm, rm = np.maximum.accumulate(po), np.minimum.accumulate(ro)
            lo, hi = max(pm[0], rm[-1], 0.03), min(pm[-1], rm[0])
            if hi - lo >= 0.1:
                lv = np.linspace(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), 9)
                tp = np.array([pt[np.argmax(pm >= v)] for v in lv]); tr = np.array([rt[np.argmax(rm <= v)] for v in lv])
                half = 0.5 * (tp - tr)
                row.update({"f_left": float(np.median(half)), "f_left_min": float(half.min()), "f_left_max": float(half.max()),
                            "centre": float(np.median(0.5 * (tp + tr) - lv)), "obj_range": [float(lo), float(hi)]})
            k = po >= 0.03
            if k.sum() >= 4:
                M = np.vstack([pt[k], np.ones(int(k.sum()))]).T; (sl, c), *_ = np.linalg.lstsq(M, po[k], rcond=None)
                row.update({"transmission": float(sl), "rms": float(np.sqrt(((po[k] - M @ (sl, c)) ** 2).mean())), "n_contact": int(k.sum())})
        res["P7"].append(row)
    return res


# ---------------------------------------------------------------------------------------------------------------- output
def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def print_card(name, out, ref=True):
    G_, R = out["generic"], out.get("d2")
    L = [f"\n==== D2 unit card: {name}  ({out['source']}; {out['note']})"]
    mt = out.get("meta", {})
    if mt: L.append("   " + "; ".join(f"{k} {mt[k]}" for k in ("unit_note", "thermal", "kp_sent", "kd_sent", "lerobot_installed", "mock") if k in mt))
    if R:
        r = R["run"]
        L.append(f"   run {r['duration_s']:.0f} s, loop {r['rate_hz']:.1f} Hz (tick p99 {r['dt_p99_ms']:.1f} ms), fresh frames "
                 f"{r['fresh']:.1%}, T_mos {r['t_mos'][0]:.0f}..{r['t_mos'][1]:.0f} C, T_rotor {r['t_rotor'][0]:.0f}..{r['t_rotor'][1]:.0f} C; "
                 f"stops closed {fmt(r['stops'][0], 2)} / open {fmt(r['stops'][1], 2)} deg; end: {r['end']}")
    st = G_["steps"]
    L.append(f"1. scale  steps tau {fmt(st['tau'], 6)} (expect {T_LSB:.6f}) Nm, q {fmt(st['q'], 6)} (expect {Q_LSB:.6f}) deg, "
             f"v {fmt(st['v'], 4)} (expect {V_LSB:.4f}) deg/s; range {G_['range'][0]:.2f}..{G_['range'][1]:.2f} deg")
    if "kp_eff" in G_: L.append(f"   kp_eff {G_['kp_eff']['slope']:.2f} Nm/rad over {G_['kp_eff']['n']} still frames (unit_floor.py)")
    if R and "law" in R:
        w = R["law"]
        L.append(f"   torque law on P1: tau = {w['a']:.2f} (q_cmd - q) {w['b']:+.3f} v {w['c']:+.3f} [rad, rad/s] with the {w['command']} "
                 f"tick's command, rms {w['rms']:.4f} Nm (other {w['rms_other']:.4f}); sent kp {w['kp_sent']:.3f}, kd {w['kd_sent']:.3f}; "
                 f"at rest tau / kp(q_cmd - q) = {fmt(w.get('rest_ratio'), 4)}")
    L.append("2. push-still (unit_floor.py definition; depends on how the jaw was driven):   n    p50    p90 Nm")
    for k, v in G_["push_still"].items():
        L.append(f"   {k:20s} {v['n']:6d} {fmt(v['p50'])} {fmt(v['p90'])}")
    L.append("3. moving friction (unit_floor.py bands, 30 Hz view):  n  |tau| p50  half-scatter about 5-deg bins")
    for k, v in G_["moving"].items():
        if "p50" in v: L.append(f"   {k:14s} dps {v['n']:5d} {fmt(v['p50'], 2):>9s} {fmt(v['half']):>8s}")
    if R:
        L.append("   kinetic friction per commanded speed, steady part of P1 strokes at full rate: friction = trimmed tick-mean "
                 "applied torque; reading = trimmed mean of the logged torque [p10..p90]; s = scatter within a stroke")
        for k, v in sorted(R["kinetic"].items(), key=lambda kv: (kv[0].split("/")[1], float(kv[0].split("/")[0]))):
            L.append(f"   {k:14s} friction {fmt(v['friction'])}  reading {fmt(v['reading'])} [{fmt(v['p10'])}..{fmt(v['p90'])}]"
                     f"  s {fmt(v['s_within'])}  n {v['n']}, strokes {v['strokes']}")
        keys = sorted({k for b in R["kinetic_bins"].values() for k in b}, key=lambda s: (s.split("/")[1], float(s.split("/")[0])))
        L.append("   friction per 5-deg bin " + " ".join(f"{k:>11s}" for k in keys))
        for b in sorted(R["kinetic_bins"], key=float):
            L.append(f"   [{float(b):5.0f},{float(b) + 5:4.0f})           " + " ".join(f"{fmt(R['kinetic_bins'][b].get(k)):>11s}" for k in keys))
    w = G_["within"]
    L.append(f"4. within-stroke scatter s (within_stroke.py: closing 40-100 dps, 30 Hz): pooled >=2 {fmt(w.get('pooled2'), 4)} "
             f"(dof {w.get('dof2', 0)}), >=3 {fmt(w.get('pooled3'), 4)} (dof {w.get('dof3', 0)}), robust {fmt(w.get('robust2'), 4)}; frames {w['n']}")
    if R:
        L.append("5. creep starts (P2, 0.2 deg/s): tau_s peak of kp(q_cmd - q) before 0.5 deg, tau_k trimmed mean after 1 deg")
        for r in R["creep"]:
            L.append(f"   seg {r['seg']:3d} {'closing' if r['dir'] > 0 else 'opening'} from {r['q0']:7.2f} deg, travel {r['travel']:.2f}, "
                     f"steps visited {r.get('steps_visited', float('nan')):.0%}: "
                     + (f"tau_s {fmt(r.get('tau_s'))}  tau_k {fmt(r.get('tau_k'))} (kp-based {fmt(r.get('tau_k_kp'))})  margin {fmt(r.get('margin'))}"
                        if "tau_s_above" not in r else f"no breakaway; tau_s > {r['tau_s_above']:.3f}"))
        for d in ("closing", "opening"):
            c = R[f"creep_{d}"]
            if c: L.append(f"   {d}: tau_s {fmt(c['tau_s'])}, tau_k {fmt(c['tau_k'])}, tau_s - tau_k {fmt(c['margin'])} Nm")
        rp = R["ripple"]
        if rp:
            L.append(f"6. ripple (P2): dominant period {rp.get('period_steps', 'none')} encoder steps"
                     + (f" = {rp['period_deg']:.3f} deg, acf {rp['acf_peak']:.2f}" if "period_steps" in rp else "")
                     + f"; rms {rp['rms']:.4f} Nm above the per-step noise {rp['noise_rms_per_step']:.4f}; same-span run-to-run r "
                     + ", ".join(f"{x:.2f}" for x in rp["run_to_run_r"]) + f" (periods up to {rp['max_period_steps']} steps resolvable)")
        for k, h in R["hold"].items():
            L.append(f"7. hold quiet {k:9s} at {h['q_mean']:7.2f} deg: tau {h['tau_mean']:+.3f} std {h['tau_std']:.4f} Nm "
                     f"(drift {h['tau_slope_per_min']:+.4f} Nm/min); q std {h['q_std']:.4f} deg = {h['q_std_lsb']:.2f} LSB, {h['q_distinct']} distinct")
        if "drift" in R:
            dr = R["drift"]; t1, t5 = dr["P1_temp"], dr["P5_temp"]
            L.append(f"8. drift bracket (1 deg/s, same span): P1 at {t1['t_s']:.0f} s (T_mos {t1['t_mos']:.1f}, T_rotor {t1['t_rotor']:.1f} C) "
                     f"-> P5 at {t5['t_s']:.0f} s (T_mos {t5['t_mos']:.1f}, T_rotor {t5['t_rotor']:.1f} C)")
            for d in ("closing", "opening"):
                if d in dr:
                    x = dr[d]; L.append(f"   {d}: {x['P1']:.3f} -> {x['P5']:.3f} Nm ({x['change']:+.3f}, {x['change_pct']:+.1f} %); strokes "
                                        + " ".join(f"{v:.3f}" for v in x["P1_strokes"]) + " | " + " ".join(f"{v:.3f}" for v in x["P5_strokes"]))
        p = R["p4"]
        if p:
            for f in (1, 2):
                if f in p:
                    L.append(f"9. lever P4 finger {f}: r_eff {1e3 * p[f]['r_eff_m']:.1f} +- {1e3 * p[f]['se_m']:.1f} mm "
                             f"(bottle {p[f]['mass_kg']:.3f} kg; round-trip sums without " + " ".join(f"{x:+.3f}" for x in p[f]["sum_without"])
                             + " / with " + " ".join(f"{x:+.3f}" for x in p[f]["sum_with"]) + " Nm)")
            L.append(f"   verdict: {p['verdict']}")
        else:
            L.append("9. lever P4: not run")
        dz = R.get("dither", {})
        if dz.get("P6"):
            L.append("10. dither P6, unloaded creep 0.2 deg/s, per dither cycle: offset = mean torque x dir [Nm], scatter about the trend "
                     "[Nm], jaw amplitude [deg], share of cycles the jaw reversed")
            for r in dz["P6"]:
                L.append(f"   {r['dir']:7s} a {r['amp_deg']:3.1f} deg @ {r['hz']:g} Hz: offset {r['offset']:+.3f}  scatter {r['scatter']:.3f}  "
                         f"q_amp {r['q_amp']:.2f}  moved {100 * r['moved']:3.0f} %  (n {r['n']}, t {r['t_s']:.0f} s)")
        else:
            L.append("10. dither P6: not run")
        if dz.get("P7"):
            L.append("11. dither P7, press on the scale, per amplitude: f_left = friction still between applied torque and pad [Nm] "
                     "(half the press/release loop width, [min, max] over 9 levels); centre = zero/lever error [Nm]; "
                     "q_amp = encoder dither amplitude with the pad blocked [deg] (grows once kp a exceeds the friction upstream of the encoder)")
            for r in dz["P7"]:
                if "f_left" in r:
                    L.append(f"   run {r['run']} a {r['amp_deg']:3.1f} deg @ {r['hz']:g} Hz: f_left {fmt(r.get('f_left'))} "
                             f"[{fmt(r.get('f_left_min'))}, {fmt(r.get('f_left_max'))}]  centre {fmt(r.get('centre'))}  "
                             f"q_amp {r['q_amp']:.2f}  tau_max {r['tau_max']:.2f}  (cycles {r['n']}, obj {r['obj_range'][0]:.2f}..{r['obj_range'][1]:.2f} Nm)")
                else:
                    L.append(f"   run {r['run']} a {r['amp_deg']:3.1f} deg @ {r['hz']:g} Hz: no object torque in contact "
                             "(pass --scale-csv T_KG.csv --lever-mm R; t_s = CSV time of each reading)")
        else:
            L.append("11. dither P7: not run")
    if ref:
        ps, mv = G_["push_still"], G_["moving"]
        mine = {"push_close_p50": ps["all/closing"]["p50"], "push_close_p90": ps["all/closing"]["p90"],
                "push_open_p50": ps["all/opening"]["p50"], "push_open_p90": ps["all/opening"]["p90"],
                "move_close_40": mv.get("closing/40", {}).get("p50"), "move_open_40": mv.get("opening/40", {}).get("p50"),
                "s_pooled3": w.get("pooled3"), "range": f"{G_['range'][0]:.1f}..{G_['range'][1]:.1f}"}
        L.append("next to unit A (100ep-v2) and unit B (newarm), same definitions:   this unit        A        B")
        for k, v in mine.items():
            g = (lambda x: f"{x:>8s}") if k == "range" else (lambda x: f"{fmt(x):>8s}")
            L.append(f"   {k:16s} {'':30s}{g(v)} {g(REF['A'][k])} {g(REF['B'][k])}")
        out["vs_ref"] = mine
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="Print a D2 friction-floor card for a logger CSV or a LeRobot v3 dataset root.")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json", default=None, help="JSON output path (one input), default <input>.card.json for a CSV")
    ap.add_argument("--gripper-col", type=int, default=21)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--scale-csv", default=None, help="P7: readings transcribed from the video, columns t_s,kg (t_s = CSV time)")
    ap.add_argument("--lever-mm", type=float, default=None, help="P7: pad lever [mm] that turns the scale reading into motor torque")
    a = ap.parse_args()
    scale = None
    if (a.scale_csv is None) != (a.lever_mm is None): ap.error("--scale-csv and --lever-mm go together")
    if a.scale_csv:
        rows = [r for r in csv.reader(open(a.scale_csv)) if r and not r[0].startswith("#")]
        try: ts = np.array([float(r[0]) for r in rows]); kg = np.array([float(r[1]) for r in rows])
        except ValueError: ts = np.array([float(r[0]) for r in rows[1:]]); kg = np.array([float(r[1]) for r in rows[1:]])
        o = np.argsort(ts); scale = (ts[o], kg[o] * G * a.lever_mm / 1e3)
    for path in a.inputs:
        if os.path.isdir(path):
            V = load_dataset(path, a.gripper_col); out = {"source": "LeRobot v3 dataset", "note": V["note"]}
            if V["flipped"]: out["note"] += ", signs flipped (holds had negative torque)"
            generic(V, out)
        else:
            D, meta = load_csv(path); V = view30(D)
            out = {"source": "D2 CSV", "note": V["note"], "meta": meta.get(0, {})}
            generic(V, out); d2(D, meta, out, scale)
        print_card(os.path.basename(os.path.normpath(path)), out, ref=not a.no_ref)
        jp = a.json if a.json and len(a.inputs) == 1 else (None if os.path.isdir(path) else path + ".card.json")
        if jp:
            with open(jp, "w") as f: json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
            print(f"wrote {jp}")


if __name__ == "__main__":
    main()
