"""F8: is the gripper friction floor a property of the unit or of the session?

Runs the unit-card definitions of d2_analyze.py (generic(): push-still band, moving friction bands, within-stroke scatter,
kp_eff) over every torque-logged gtgando OpenArm LeRobot dataset root given, and adds what a single card cannot show:
  - Hugging Face dates (recording-date proxy; info.json has no date): first / last commit and lastModified, and, for every
    "Upload folder" commit, total_episodes in meta/info.json at that revision, so each upload batch is an episode range
    with an upload time (an upper bound on when those episodes were recorded);
  - moving closing friction (|tau| p50, unit_floor.py section 5, 40-100 deg/s) per episode-index quartile (the rule that
    reproduces the 0.78 -> 0.77 -> 0.69 -> 0.64 Nm drift of 100ep-v2, see quartiles()) and per upload batch;
  - a 95 % interval for every p50 by resampling episodes (frames within an episode are not independent);
  - a bin-matched level per group (mean of per-5-deg-bin medians over bins every group of that dataset fills with >= 5
    frames), so a change in which jaw angles were visited is not read as a change in friction;
  - a unit assignment from three markers, kept apart: config (kp_eff within 2 % of 16.01 Nm/rad and the closed end p99.5
    within 3 deg of 0; otherwise "other config"), level (moving closing 40-100 deg/s p50 nearest of the unit A / unit B
    reference levels in log ratio, "other" if more than a factor 1.5 from both) and range (range minimum nearer to A's
    -55.7 or B's -45.1 deg). The 15-40 deg/s band is not used for the level: in two datasets it is dominated by the jaw
    squeezing an object while still moving (p50 2.8 Nm), which the moving-band definition does not exclude;
  - within each upload batch, the level in the first and second half of its episodes (within-session drift);
  - per 5-deg bin medians of the 40-100 band for every dataset and batch, its median speed and in-band slope of |tau| on
    |v|, and an angle-adjusted level (each frame divided by the 100ep-v2 band median of its 5-deg bin), so neither the jaw
    angles nor the speeds visited are read as a change in friction; a spread summary (max / min, coefficient of variation)
    over the datasets and sessions of each unit;
  - for datasets with >= 2 upload batches, the whole card per batch (generic() run on that batch's episodes) and every card
    quantity per batch with an episode-bootstrap interval, a jaw-angle-matched level on a fixed 5-deg grid (the push-still
    mask is copied verbatim from generic() and checked against its n and p50) and an odd/even-episode split-half as the
    within-session noise scale.
Every number that rests on fewer than MIN_N = 30 frames is printed with a "!" (n too small). d2_analyze.py is imported,
not modified; the copied masks assert equality with its output on every dataset and batch.

Datasets without meta/info.json (openarm-cube-organization-*) are read with load_noinfo(): same column mapping as
d2_analyze.load_dataset (state 21/22/23 = gripper pos/vel/torque, action 7 = gripper command), fps from the timestamps.
Datasets whose data parquet has no footer (written but never closed) cannot be read; their row keeps the info.json totals
and the meta/stats.json gripper range.

Network: only huggingface.co (api/datasets/..., datasets/.../resolve/<sha>/meta/info.json), only when a cache file is
missing, and only with --fetch. Downloaded JSON is treated as data.
usage: python3 -I f8_sessions.py ROOT [ROOT ...] --hf-cache DIR --out TABLE.txt [--fetch] [--boot 2000]"""
import argparse, contextlib, glob, io, json, math, os, sys, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I leaves the script directory off sys.path
import d2_analyze as d2                                          # noqa: E402  (the unit-card definitions, unchanged)

MIN_N = 30                  # same threshold as the analyzer's "too few" rule for a speed band
REF = {"A": d2.REF["A"]["move_close_40"], "B": d2.REF["B"]["move_close_40"]}   # 0.696 / 0.203 Nm
REF_OPEN_END = {"A": -55.7, "B": -45.1}                          # range minimum of 100ep-v2 / newarm (deg)
LEVEL_FACTOR = 1.5          # "other" if the level is more than this factor away from both references


# ------------------------------------------------------------------------------------------------------- Hugging Face
def hf_json(url, cache, fetch):
    if os.path.exists(cache):
        with open(cache) as f: return json.load(f)
    if not fetch: return None
    if not url.startswith("https://huggingface.co/"): raise ValueError(url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "f8-sessions/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r: data = json.loads(r.read())
    except Exception as e:
        print(f"  fetch failed {url}: {type(e).__name__}", file=sys.stderr); return None
    with open(cache, "w") as f: json.dump(data, f)
    return data


def hf_dates(repo, cache_dir, fetch):
    """Commit list, dataset info and the upload batches (episode ranges per 'Upload folder' commit)."""
    name = repo.split("/")[1]
    # the commits endpoint returns 50 per page; page p >= 1 is ?p=p (page 0 is cached as {name}__commits.json)
    commits, pg = [], 0
    while True:
        url = f"https://huggingface.co/api/datasets/{repo}/commits/main" + (f"?p={pg}" if pg else "")
        page = hf_json(url, os.path.join(cache_dir, f"{name}__commits" + (f"_p{pg}" if pg else "") + ".json"), fetch) or []
        commits += page
        if len(page) < 50 or pg >= 20: break
        pg += 1
    commits = list({c["id"]: c for c in commits}.values())
    info = hf_json(f"https://huggingface.co/api/datasets/{repo}", os.path.join(cache_dir, f"{name}__info.json"), fetch) or {}
    commits = sorted(commits, key=lambda c: c["date"])
    batches, prev = [], 0
    for c in commits:
        if not c.get("title", "").startswith("Upload folder"): continue
        ij = hf_json(f"https://huggingface.co/datasets/{repo}/resolve/{c['id']}/meta/info.json",
                     os.path.join(cache_dir, f"{name}__info_at_{c['id'][:10]}.json"), fetch)
        if not ij or "total_episodes" not in ij: continue          # no info.json at that revision (upload-large-folder sets)
        tot = int(ij["total_episodes"])
        if tot > prev: batches.append({"date": c["date"], "sha": c["id"][:10], "ep0": prev, "ep1": tot - 1, "frames_total": ij.get("total_frames")})
        prev = max(prev, tot)
    return {"first": commits[0]["date"] if commits else None, "last": commits[-1]["date"] if commits else None,
            "n_commits": len(commits), "lastModified": info.get("lastModified"), "createdAt": info.get("createdAt"), "batches": batches}


# ------------------------------------------------------------------------------------------------------------- input
def load_noinfo(root, gcol=21):
    """d2_analyze.load_dataset without meta/info.json: same mapping and sign check; fps from the timestamps."""
    import pyarrow as pa, pyarrow.parquet as pq
    tabs = []
    for f in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet"))):
        try: tabs.append(pq.read_table(f, columns=["observation.state", "action", "episode_index", "timestamp"]))
        except Exception as e: print(f"  skip unreadable {os.path.basename(f)} ({type(e).__name__})")
    tab = pa.concat_tables(tabs)
    S = np.array(tab.column("observation.state").to_pylist(), float); A = np.array(tab.column("action").to_pylist(), float)
    ep = np.array(tab.column("episode_index").to_pylist()); ts = np.array(tab.column("timestamp").to_pylist(), float)
    dt = np.diff(ts)[np.diff(ep) == 0]; fps = float(round(1.0 / np.median(dt)))
    pos, vel, tau, act = S[:, gcol].copy(), S[:, gcol + 1].copy(), S[:, gcol + 2].copy(), A[:, 7].copy()
    loaded = (np.abs(tau) > 1.5) & (np.abs(vel) < 2)
    flipped = bool(loaded.any() and np.median(tau[loaded]) < 0)
    if flipped: pos, vel, tau, act = -pos, -vel, -tau, -act
    return dict(pos=pos, vel=vel, tau=tau, act=act, ep=ep, grp=ep, fps=fps, flipped=flipped,
                note=f"{len(np.unique(ep))} episodes, {len(pos)} frames at {fps:g} fps (no info.json: mapping assumed, fps from timestamps)")


def load(root):
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            if os.path.exists(os.path.join(root, "meta/info.json")): V = d2.load_dataset(root, 21)
            else: V = load_noinfo(root)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:90]}", log.getvalue()
    return V, None, log.getvalue()


# ---------------------------------------------------------------------------------------------------------- analysis
def band_mask(V, b0, b1, closing=True):
    """unit_floor.py section 5 / d2_analyze.generic() moving-friction frames, copied verbatim (checked against generic())."""
    pos, vel, ep, fps = V["pos"], V["vel"], V["ep"], V["fps"]
    lo, hi = np.percentile(pos, 0.5), np.percentile(pos, 99.5); mid = (pos > lo + 3) & (pos < hi - 3)
    acc = np.gradient(vel) * fps
    same = np.r_[True, ep[1:] == ep[:-1]] & np.r_[ep[:-1] == ep[1:], True]
    steady = (np.abs(vel) > 15) & (np.abs(acc) < 150) & mid & same
    return steady & ((vel > 0) if closing else (vel < 0)) & (np.abs(vel) >= b0) & (np.abs(vel) < b1)


def push_mask(V):
    """unit_floor.py section 4 / d2_analyze.generic() push-still frames (closing, opening), copied verbatim (checked against
    generic()): still for 7 frames within 0.03 deg, mid-range, closing = tau > 0 and > 6 deg wider than the episode's hold."""
    pos, vel, tau, ep = V["pos"], V["vel"], V["tau"], V["ep"]
    n = len(pos); lo, hi = np.percentile(pos, 0.5), np.percentile(pos, 99.5)
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
    return {"closing": still & mid & (tau > 0) & wider, "opening": still & mid & (tau < 0)}


def boot_ci(t, e, B, seed=8):
    """95 % percentile interval of the pooled median when whole episodes are resampled."""
    ue = np.unique(e)
    if len(ue) < 3 or B <= 0: return (float("nan"), float("nan"))
    groups = [t[e == u] for u in ue]; rng = np.random.default_rng(seed)
    meds = np.empty(B)
    for i in range(B):
        meds[i] = np.median(np.concatenate([groups[j] for j in rng.integers(0, len(ue), len(ue))]))
    return tuple(float(x) for x in np.percentile(meds, [2.5, 97.5]))


def quartiles(e):
    """Episode-index quartile label per frame. Rule that reproduces the 100ep-v2 drift 0.78 / 0.77 / 0.69 / 0.64 Nm: cut
    points = 25/50/75th percentiles (linear) of the indices of the episodes that contribute >= 1 frame to the band."""
    q = np.percentile(np.unique(e), [25, 50, 75])
    return np.searchsorted(q, e, side="left")


def bin_matched(t, p, labels, minn=5):
    """Per group: mean of its per-5-deg-bin medians over the bins that every group fills with >= minn frames."""
    b = np.floor(p / 5) * 5; groups = [g for g in np.unique(labels)]
    common = [bb for bb in np.unique(b) if all(np.sum((labels == g) & (b == bb)) >= minn for g in groups)]
    lv = {g: (float(np.mean([np.median(t[(labels == g) & (b == bb)]) for bb in common])) if common else float("nan")) for g in groups}
    return lv, [float(x) for x in common]


def binm_ci(t, e, p, common, B, seed=8):
    """95 % interval of the bin-matched level (mean over the given 5-deg bins of the per-bin medians) when whole episodes are
    resampled; a bin left empty by a resample is skipped in that resample's mean."""
    ue = np.unique(e)
    if len(ue) < 3 or B <= 0 or not common: return (float("nan"), float("nan"))
    b = np.floor(p / 5) * 5; idx = [np.flatnonzero(e == u) for u in ue]; rng = np.random.default_rng(seed); v = np.empty(B)
    for i in range(B):
        k = np.concatenate([idx[j] for j in rng.integers(0, len(ue), len(ue))]); tb, bb = t[k], b[k]
        meds = [np.median(tb[bb == c]) for c in common if np.any(bb == c)]
        v[i] = np.mean(meds) if meds else np.nan
    return tuple(float(x) for x in np.nanpercentile(v, [2.5, 97.5]))


def group_stats(t, e, p, labels, B):
    rows = []
    lv, common = bin_matched(t, p, labels)
    for g in np.unique(labels):
        s = labels == g
        rows.append({"label": g, "n": int(s.sum()), "eps": int(len(np.unique(e[s]))), "ep_range": (int(e[s].min()), int(e[s].max())),
                     "p50": float(np.median(t[s])), "ci": boot_ci(t[s], e[s], B), "binm": lv[g]})
    return rows, common


def analyse(root, B):
    V, err, log = load(root)
    R = {"root": root, "error": err, "log": log.strip()}
    if V is None: return R
    out = {}
    with contextlib.redirect_stdout(io.StringIO()): d2.generic(V, out)
    G = out["generic"]; R["G"] = G; R["V"] = V; R["frames"] = len(V["pos"]); R["episodes"] = int(len(np.unique(V["ep"])))
    R["fps"] = V["fps"]; R["flipped"] = V["flipped"]
    R["p_lo"], R["p_hi"] = float(np.percentile(V["pos"], 0.5)), float(np.percentile(V["pos"], 99.5))
    tau, ep, pos, spd = np.abs(V["tau"]), V["ep"], V["pos"], np.abs(V["vel"])
    for key, (b0, b1) in (("c40", (40, 100)), ("c15", (15, 40)), ("call", (15, 1e9))):
        m = band_mask(V, b0, b1)
        R[key] = {"n": int(m.sum()), "eps": int(len(np.unique(ep[m]))), "t": tau[m], "e": ep[m], "p": pos[m], "v": spd[m]}
        if m.sum():
            R[key]["p50"] = float(np.median(tau[m])); R[key]["ci"] = boot_ci(tau[m], ep[m], B)
    # the copied mask must give the analyzer's own numbers
    for key, gk in (("c40", "closing/40"), ("c15", "closing/15")):
        g = G["moving"][gk]
        assert g["n"] == R[key]["n"], (root, key, g["n"], R[key]["n"])
        if "p50" in g: assert abs(g["p50"] - R[key]["p50"]) < 1e-12, (root, key)
    c = R["c40"]
    if c["n"] >= 4 * MIN_N:
        R["quart"], R["quart_common"] = group_stats(c["t"], c["e"], c["p"], quartiles(c["e"]), B)
    return R


KP_REF = 16.01


def assign(R):
    """(config, level, range) markers and the unit: 'other config' if the config differs, else the level marker."""
    G = R.get("G")
    if G is None: cfg = "?"
    else:
        kp = G.get("kp_eff", {}).get("slope", float("nan"))
        cfg = "same" if abs(kp / KP_REF - 1) < 0.02 and abs(R["p_hi"]) < 3 else "other"
    c = R.get("c40", {}); lvl = c.get("p50")
    if lvl is None or not np.isfinite(lvl): lv = "?"
    else:
        d = {k: abs(math.log(lvl / v)) for k, v in REF.items()}; k = min(d, key=d.get)
        lv = (f"{k}-like" if d[k] <= math.log(LEVEL_FACTOR) else "other") + ("!" if c["n"] < MIN_N else "")
    lo = R.get("range_open")
    rg = "?" if lo is None else min(REF_OPEN_END, key=lambda k: abs(lo - REF_OPEN_END[k])) + "-like"
    unit = "other config" if cfg == "other" else ("?" if cfg == "?" else lv)
    return cfg, lv, rg, unit


def halves(t, e):
    """Level in the first and second half of a group's episodes (split at the median episode index of its frames' episodes)."""
    ue = np.unique(e)
    if len(ue) < 4: return None
    cut = np.median(ue); a, b = e <= cut, e > cut
    return [(float(np.median(t[m])), int(m.sum()), int(len(np.unique(e[m])))) for m in (a, b)]


# ------------------------------------------------------------------------------------------------------------ output
def f(x, d=3, n=None):
    if x is None or (isinstance(x, float) and not math.isfinite(x)): return "n/a"
    return f"{x:.{d}f}" + ("!" if n is not None and n < MIN_N else "")


def short(date): return date[:16].replace("T", " ") if date else "?"


def main():
    ap = argparse.ArgumentParser(description="F8 table: friction floor per dataset, upload batch and quartile.")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--hf-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fetch", action="store_true", help="fetch missing Hugging Face JSON (huggingface.co only)")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.hf_cache, exist_ok=True)
    rows = []
    for root in a.roots:
        base = os.path.basename(os.path.normpath(root)); repo = base.replace("__", "/", 1)
        print(f"{base} ...", file=sys.stderr)
        R = analyse(root, a.boot); R["name"] = repo.split("/")[1]; R["hf"] = hf_dates(repo, a.hf_cache, a.fetch)
        ij = os.path.join(root, "meta/info.json")
        R["info"] = json.load(open(ij)) if os.path.exists(ij) else {}
        sj = os.path.join(root, "meta/stats.json")
        if os.path.exists(sj):
            st = json.load(open(sj)).get("observation.state", {})
            if "min" in st and len(st["min"]) > 21: R["stats_range"] = (st["min"][21], st["max"][21])
        R["range_open"] = R["G"]["range"][0] if "G" in R else (R["stats_range"][0] if "stats_range" in R else None)
        if "batches" in R["hf"] and "c40" in R and len(R["hf"]["batches"]) > 1:
            lab = np.full(len(R["c40"]["e"]), -1)
            for i, b in enumerate(R["hf"]["batches"]): lab[(R["c40"]["e"] >= b["ep0"]) & (R["c40"]["e"] <= b["ep1"])] = i
            k = lab >= 0
            R["batch"], R["batch_common"] = group_stats(R["c40"]["t"][k], R["c40"]["e"][k], R["c40"]["p"][k], lab[k], a.boot)
            cl = R["call"]; lab2 = np.full(len(cl["e"]), -1)
            for i, b in enumerate(R["hf"]["batches"]): lab2[(cl["e"] >= b["ep0"]) & (cl["e"] <= b["ep1"])] = i
            k2 = lab2 >= 0
            R["batch_all"], R["batch_all_common"] = group_stats(cl["t"][k2], cl["e"][k2], cl["p"][k2], lab2[k2], a.boot)
        rows.append(R)
    rows.sort(key=lambda R: R["hf"]["first"] or "")
    write(rows, a)


def batch_rows(R, whole=False, speed=False):
    """[(label, |tau|, episode, position, batch)] of the 40-100 band frames for each upload batch. With whole=True a dataset
    without any known batch (no info.json at any revision) is returned as one group labelled '(whole)', dated by its
    lastModified. With speed=True |v| is appended to each tuple."""
    out = []
    c = R.get("c40")
    if c is None: return out
    bt = R["hf"].get("batches", [])
    if not bt and whole and c["n"]:
        bt = [{"date": R["hf"].get("lastModified"), "sha": "-", "ep0": int(c["e"].min()), "ep1": int(c["e"].max()), "whole": True}]
    for i, b in enumerate(bt):
        m = (c["e"] >= b["ep0"]) & (c["e"] <= b["ep1"])
        lab = f"{R['name']} " + ("(whole)" if b.get("whole") else f"b{i + 1}")
        out.append((lab, c["t"][m], c["e"][m], c["p"][m], b) + ((c["v"][m],) if speed else ()))
    return out


ADJ_BINS = np.arange(-60, 10, 5)


def angle_profile(R):
    """Per-5-deg-bin median |tau| of a dataset's 40-100 band (bins with < 5 frames filled from the nearest filled bin)."""
    c = R["c40"]; k = np.digitize(c["p"], ADJ_BINS)
    med = {i: float(np.median(c["t"][k == i])) for i in np.unique(k) if (k == i).sum() >= 5}
    filled = np.array(sorted(med))
    return {i: med[int(filled[np.argmin(np.abs(filled - i))])] for i in range(len(ADJ_BINS) + 1)}, float(np.median(c["t"]))


def adj_level(t, p, prof):
    """Angle-adjusted level: median of |tau| / reference bin median, times the reference p50 (the level this group would
    show with the reference dataset's jaw-angle mix, if only the level and not the angle profile differs)."""
    if not len(t): return float("nan")
    m, ref50 = prof; k = np.digitize(p, ADJ_BINS)
    return float(np.median(t / np.array([m[i] for i in k])) * ref50)


def subset(V, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in V.items()}


def card(W):
    """d2_analyze.generic() on a frame subset (its own range, so the mid-range masks are those of the subset), plus the
    moving 40-100 deg/s frames of both directions from the verbatim copy of its mask (checked against generic())."""
    out = {}
    with contextlib.redirect_stdout(io.StringIO()): d2.generic(W, out)
    G = out["generic"]; mv = {}
    for d, cl in (("closing", True), ("opening", False)):
        m = band_mask(W, 40, 100, closing=cl)
        assert int(m.sum()) == G["moving"][f"{d}/40"]["n"], (d, int(m.sum()), G["moving"][f"{d}/40"]["n"])
        mv[d] = (np.abs(W["tau"][m]), W["ep"][m], W["pos"][m])
    for d, m in push_mask(W).items():
        g = G["push_still"][f"all/{d}"]; x = np.abs(W["tau"][m])
        assert int(m.sum()) == g["n"] and (g["n"] == 0 or abs(float(np.median(x)) - g["p50"]) < 1e-12), ("push", d)
        mv["push-" + d] = (x, W["ep"][m], W["pos"][m])
    return G, mv


def session_card(V, ep0, ep1):
    """The unit card of one upload batch: card() on the frames of episodes ep0..ep1 only."""
    m = (V["ep"] >= ep0) & (V["ep"] <= ep1); W = subset(V, m)
    G, mv = card(W)
    return G, mv, W, int(m.sum()), int(len(np.unique(W["ep"])))


def push_bins_matched(cards, key):
    """Mean of per-5-deg-bin push-still p50 (bins with >= 15 frames, as generic() keeps them) over the bins every card has."""
    bs = [c["push_still"][key]["bins"] for c in cards]
    common = sorted(set.intersection(*[set(b) for b in bs]), key=float) if bs else []
    return [float(np.mean([b[k][1] for k in common])) if common else float("nan") for b in bs], common


def slope(t, v):
    """OLS slope of |tau| on |v| inside the band, Nm per 100 deg/s (nan if < 10 frames)."""
    if len(t) < 10: return float("nan")
    return float(np.polyfit(v, t, 1)[0] * 100)


def write(rows, a):
    L = []
    L.append("F8 table: gripper friction floor per gtgando OpenArm dataset (definitions: floor/d2_analyze.py generic(), "
             "unit_floor.py / within_stroke.py). Nm unless noted. '!' = rests on < 30 frames (n too small).")
    L.append("HF dates are upload times (UTC); a batch's episodes were recorded before its upload time. CI = 95 % interval of the "
             f"median, {a.boot} resamples of whole episodes. Quartiles: cut points at the 25/50/75th percentiles of the indices of "
             "the episodes that contribute >= 1 frame to the closing 40-100 deg/s band; reported only if the band has >= 120 frames.")
    L.append(f"Unit markers: config = kp_eff within 2 % of {KP_REF} and closed end p99.5 within 3 deg of 0; level = moving closing "
             f"40-100 dps p50 nearest of A {REF['A']:.3f} / B {REF['B']:.3f} Nm in log ratio, 'other' if more than x{LEVEL_FACTOR} "
             f"from both ('!' if the band has < 30 frames); range = nearer of the open ends A {REF_OPEN_END['A']} / B "
             f"{REF_OPEN_END['B']} deg (range minimum). unit = 'other config' if config differs, else the level marker.")
    # ---- table 1
    L.append("\nTable 1. one row per dataset")
    L.append(f"{'dataset':40s} {'HF first':16s} {'HF last':16s} {'ep':>4s} {'frames':>6s} {'range deg':>13s} {'p0.5/p99.5':>12s} "
             f"{'kp_eff':>6s} {'push-close p50/p90 (n)':>23s} {'push-open p50/p90 (n)':>24s} {'move-close 40-100 p50 [CI] (n fr/ep)':>38s} "
             f"{'move-close 15-40 p50 (n, >1.5)':>25s} {'s pooled>=3 (dof)':>18s} {'config':>6s} {'level':>8s} {'range':>7s}  unit")
    for R in rows:
        h = R["hf"]; inf = R["info"]
        ep_s = str(R.get("episodes", inf.get("total_episodes", "?"))); fr_s = str(R.get("frames", inf.get("total_frames", "?")))
        cfg, lv, rg, unit = assign(R)
        if "G" not in R:
            sr = R.get("stats_range"); rs = f"{sr[0]:.1f}..{sr[1]:.1f}*" if sr else "?"
            L.append(f"{R['name']:40s} {short(h['first']):16s} {short(h['last']):16s} {ep_s:>4s} {fr_s:>6s} {rs:>13s} "
                     f"  UNREADABLE data parquet (no footer; {R['error']}); range from meta/stats.json; range marker {rg}")
            continue
        G = R["G"]; ps = G["push_still"]; pc, po = ps["all/closing"], ps["all/opening"]
        c40, c15, w = R["c40"], R["c15"], G["within"]
        kp = G.get("kp_eff", {}).get("slope")
        ci = c40.get("ci", (float("nan"),) * 2)
        mc40 = f"{f(c40.get('p50'), 3, c40['n'])} [{f(ci[0])},{f(ci[1])}] ({c40['n']}/{c40['eps']})"
        L.append(f"{R['name']:40s} {short(h['first']):16s} {short(h['last']):16s} {ep_s:>4s} {fr_s:>6s} "
                 f"{G['range'][0]:6.1f}..{G['range'][1]:5.1f} {R['p_lo']:6.1f}/{R['p_hi']:5.1f} {f(kp, 2):>6s} "
                 f"{f(pc['p50'], 3, pc['n']) + '/' + f(pc['p90'], 3, pc['n']) + ' (' + str(pc['n']) + ')':>23s} "
                 f"{f(po['p50'], 3, po['n']) + '/' + f(po['p90'], 3, po['n']) + ' (' + str(po['n']) + ')':>24s} {mc40:>38s} "
                 f"{f(c15.get('p50'), 3, c15['n']) + ' (' + str(c15['n']) + (', >1.5 ' + format(float(np.mean(c15['t'] > 1.5)), '.2f') if c15['n'] else '') + ')':>25s} "
                 f"{f(w.get('pooled3'), 4, w['n']) + ' (' + str(w.get('dof3', 0)) + ')':>18s} {cfg:>6s} {lv:>8s} {rg:>7s}  {unit}"
                 + ("  [signs flipped]" if R.get("flipped") else ""))
    L.append("* range of an unreadable dataset is the min/max of the whole recording from meta/stats.json (no 0.5 % trim).")
    # ---- table 2: quartiles
    L.append("\nTable 2. drift within a recording: moving closing 40-100 dps |tau| p50 per episode-index quartile "
             "(n frames / episodes, episode range) [95 % CI]; bin-matched level over the common 5-deg bins")
    for R in rows:
        if "G" not in R: continue
        c = R["c40"]
        if "quart" not in R:
            L.append(f"  {R['name']:40s} not reported: band has {c['n']} frames (< 120 = 4 x 30)"); continue
        q = R["quart"]
        s = " -> ".join(f"{f(x['p50'], 3, x['n'])} ({x['n']}/{x['eps']}, ep {x['ep_range'][0]}-{x['ep_range'][1]}) [{f(x['ci'][0])},{f(x['ci'][1])}]" for x in q)
        L.append(f"  {R['name']:40s} {s}")
        L.append(f"  {'':40s} Q4/Q1 = {q[-1]['p50'] / q[0]['p50']:.3f}; bin-matched " + " -> ".join(f(x['binm']) for x in q)
                 + f" over bins {R['quart_common']}")
    # ---- table 3: upload batches
    L.append("\nTable 3. sessions: every upload batch (episodes added by one 'Upload folder' commit; a dataset with one commit is one "
             "batch). Moving closing |tau| p50 [95 % CI] (n frames / episodes) in the 40-100 dps band and over all closing speeds "
             ">= 15 dps; bin-matched level (datasets with >= 2 batches); first half -> second half of the batch's episodes (40-100)")
    for R in rows:
        if "G" not in R: continue
        bt = R["hf"].get("batches", [])
        L.append(f"  {R['name']}")
        for i, (lab, t, e, pp, b) in enumerate(batch_rows(R)):
            x = next((r for r in R.get("batch", []) if r["label"] == i), None)
            y = next((r for r in R.get("batch_all", []) if r["label"] == i), None)
            if x is None and len(t):                               # single-batch dataset: the dataset itself
                x = {"p50": float(np.median(t)), "ci": R["c40"].get("ci", (float("nan"),) * 2), "n": len(t), "eps": len(np.unique(e)), "binm": float("nan")}
            if y is None and len(bt) == 1 and R["call"]["n"]:
                cl = R["call"]; y = {"p50": cl["p50"], "ci": cl["ci"], "n": cl["n"], "eps": cl["eps"], "binm": float("nan")}
            sx = f"{f(x['p50'], 3, x['n'])} [{f(x['ci'][0])},{f(x['ci'][1])}] ({x['n']}/{x['eps']}) binm {f(x['binm'])}" if x else "no frames"
            sy = f"{f(y['p50'], 3, y['n'])} [{f(y['ci'][0])},{f(y['ci'][1])}] ({y['n']}/{y['eps']}) binm {f(y['binm'])}" if y else "no frames"
            hv = halves(t, e) if len(t) else None
            sh = " -> ".join(f"{f(v, 3, n)} ({n}/{k})" for v, n, k in hv) if hv else "-"
            L.append(f"    b{i + 1} uploaded {short(b['date'])} ({b['sha']}) ep {b['ep0']:3d}-{b['ep1']:3d}: 40-100 {sx:47s} | >=15 {sy:47s} | halves {sh}")
        if len(bt) > 1: L.append(f"    common bins: 40-100 {R.get('batch_common')}, >=15 {R.get('batch_all_common')}")
        if not bt and R["c40"]["n"]:                                # no upload batches known (upload-large-folder sets)
            c = R["c40"]; hv = halves(c["t"], c["e"])
            L.append(f"    (no info.json at any revision: whole recording) 40-100 {f(c['p50'], 3, c['n'])} ({c['n']}/{c['eps']}) | halves "
                     + (" -> ".join(f"{f(v, 3, n)} ({n}/{k})" for v, n, k in hv) if hv else "-"))
    # ---- table 4: HF dates in full
    L.append("\nTable 4. Hugging Face metadata (UTC): createdAt, lastModified, commits, upload batches (episodes at that revision)")
    for R in rows:
        h = R["hf"]; inf = R["info"]
        bs = "; ".join(f"{short(b['date'])} ep {b['ep0']}-{b['ep1']}" for b in h.get("batches", []))
        L.append(f"  {R['name']:40s} created {short(h['createdAt'])}  lastModified {short(h['lastModified'])}  commits {h['n_commits']:2d}  "
                 f"info.json: episodes {inf.get('total_episodes', '-')}, frames {inf.get('total_frames', '-')}, fps {inf.get('fps', '-')}, "
                 f"version {inf.get('codebase_version', '-')}; frames read {R.get('frames', 0)}; batches: {bs or '-'}"
                 + (f"; load: {R['log']}" if R.get("log") else ""))
    # ---- table 5: band detail per 5-deg bin
    ref = next((R for R in rows if R["name"] == "openarm-pick-place-100ep-v2" and "G" in R), None)
    prof = angle_profile(ref) if ref is not None else None
    adj = (lambda t, p: adj_level(t, p, prof)) if prof else (lambda t, p: float("nan"))
    bins = np.arange(-45, 0, 5)
    L.append("\nTable 5. moving closing 40-100 dps band per dataset and batch: p10 / p50 / p90, share of frames with |tau| > 1.5 Nm "
             "(in contact), median |v| (deg/s), dt/dv = OLS slope of |tau| on |v| inside the band (Nm per 100 deg/s), the angle-adjusted level (each frame divided by the 100ep-v2 band's median in its 5-deg bin, "
             "median ratio times 100ep-v2's p50; removes differences in which jaw angles were visited), and the median per 5-deg bin of jaw "
             "angle (n); bins with < 5 frames left blank")
    L.append(f"  {'group':48s} {'n':>4s} {'p10':>6s} {'p50':>6s} {'p90':>6s} {'>1.5':>5s} {'v50':>5s} {'dt/dv':>6s} {'adj':>6s} " + " ".join(f"{'[' + str(b) + ',' + str(b + 5) + ')':>14s}" for b in bins))
    for R in rows:
        if "G" not in R or R["c40"]["n"] == 0: continue
        c = R["c40"]; groups = [(R["name"], c["t"], c["e"], c["p"])]
        groups = [groups[0] + (c["v"],)]
        if len(R["hf"].get("batches", [])) > 1: groups += [(lab, t, e, pp, v) for lab, t, e, pp, b, v in batch_rows(R, speed=True)]
        for lab, t, e, pp, v in groups:
            if not len(t): continue
            cells = []
            for b in bins:
                k = (pp >= b) & (pp < b + 5)
                cells.append(f"{np.median(t[k]):.3f} ({k.sum():3d})" if k.sum() >= 5 else "")
            L.append(f"  {lab:48s} {len(t):4d} {np.percentile(t, 10):6.3f} {np.median(t):6.3f} {np.percentile(t, 90):6.3f} "
                     f"{np.mean(t > 1.5):5.2f} {np.median(v):5.1f} {slope(t, v):6.3f} {adj(t, pp):6.3f} " + " ".join(f"{x:>14s}" for x in cells))
    # ---- table 6: spread summary per unit
    L.append("\nTable 6. spread of the moving closing 40-100 dps level within each unit (readable datasets with config 'same'; "
             "unit = level marker). max/min and CV over (a) datasets with >= 30 frames, (b) sessions = upload batches with >= 30 "
             "frames (single-batch datasets count as one session; a dataset with no known batches counts as one session '(whole)', dated by "
             "lastModified), (c) as (b) but including thin cells. 'adj' = the same spread of the angle-adjusted levels (Table 5)")
    for u in ("A", "B"):
        ds, ses, ses_all = [], [], []
        for R in rows:
            if "G" not in R: continue
            cfg, lv, rg, unit = assign(R)
            if cfg != "same" or not lv.startswith(f"{u}-like"): continue
            c = R["c40"]
            if c["n"] >= MIN_N: ds.append((R["name"], c["p50"], c["n"], None, adj(c["t"], c["p"])))
            for lab, t, e, pp, b in batch_rows(R, whole=True):
                if len(t): ses_all.append((lab, float(np.median(t)), len(t), b["date"], adj(t, pp)))
                if len(t) >= MIN_N: ses.append((lab, float(np.median(t)), len(t), b["date"], adj(t, pp)))
        for tag, lst in (("(a) datasets", ds), ("(b) sessions n>=30", ses), ("(c) sessions all", ses_all)):
            if not lst: L.append(f"  unit {u} {tag:20s} none"); continue
            for nm, col in (("p50", 1), ("adj", 4)):
                v = np.array([x[col] for x in lst])
                cv = float(np.std(v, ddof=1) / np.mean(v)) if len(v) > 1 else float("nan")
                L.append(f"  unit {u} {tag:20s} {nm:3s} k {len(v):2d}  min {v.min():.3f}  max {v.max():.3f}  max/min {v.max() / v.min():.3f}  "
                         f"mean {v.mean():.3f}  CV {cv:.3f}  half-range/mean {(v.max() - v.min()) / 2 / v.mean():.3f}")
            L.append("      " + "; ".join(f"{x[0]} {x[1]:.3f}/adj {x[4]:.3f} (n {x[2]}" + (f", up {short(x[3])[:10]}" if x[3] else "") + ")" for x in lst))
    # ---- table 7: the full card per session
    L.append("\nTable 7. the unit card of each upload batch (d2_analyze.generic() run on that batch's episodes only), datasets with >= 2 "
             "batches: range, kp_eff, push-still closing and opening p50/p90 (n), moving closing and opening 40-100 dps p50 (n), "
             "within-stroke s pooled >= 3 (dof). Does the opening direction and the still push move with the closing level?")
    L.append(f"  {'session':34s} {'uploaded':16s} {'fr':>6s} {'ep':>3s} {'range deg':>13s} {'kp':>6s} {'push-close p50/p90 (n)':>24s} "
             f"{'push-open p50/p90 (n)':>24s} {'move-close40 (n)':>17s} {'move-open40 (n)':>17s} {'s>=3 (dof)':>14s}")
    extra = []
    for R in rows:
        if "G" not in R or len(R["hf"].get("batches", [])) < 2: continue
        cards = []
        for i, b in enumerate(R["hf"]["batches"]):
            G, mv, W, nf, ne = session_card(R["V"], b["ep0"], b["ep1"]); cards.append((G, mv, W, b))
            pc, po = G["push_still"]["all/closing"], G["push_still"]["all/opening"]
            mc, mo, w = G["moving"]["closing/40"], G["moving"]["opening/40"], G["within"]
            L.append(f"  {R['name'] + ' b' + str(i + 1):34s} {short(b['date']):16s} {nf:6d} {ne:3d} {G['range'][0]:6.1f}..{G['range'][1]:5.1f} "
                     f"{f(G['kp_eff'].get('slope'), 2):>6s} "
                     f"{f(pc['p50'], 3, pc['n']) + '/' + f(pc['p90'], 3, pc['n']) + ' (' + str(pc['n']) + ')':>24s} "
                     f"{f(po['p50'], 3, po['n']) + '/' + f(po['p90'], 3, po['n']) + ' (' + str(po['n']) + ')':>24s} "
                     f"{f(mc.get('p50'), 3, mc['n']) + ' (' + str(mc['n']) + ')':>17s} {f(mo.get('p50'), 3, mo['n']) + ' (' + str(mo['n']) + ')':>17s} "
                     f"{f(w.get('pooled3'), 4, w['n']) + ' (' + str(w.get('dof3', 0)) + ')':>14s}")
        extra.append((R, cards))
    # ---- table 8: every card quantity per session, jaw-angle matched, with a split-half noise scale
    L.append("\nTable 8. same sessions, every card quantity, p50 [95 % episode CI] (n) and a jaw-angle-matched level 'binm' (mean of "
             "per-5-deg-bin medians, fixed 5-deg grid, over the bins every batch of the dataset fills: >= 15 frames for push-still as in "
             "generic(), >= 5 for the moving bands). Needed because the opening direction rises toward the open end (floor_ja.md F1) "
             "and the sessions reach different open ends. Split-half: p50 (n) in the odd and the even episodes of the batch, each "
             "half run through the same definitions (a within-session noise scale).")
    Q = (("push-closing", "push-close", 15), ("push-opening", "push-open", 15), ("closing", "move-close40", 5), ("opening", "move-open40", 5))
    for R, cards in extra:
        bm = {}
        for key, nm, mn in Q:
            t = np.concatenate([c[1][key][0] for c in cards]); pp = np.concatenate([c[1][key][2] for c in cards])
            lab = np.concatenate([np.full(len(c[1][key][0]), i) for i, c in enumerate(cards)])
            bm[key] = bin_matched(t, pp, lab, mn) if len(t) else ({}, [])
        L.append(f"  {R['name']}: common bins " + ", ".join(f"{nm} {bm[key][1]}" for key, nm, mn in Q))
        for i, (G, mv, W, b) in enumerate(cards):
            cells = []
            for key, nm, mn in Q:
                t, e, pp = mv[key]
                ci = boot_ci(t, e, a.boot) if len(t) else (float("nan"),) * 2
                bci = binm_ci(t, e, pp, bm[key][1], a.boot) if len(t) else (float("nan"),) * 2
                cells.append(f"{nm} {f(float(np.median(t)) if len(t) else None, 3, len(t))} [{f(ci[0])},{f(ci[1])}] ({len(t)}) "
                             f"binm {f(bm[key][0].get(i, float('nan')))} [{f(bci[0])},{f(bci[1])}]")
            L.append(f"    b{i + 1} {short(b['date'])}: " + "  ".join(cells))
            pb = []
            for key, nm, mn in Q:
                t, e, pp = mv[key]; bb = np.floor(pp / 5) * 5
                pb.append(f"{nm} " + (" ".join(f"{c:g}:{np.median(t[bb == c]):.3f}({int(np.sum(bb == c))})" for c in bm[key][1]) or "-"))
            L.append("       per common bin: " + "  ".join(pb))
            half = []
            for par in (1, 0):
                Gh, mvh = card(subset(W, W["ep"] % 2 == par))
                half.append(" ".join(f"{nm} {f(float(np.median(mvh[key][0])) if len(mvh[key][0]) else None, 3, len(mvh[key][0]))} "
                                     f"({len(mvh[key][0])})" for key, nm, mn in Q))
            L.append(f"       split-half odd: {half[0]}")
            L.append(f"       split-half even: {half[1]}")
    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh: fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
