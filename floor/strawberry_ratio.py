"""Camera vs encoder after landing, on the strawberry dataset (F7, pre-registered in notes/floor_ja.md).

The left wrist camera is fixed to the gripper body, so where a pad appears in the image depends only on the finger
angle. After the pads land on an object the gripper motor keeps advancing; the ratio
    r = (pad travel seen by the camera) / (pad travel the encoder predicts through the camera's px/deg calibration)
is ~1 if the pads really move into the object and ~0 if the advance is wind-up between motor and pad (F5: 0.11 on a
wood cube). This script measures r on xhaka3456/openarm_pick_strawberry_and_place_on_white_plate2 (and on v1).

Dataset facts this script checks and relies on (printed in section (0)):
  * the gripper column of observation.state equals the gripper column of action in every frame; gripper velocity and
    effort are all zero; the value is quantized at (25/65535 rad) * (0.044/1.0472 m/rad), i.e. a Damiao 16-bit motor
    position converted with openarm_hardware's 0.044 m <-> 1.0472 rad gripper mapping. We convert it back to motor
    degrees, 0 = closed and open negative:  q_deg = -value / (0.044/1.0472) * 180/pi.
  * there is no separate gripper command, so the pre-registered landing rule (lag = q_cmd - q) cannot be evaluated;
    landing A is where the encoder's fast closing step ends (speed below 40 deg/s, F5's threshold, for 2 frames:
    two successive steps < 1.333 deg/frame), B5 the start of the first 5-frame encoder plateau (|dq| < 0.05 deg),
    Bend 2 frames before the hold ends (release, or a mid-hold jump > 1 deg/frame), S = A+5 -> Bend.

Pad position in the image, two measures per pad:
  * holes (primary): the dark through-holes in each pale pad (one large hole on the left pad, two on the right pad),
    found as compact blobs darker than DARK x the local mean brightness whose ring has pad colour; sub-pixel by a
    contrast-weighted centroid. They are rigid on the pad and do not depend on what lies next to the pad.
  * edges (secondary): the pad's inner boundary on each contact row, scanning from inside the pad (7 px past the
    holes) toward the image centre until the colour leaves the pad's own colour (the median colour of the hole's ring,
    so it follows the lighting) for 3 px; sub-pixel at the midpoint crossing. Each row records what lies beyond the
    boundary: 1 bright background, 2 red, 3 green (leaves), 4 dark, 0 other.
Blobs are assigned to the three hole slots by nearest distance to a rough q-based prior (PRIOR, read off ep 0 by
eye, +-16 px); analyse() refits the prior from the data and re-assigns with +-6 px.

Analysis (analyse()):
  * timing: at each closing step, the sub-frame time where each hole crosses the middle of its travel, minus the
    encoder's; the median over all holes and grasps sets the integer camera shift s (camera index = encoder
    index + s). The camera LEADS the recorded q (median -1.89 frames on v2, -1.71 on v1), the opposite sign to F5.
    Frames are decoded sequentially from a quarter frame before the episode's from_timestamp (see extract()).
  * calibration: (i) still open-end clusters (spans only ~2.4 deg); (ii) hold angles: across holds, encoder q at
    A+5 against the aligned camera x (q -22..-14 deg). (ii) is the primary px/deg because r is measured at the
    hold angle; (i) is reported alongside. Both are loaded/unloaded mixtures; see section (4a)(iii).
  * r of the gap (Ru, or Rl if Ru is hidden, minus L) for A->B5 (pre-registered), A->Bend and S, at camera
    shifts -2..+2, at each grasp's own lag, and on a fractional grid; rule 4 excludes grasps whose predicted gap
    change is < 3x the single-edge hold jitter. Within-hold regression of gap on q is a fourth estimator.

usage: python3 -I strawberry_ratio.py ROOT CAMKEY [--episodes 0-39] [--rows 230-335] [--dark 0.55] [--save x.npz]
           [--load x.npz] [--out DIR] [--tag v2] [--col 14] [--fig x.png] [--montage 0,19,27,33] [--cube padcal.npz]
  ROOT is a LeRobot v3 dataset directory, CAMKEY e.g. observation.images.left_wrist. --load skips decoding.
  Outputs (npz, per-episode CSV, montages, figure) go to --out (default: the current directory).
  --cube runs section (5), the post hoc control on F5's wood cube (padcal.npz written by floor/padcal.py).
"""
import glob, json, os, subprocess, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa
from scipy import ndimage as ndi

M_PER_RAD = 0.044 / 1.0472           # openarm_hardware gripper mapping (v1.0 constants)
LSB_RAD = 25.0 / 65535.0             # Damiao 16-bit position over [-12.5, 12.5] rad
HOLE_Y0, HOLE_Y1 = 215, 360
WIN = {0: (130, 320), 1: (270, 480)}  # x window for the left (0) and right (1) pad
# hole slots: (pad, x at q = -35.6 deg, px/deg, y). L-big, R-upper, R-lower. Rough, for association only.
PRIOR = [(0, 186.1, 2.90, 259.0), (1, 400.5, -2.75, 246.0), (1, 394.0, -2.55, 289.0)]
NCAND = 8
NB_NAMES = ["other", "bg", "red", "green", "dark"]


def opt(k, d):
    return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def to_deg(v):
    return -np.degrees(np.asarray(v, float) / M_PER_RAD)


# ----------------------------------------------------------------------------------------------- per-frame features
def find_holes(frame, dark_frac):
    """Compact dark blobs with a pad-coloured ring in each pad window: list of (pad, cx, cy, area, contrast, ringRGB)."""
    res = []
    for s, (x0, x1) in WIN.items():
        sub = frame[HOLE_Y0:HOLE_Y1, x0:x1].astype(np.float32); L = sub.mean(2)
        loc = ndi.uniform_filter(L, 25)
        dark = L < dark_frac * loc
        lab, n = ndi.label(dark)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None: continue
            m = lab[sl] == i; a = int(m.sum()); h, w = m.shape
            if a < 12 or a > 260 or w < 3 or h < 4 or w > 22 or h > 30 or a / (h * w) < 0.45: continue
            if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == L.shape[0] or sl[1].stop == L.shape[1]: continue
            ys = slice(max(sl[0].start - 3, 0), min(sl[0].stop + 3, L.shape[0])); xs = slice(max(sl[1].start - 3, 0), min(sl[1].stop + 3, L.shape[1]))
            mm = lab[ys, xs] == i
            ring = ndi.binary_dilation(mm, iterations=3) & ~ndi.binary_dilation(mm, iterations=1)
            rr = sub[ys, xs][ring]
            if len(rr) < 10: continue
            rgb = np.median(rr, 0); Lr = float(np.median(L[ys, xs][ring])); con = Lr - float(L[ys, xs][mm].mean())
            if con < 18 or not (5 <= rgb[0] - rgb[1] <= 65 and rgb[2] - rgb[1] >= -3 and rgb[1] >= 12): continue
            if dark[ys, xs][ring].mean() > 0.15: continue
            w8 = np.clip(Lr - L[ys, xs], 0, None) * ndi.binary_dilation(mm, iterations=2)
            yy, xx = np.mgrid[ys, xs]
            res.append((s, float((w8 * xx).sum() / w8.sum() + x0), float((w8 * yy).sum() / w8.sum() + HOLE_Y0), a, con, rgb))
    return res


def assign(cands, xp, yp, wx, wy):
    """cands (2, NCAND, 4+) per pad: cx, cy, area, contrast. xp/yp: predicted (3,) slot positions. Returns (3,4), index."""
    out = np.full((3, 4), np.nan, np.float32); which = np.full(3, -1)
    for j, (s, _, _, _) in enumerate(PRIOR):
        c = cands[s]; ok = np.isfinite(c[:, 0]) & (np.abs(c[:, 0] - xp[j]) <= wx) & (np.abs(c[:, 1] - yp[j]) <= wy)
        if ok.any():
            i = np.nonzero(ok)[0][np.argmin(np.abs(c[ok, 0] - xp[j]) + 0.5 * np.abs(c[ok, 1] - yp[j]))]
            out[j] = c[i, :4]; which[j] = i
    return out, which


def prior_xy(q):
    xp = np.array([x0 + sl * (q + 35.6) for _, x0, sl, _ in PRIOR]); yp = np.array([y for *_, y in PRIOR])
    return xp, yp


def pad_edges(frame, slots, rings, r0, r1, xc):
    """Inner boundary x of each pad on rows r0..r1 (nan if not found) and the class of what lies beyond it."""
    band = frame[r0:r1 + 1].astype(np.float32); nr, W = band.shape[:2]
    ex = np.full((2, nr), np.nan, np.float32); nb = np.zeros((2, nr), np.int8)
    for s in (0, 1):                                     # left pad: slot 0, scan +x; right pad: slots 1/2, scan -x
        js = (0,) if s == 0 else (1, 2)
        anchors = [slots[j, 0] for j in js if np.isfinite(slots[j, 0])]
        ring = next((rings[j] for j in js if rings[j] is not None), None)
        if not anchors or ring is None: continue
        xa = int(round(max(anchors) + 7)) if s == 0 else int(round(min(anchors) - 7))
        if s == 0 and not (0 < xa < xc - 8): continue
        if s == 1 and not (xc + 8 < xa < W - 1): continue
        sub = band[:, xa:xc] if s == 0 else band[:, xc:xa + 1][:, ::-1]   # scan direction always increasing index
        ref = np.asarray(ring, np.float32)
        d = np.sqrt(((sub - ref) ** 2).sum(2)) / (ref.mean() + 20.0)
        Ls = sub.mean(2); R, G, B = sub[..., 0], sub[..., 1], sub[..., 2]
        cls = np.zeros(Ls.shape, np.int8)
        cls[(Ls > 150) & (R - G < 25)] = 1; cls[(R - G > 50) & (Ls <= 200)] = 2; cls[(G > R + 3) & (G > B)] = 3; cls[Ls < 40] = 4
        hi = d > 0.45
        run = hi[:, :-2] & hi[:, 1:-1] & hi[:, 2:]
        ok = run.any(1) & (d[:, 0] < 0.35)                 # the anchor itself must be pad-coloured
        i0 = np.argmax(run, 1)
        for i in np.nonzero(ok)[0]:
            b = i0[i]
            if b < 2 or b + 5 > d.shape[1]: continue
            T = 0.5 * (np.median(d[i, max(b - 4, 0):b]) + np.median(d[i, b:b + 4]))
            xx = None
            for t in range(b - 2, b + 3):
                a1, a2 = d[i, t], d[i, t + 1]
                if (a1 - T) * (a2 - T) <= 0 and a1 != a2:
                    xx = t + (T - a1) / (a2 - a1); break
            if xx is None: xx = b - 0.5
            ex[s, i] = xa + xx if s == 0 else xa - xx
            cs = cls[i, b + 1:b + 6]
            if len(cs): nb[s, i] = np.bincount(cs, minlength=5).argmax()
    return ex, nb


def frame_features(fr, q, dark_frac, r0, r1, xc):
    hl = find_holes(fr, dark_frac)
    cands = np.full((2, NCAND, 4), np.nan, np.float32); rings_all = [[None] * NCAND, [None] * NCAND]
    for s in (0, 1):
        hs = sorted([h for h in hl if h[0] == s], key=lambda t: -t[3])[:NCAND]
        for i, h in enumerate(hs): cands[s, i] = h[1:5]; rings_all[s][i] = h[5]
    xp, yp = prior_xy(q)
    slots, which = assign(cands, xp, yp, 16.0, 14.0)
    rings = [rings_all[PRIOR[j][0]][which[j]] if which[j] >= 0 else None for j in range(3)]
    ex, nb = pad_edges(fr, slots, rings, r0, r1, xc)
    c = fr[240:331, 250:346].astype(np.int16)
    red = float(((c[..., 0] - c[..., 1] > 50) & (c[..., 0] > 100)).mean())
    return cands, slots, ex, nb, red


# ------------------------------------------------------------------------------------------------------- data load
def load_state(root):
    tab = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))])
    names = tab.column_names
    EP = np.array(tab.column("episode_index").to_pylist()); FI = np.array(tab.column("frame_index").to_pylist())
    S = np.array(tab.column("observation.state").to_pylist(), float); A = np.array(tab.column("action").to_pylist(), float)
    V = np.array(tab.column("observation.velocity").to_pylist(), float) if "observation.velocity" in names else None
    E = np.array(tab.column("observation.effort").to_pylist(), float) if "observation.effort" in names else None
    return EP, FI, S, A, V, E


def data_facts(S, A, V, E, col):
    print("(0) data facts")
    g, c = S[:, col], A[:, col]
    print(f"   gripper col {col}: state == action in {np.mean(g == c) * 100:.2f} % of {len(g)} frames (max |diff| {np.abs(g - c).max():.3g})")
    if V is not None: print(f"   gripper velocity col {col}: max |v| {np.abs(V[:, col]).max():.3g}")
    if E is not None:
        print(f"   effort: all-zero columns {[j for j in range(E.shape[1]) if not E[:, j].any()]}; "
              f"max |effort| over the other columns {np.abs(E).max():.3g}")
    u = np.unique(g); du = np.diff(u); step = np.min(du[du > 1e-9])
    print(f"   gripper value range {g.min():.5f}..{g.max():.5f}; smallest step {step:.6g} = {step / (LSB_RAD * M_PER_RAD):.4f} x (25/65535 rad * 0.044/1.0472 m/rad)")
    x = (g / M_PER_RAD) / LSB_RAD
    print(f"   value/(m/rad)/LSB fractional part: median {np.median(x - np.floor(x)):.4f}, IQR {np.percentile(x - np.floor(x), 25):.4f}..{np.percentile(x - np.floor(x), 75):.4f}")


# --------------------------------------------------------------------------------------------------- extraction
def extract(root, cam, eps, r0, r1, xc, dark_frac, EP, S, col):
    meta = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(root, "meta/episodes/chunk-*/file-*.parquet")))])
    MEP = np.array(meta.column("episode_index").to_pylist())
    info = json.load(open(os.path.join(root, "meta/info.json"))); H, W = info["features"][cam]["shape"][:2]
    rec = {k: [] for k in ("ep", "k", "q", "cands", "slots", "ex", "nb", "red")}
    for e in eps:
        idx = np.nonzero(EP == e)[0]; mi = np.nonzero(MEP == e)[0]
        if len(idx) == 0 or len(mi) == 0: continue
        seg = {c.split("/")[-1]: meta.column(c)[mi[0]].as_py() for c in meta.column_names if c.startswith(f"videos/{cam}/")}
        vf = os.path.join(root, f"videos/{cam}/chunk-{seg.get('chunk_index', 0):03d}/file-{seg.get('file_index', 0):03d}.mp4")
        if not os.path.exists(vf): print(f"  ep {e}: video missing"); continue
        n = len(idx); qd = to_deg(S[idx, col])
        # Seek a quarter frame early: "%.4f" can round the start a few microseconds past the first frame's pts
        # (e.g. 1217/30 -> 40.5667), and ffmpeg's accurate seek then drops that frame, shifting the whole episode by one.
        ss = max(0.0, seg["from_timestamp"] - 0.25 / float(info.get("fps", 30)))
        p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-ss", f"{ss:.4f}", "-i", vf, "-frames:v", str(n),
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        k = 0; nbyte = H * W * 3
        while k < n:
            buf = p.stdout.read(nbyte)
            if len(buf) < nbyte: break
            fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            cands, slots, ex, nb, red = frame_features(fr, qd[k], dark_frac, r0, r1, xc)
            for key, v in (("ep", e), ("k", k), ("q", qd[k]), ("cands", cands), ("slots", slots), ("ex", ex), ("nb", nb), ("red", red)):
                rec[key].append(v)
            k += 1
        p.stdout.close(); p.wait()
        print(f"  ep {e}: {k}/{n} frames decoded", flush=True)
    return {k: np.array(v) for k, v in rec.items()}


# ------------------------------------------------------------------------------------------------------- analysis
K_G = 140.0                       # Nm/rad, gripper wind-up stiffness of arm A (F5), assumed for this arm
R_C = (0.060, 0.097)              # m, contact lever range (F7)
SLOT_NAMES = ["L", "Ru", "Rl"]


def robust_poly(x, y, deg, it=4):
    keep = np.isfinite(x) & np.isfinite(y)
    for _ in range(it):
        p = np.polyfit(x[keep], y[keep], deg); res = y - np.polyval(p, x)
        mad = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.isfinite(res) & (np.abs(res) < max(4 * mad, 1.5))
    return p, keep


def reassign(D, still, wx=6.0, wy=8.0, dset=(-1, 0, 1, 2, 3)):
    """Second pass: each slot's x(q), y(q) as medians of the first-pass positions over still frames in 0.5-deg bins
    (>= 5 frames, linear interpolation between bins); re-assign candidates within +-wx/wy. The camera leads the
    recorded q by ~1-2 frames (section (3)), so for a frame k the x window spans the predictions at q[k+d], d in dset;
    for still frames this is the same single prediction."""
    ep = D["ep"]; q = D["q"].astype(float); sl = D["slots"].astype(float); c = D["cands"].astype(float); N = len(q)
    fits = []; out = np.full((N, 3, 4), np.nan)
    # q[k+d] within the same episode (clamped at the episode ends)
    qd = []
    for d in dset:
        j = np.clip(np.arange(N) + d, 0, N - 1); j = np.where(ep[j] == ep, j, np.arange(N)); qd.append(q[j])
    qd = np.array(qd)
    for j, (s, *_) in enumerate(PRIOR):
        m = still & np.isfinite(sl[:, j, 0])
        edges = np.arange(-40.0, 1.5, 0.5); cen, mx, my = [], [], []
        for lo in edges:
            mm = m & (q >= lo) & (q < lo + 0.5)
            if mm.sum() >= 5: cen.append(np.median(q[mm])); mx.append(np.median(sl[mm, j, 0])); my.append(np.median(sl[mm, j, 1]))
        cen, mx, my = np.array(cen), np.array(mx), np.array(my)
        fits.append((cen, mx, my, m.sum()))
        xps = np.interp(qd, cen, mx); yps = np.interp(qd, cen, my)          # (len(dset), N)
        xlo, xhi = xps.min(0) - wx, xps.max(0) + wx; ylo, yhi = yps.min(0) - wy, yps.max(0) + wy
        xc_ = np.median(xps, 0); yc_ = np.median(yps, 0)
        cc = c[:, s]                                                  # (N, NCAND, 4)
        ok = (cc[..., 0] >= xlo[:, None]) & (cc[..., 0] <= xhi[:, None]) & (cc[..., 1] >= ylo[:, None]) & (cc[..., 1] <= yhi[:, None])
        sc = np.where(ok, np.abs(cc[..., 0] - xc_[:, None]) + 0.5 * np.abs(cc[..., 1] - yc_[:, None]), np.inf)
        i = np.argmin(sc, 1); okk = np.isfinite(sc[np.arange(N), i])
        out[okk, j] = cc[np.nonzero(okk)[0], i[okk]]
    return out, fits


def segment(qe):
    """Closing steps from the open state in one episode's q (deg). Each is one grasp event; see the module docstring."""
    n = len(qe); out = []; t = 1
    while t < n - 10:
        if qe[t - 1] < -28 and qe[min(t + 4, n - 1)] - qe[t - 1] > 5 and abs(qe[t - 1] - qe[max(t - 3, 0)]) < 0.2:
            s0 = t - 1
            A = next((u for u in range(s0 + 1, min(s0 + 15, n - 2)) if qe[u] > qe[s0] + 5
                      and abs(qe[u + 1] - qe[u]) < 1.333 and abs(qe[u + 2] - qe[u + 1]) < 1.333), None)
            if A is None: t += 1; continue
            rel = next((u for u in range(A + 1, n) if qe[u] < np.max(qe[A:u + 1]) - 3), n)
            o = rel
            while o - 1 > A and qe[o - 1] < qe[o - 2] - 0.02: o -= 1      # walk back to the start of the opening
            hend = o - 1 if rel < n else n - 1
            jumps = [u for u in range(A + 2, hend) if abs(qe[u + 1] - qe[u]) > 1.0]
            if jumps: hend = jumps[0]
            B5 = next((u for u in range(A, hend - 4) if np.all(np.abs(np.diff(qe[u:u + 6])) < 0.05)), None)
            cls = "empty" if qe[A] > -8 else ("short" if hend - A < 15 else "hold")
            out.append(dict(s0=s0, A=A, B5=B5, hend=hend, rel=rel, jump=bool(jumps), cls=cls))
            t = max(rel, A + 1) + 1; continue
        t += 1
    return out


def crossing(sig, k0, k1, lo, hi):
    """Sub-frame time where sig crosses (lo+hi)/2 going from lo toward hi between k0 and k1 (linear interpolation)."""
    mid = 0.5 * (lo + hi); sgn = np.sign(hi - lo)
    idx = [k for k in range(k0, k1 + 1) if np.isfinite(sig[k])]
    for a, b in zip(idx[:-1], idx[1:]):
        if sgn * (sig[a] - mid) < 0 <= sgn * (sig[b] - mid):
            if b - a > 2: return np.nan
            return a + (mid - sig[a]) / (sig[b] - sig[a]) * (b - a)
    return np.nan


def cam_x(X, I, j, t):
    """Camera x of slot j at fractional frame index t of the episode with rows I (linear interpolation; nan if a
    neighbour is missing or t is outside the episode)."""
    n = len(I); lo = int(np.floor(t)); f = t - lo
    if lo < 0 or lo >= n: return np.nan
    if f < 1e-9: return float(X[I[lo], j, 0])
    if lo + 1 >= n: return np.nan
    a, b = X[I[lo], j, 0], X[I[lo + 1], j, 0]
    return float(a + f * (b - a))


def qstat(v):
    v = np.asarray(v, float)
    return (np.median(v), np.percentile(v, 25), np.percentile(v, 75)) if len(v) else (np.nan, np.nan, np.nan)


def analyse(D, out, tag, root, cam):
    os.makedirs(out, exist_ok=True)
    ep, kk, q = D["ep"], D["k"], D["q"].astype(float); N = len(q); red = D["red"].astype(float)
    ex, nb = D["ex"].astype(float), D["nb"]
    same = np.r_[False, ep[1:] == ep[:-1]]
    dqp = np.where(same, np.r_[0, np.diff(q)], np.nan); dqn = np.r_[dqp[1:], np.nan]
    still = (np.abs(dqp) < 0.06) & (np.abs(dqn) < 0.06)
    X, fits = reassign(D, still)
    first = D["slots"].astype(float)
    print(f"\n(1) hole association. {N} frames, {still.sum()} still.")
    for j in range(3):
        cen, mx, my, nfit = fits[j]
        f1 = np.isfinite(first[:, j, 0]); f2 = np.isfinite(X[:, j, 0])
        moved = f1 & f2 & (np.abs(first[:, j, 0] - X[:, j, 0]) > 0.01)
        print(f"   {SLOT_NAMES[j]:2s}: found {f2.mean() * 100:5.1f} % of frames (first pass {f1.mean() * 100:5.1f} %), "
              f"still frames {np.mean(f2[still]) * 100:5.1f} %; lost vs first pass {np.sum(f1 & ~f2)}, gained {np.sum(~f1 & f2)}, "
              f"re-assigned to a different blob {moved.sum()}; prior from {nfit} still frames in {len(cen)} bins")

    # per-episode CSV of the per-frame features
    os.makedirs(os.path.join(out, f"csv_{tag}"), exist_ok=True)
    eL = np.nanmedian(np.where(np.isfinite(ex[:, 0]), ex[:, 0], np.nan), 1)
    eR = np.nanmedian(np.where(np.isfinite(ex[:, 1]), ex[:, 1], np.nan), 1)
    for e in np.unique(ep):
        m = ep == e
        with open(os.path.join(out, f"csv_{tag}", f"ep{e:03d}.csv"), "w") as f:
            f.write("ep,k,q_deg,xL,yL,xRu,yRu,xRl,yRl,edgeL_med,edgeR_med,red_frac\n")
            for i in np.nonzero(m)[0]:
                f.write(f"{e},{kk[i]},{q[i]:.4f}," + ",".join(f"{v:.3f}" for v in (X[i, 0, 0], X[i, 0, 1], X[i, 1, 0], X[i, 1, 1], X[i, 2, 0], X[i, 2, 1], eL[i], eR[i])) + f",{red[i]:.4f}\n")

    # ---------------------------------------------------------------- grasps
    G = []
    for e in np.unique(ep):
        I = np.nonzero(ep == e)[0]
        for g in segment(q[I]):
            g.update(ep=int(e), I=I); G.append(g)
    cls = [g["cls"] for g in G]
    print(f"\n(2) grasp events (closing steps from the open state, q < -28 deg): {len(G)} in {len(set(g['ep'] for g in G))} episodes; "
          f"hold {cls.count('hold')}, short (released < 15 frames) {cls.count('short')}, empty (landed above -8 deg) {cls.count('empty')}; "
          f"holds cut at a mid-hold jump > 1 deg/frame (a re-squeeze or slip; the hold window ends before it): {sum(g['jump'] for g in G if g['cls'] == 'hold')}")
    nres = 0
    for e in np.unique(ep):
        qe = q[ep == e]
        for u in range(3, len(qe) - 3):
            if qe[u - 1] >= -28 and qe[u + 2] - qe[u - 1] > 3 and qe[u - 1] - qe[u - 2] < 0.5: nres += 1
    print(f"   re-squeezes without re-opening (rises > 3 deg in 3 frames starting above -28 deg): {nres} frames-events; not counted as landings")

    # ---------------------------------------------------------------- (3) timing: camera vs encoder at the fast closing steps
    print("\n(3) timing: camera crossing time minus encoder crossing time at the closing steps (frames; + = camera later)")
    lags = {j: [] for j in range(3)}
    for g in G:
        I = g["I"]; qe = q[I]; s0, A = g["s0"], g["A"]; g["lag"] = np.nan; gl = []
        if s0 < 6 or A + 9 >= len(I): continue
        for j, (s, *_) in enumerate(PRIOR):
            cx = D["cands"][I, s, :, 0].astype(float); cy = D["cands"][I, s, :, 1].astype(float)
            pre = np.nanmedian(X[I[s0 - 5:s0 - 1], j, 0]); post = np.nanmedian(X[I[A + 5:A + 9], j, 0])
            if not (np.isfinite(pre) and np.isfinite(post)) or abs(post - pre) < 15: continue
            ypre = np.nanmedian(X[I[s0 - 5:s0 - 1], j, 1]); ypost = np.nanmedian(X[I[A + 5:A + 9], j, 1])
            sig = np.full(len(I), np.nan)
            for k in range(s0 - 4, A + 9):
                ok = np.isfinite(cx[k]) & (cx[k] >= min(pre, post) - 6) & (cx[k] <= max(pre, post) + 6) & (np.abs(cy[k] - 0.5 * (ypre + ypost)) <= 12)
                if ok.any(): sig[k] = cx[k][ok][np.argmin(np.abs(cy[k][ok] - 0.5 * (ypre + ypost)))]
            for k in range(s0 - 4, s0): sig[k] = X[I[k], j, 0] if np.isfinite(X[I[k], j, 0]) else sig[k]
            kc = crossing(sig, s0 - 4, A + 8, pre, post); kq = crossing(qe, s0 - 1, A + 2, qe[s0 - 1], np.median(qe[A + 5:A + 9]))
            if np.isfinite(kc) and np.isfinite(kq): lags[j].append(kc - kq); gl.append(kc - kq)
        if gl: g["lag"] = float(np.median(gl))
    for j in range(3):
        a = np.array(lags[j])
        if len(a): print(f"   closing, {SLOT_NAMES[j]:2s}: n {len(a)}, median {np.median(a):+.2f}, IQR {np.percentile(a, 25):+.2f}..{np.percentile(a, 75):+.2f}, "
                         f"fraction within +-0.5 of the median {np.mean(np.abs(a - np.median(a)) <= 0.5):.2f}")
    alll = np.concatenate([np.array(lags[j]) for j in range(3) if len(lags[j])]) if any(len(lags[j]) for j in range(3)) else np.array([])
    lag_med = float(np.median(alll)) if len(alll) else 0.0
    s_al = int(np.clip(np.round(lag_med), -2, 2))
    glag = np.array([g["lag"] for g in G if np.isfinite(g.get("lag", np.nan))])
    print(f"   all slots: n {len(alll)}, median {lag_med:+.2f} frames -> aligned integer shift s = {s_al:+d} (camera index = encoder index + s)")
    if len(glag): print(f"   per grasp (median over its slots): n {len(glag)}, median {np.median(glag):+.2f}, IQR {np.percentile(glag, 25):+.2f}..{np.percentile(glag, 75):+.2f}, "
                        f"sd {np.std(glag):.2f}")
    holds = [g for g in G if g["cls"] == "hold"]

    # ---------------------------------------------------------------- (4a) calibration
    print("\n(4a) calibration")
    print("   (i) open end: still, unloaded frames (q < -30 deg), clusters of identical q with >= 15 frames")
    cal_pts = []
    openm = still & (q < -30)
    qs = np.round(q[openm], 3)
    for v in np.unique(qs):
        m = openm & (np.round(q, 3) == v)
        if m.sum() < 15: continue
        cal_pts.append([v, m.sum()] + [np.nanmedian(X[m, j, 0]) for j in range(3)] + [np.sum(np.isfinite(X[m, j, 0])) for j in range(3)] + [len(np.unique(ep[m]))])
    cal_pts = np.array(cal_pts)
    print(f"      {len(cal_pts)} clusters, {int(cal_pts[:, 1].sum())} frames, q {cal_pts[:, 0].min():.2f}..{cal_pts[:, 0].max():.2f} deg")
    slope_open = {}; fit_open = {}
    for j in range(3):
        m = np.isfinite(cal_pts[:, 2 + j]) & (cal_pts[:, 5 + j] >= 10)
        w = np.sqrt(cal_pts[m, 5 + j])
        p = np.polyfit(cal_pts[m, 0], cal_pts[m, 2 + j], 1, w=w); res = cal_pts[m, 2 + j] - np.polyval(p, cal_pts[m, 0])
        rng = np.random.default_rng(0); bs = []; idx = np.nonzero(m)[0]
        for _ in range(400):
            b = rng.choice(idx, len(idx))
            if np.ptp(cal_pts[b, 0]) < 0.5: continue
            bs.append(np.polyfit(cal_pts[b, 0], cal_pts[b, 2 + j], 1, w=np.sqrt(cal_pts[b, 5 + j]))[0])
        slope_open[j] = p[0]; fit_open[j] = p
        print(f"      {SLOT_NAMES[j]:2s}: slope {p[0]:+.3f} px/deg (bootstrap 95 %: {np.percentile(bs, 2.5):+.3f}..{np.percentile(bs, 97.5):+.3f}, {m.sum()} clusters), "
              f"residual rms {np.sqrt(np.mean(res ** 2)):.2f} px, max {np.abs(res).max():.2f} px, span {np.ptp(cal_pts[m, 0]):.2f} deg")
    # (ii) at the hold angles: across grasps, (q, x) just after landing. Different grasps land at different angles
    # (object size, orientation), so x against q across grasps traces the pad's x(q) at the hold angles. Bias: the
    # encoder's post-landing advance (wind-up) at A+5 if it correlates with the landing angle (assumed small).
    print(f"   (ii) hold angles: across holds, encoder q at A+5 against camera x at A+5{s_al:+d} (aligned); robust line and quadratic")
    HP = []
    for g in holds:
        I = g["I"]; k5 = g["A"] + 5; ke = np.arange(max(g["hend"] - 12, g["A"] + 5), g["hend"] - 2)
        if k5 + s_al >= len(I) or len(ke) < 3 or ke[-1] + s_al >= len(I): continue
        HP.append([q[I[k5]]] + [X[I[k5 + s_al], j, 0] for j in range(3)] + [np.median(q[I[ke]])] + [np.nanmedian(X[I[ke + s_al], j, 0]) for j in range(3)])
    HP = np.array(HP)
    slope_hold = {}; fit_hold = {}
    for j in range(3):
        for nm, qi, xi in (("A+5", 0, 1), ("end", 4, 5)):
            p, keep = robust_poly(HP[:, qi], HP[:, xi + j], 1)
            res = HP[keep, xi + j] - np.polyval(p, HP[keep, qi])
            p2, _ = robust_poly(HP[:, qi], HP[:, xi + j], 2); qm = np.median(HP[keep, qi])
            rng = np.random.default_rng(1); bs = []; idx = np.nonzero(keep)[0]
            for _ in range(400):
                b = rng.choice(idx, len(idx)); bs.append(np.polyfit(HP[b, qi], HP[b, xi + j], 1)[0])
            if nm == "A+5": slope_hold[j] = p[0]; fit_hold[j] = p
            print(f"      {SLOT_NAMES[j]:2s} {nm:4s}: slope {p[0]:+.3f} px/deg (bootstrap 95 %: {np.percentile(bs, 2.5):+.3f}..{np.percentile(bs, 97.5):+.3f}), "
                  f"n {keep.sum()}/{np.isfinite(HP[:, xi + j]).sum()}, q {HP[keep, qi].min():.1f}..{HP[keep, qi].max():.1f}, residual rms {np.sqrt(np.mean(res ** 2)):.2f} px; "
                  f"quadratic slope at {qm:.1f}: {2 * p2[0] * qm + p2[1]:+.3f}")
    print("   (iii) linearity over the whole stroke: the open-end line and the hold line, each evaluated at the other end and at the closed end")
    clo = still & (q > -5)
    for j in range(3):
        qo = np.median(cal_pts[:, 0]); xo = np.polyval(fit_open[j], qo); qh = np.median(HP[:, 0])
        line = f"      {SLOT_NAMES[j]:2s}: open {qo:.2f}: x {xo:.1f} (hold line {np.polyval(fit_hold[j], qo):.1f}); hold {qh:.2f}: hold line {np.polyval(fit_hold[j], qh):.1f} " \
               f"(open line {np.polyval(fit_open[j], qh):.1f}); chord slope {(np.polyval(fit_hold[j], qh) - xo) / (qh - qo):+.3f}"
        if clo.sum(): line += f"; closed end (pads touch each other, loaded, {clo.sum()} frames) q {np.median(q[clo]):.2f}: x {np.nanmedian(X[clo, j, 0]):.1f} (hold line {np.polyval(fit_hold[j], np.median(q[clo])):.1f})"
        print(line)
    SL = {"hold": slope_hold, "open": slope_open}

    # ---------------------------------------------------------------- (4d) hold jitter
    def hold_jitter(g):
        I = g["I"]; qe = q[I]; A, he = g["A"], g["hend"]
        best = (0, 0); u = A
        while u < he:
            v = u
            while v + 1 <= he and abs(qe[v + 1] - qe[u]) < 0.03: v += 1
            if v - u + 1 > best[1] - best[0]: best = (u, v + 1)
            u = v + 1
        if best[1] - best[0] < 10: return None, best
        ks = np.arange(best[0], best[1]) + s_al; ks = ks[(ks >= 0) & (ks < len(I))]
        jit = [np.nanstd(X[I[ks], j, 0]) if np.sum(np.isfinite(X[I[ks], j, 0])) >= 8 else np.nan for j in range(3)]
        gap = X[I[ks], 1, 0] - X[I[ks], 0, 0]
        jit.append(np.nanstd(gap) if np.sum(np.isfinite(gap)) >= 8 else np.nan)
        return jit, best

    for g in holds: g["jit"], g["still_run"] = hold_jitter(g)
    J = np.array([g["jit"] for g in holds if g["jit"] is not None], float)
    runs = np.array([g["still_run"][1] - g["still_run"][0] for g in holds if g["jit"] is not None])
    print(f"\n(4d) hold jitter: {len(J)} of {len(holds)} holds have a constant-q run >= 10 frames (median run {np.median(runs):.0f} frames). per-frame std, median (IQR) over holds:")
    for j, nm in enumerate(SLOT_NAMES + ["gap(Ru-L)"]):
        a = J[:, j][np.isfinite(J[:, j])]
        if len(a): print(f"   {nm:9s}: {np.median(a):.2f} px ({np.percentile(a, 25):.2f}..{np.percentile(a, 75):.2f}), n {len(a)}")
    jit_med = np.nanmedian(J, 0)

    # ---------------------------------------------------------------- (4b) per grasp
    def r_for(g, s, which, sl):
        """s: camera shift in frames (float allowed; 'g' = the grasp's own measured lag). which: 'B5' (A -> first
        5-frame plateau, the pre-registered B), 'Bend' (A -> 2 frames before the hold ends), 'S' (A+5 -> Bend, the
        settled hold without the landing transient). sl: slope dict (px/deg per slot)."""
        I = g["I"]; n = len(I)
        if which == "S": kA, kB = g["A"] + 5, g["hend"] - 2
        else: kA, kB = g["A"], (g["B5"] if which == "B5" else g["hend"] - 2)
        if kB is None or kB <= kA or (which == "S" and kB - kA < 10): return None
        sh = (g["lag"] if np.isfinite(g.get("lag", np.nan)) else lag_med) if s == "g" else float(s)
        dq = q[I[kB]] - q[I[kA]]
        dx = [cam_x(X, I, j, kB + sh) - cam_x(X, I, j, kA + sh) for j in range(3)]
        jit = g["jit"] if g["jit"] is not None else list(jit_med)
        jit = [jit[j] if np.isfinite(jit[j]) else jit_med[j] for j in range(4)]
        rj = [dx[j] / (sl[j] * dq) if np.isfinite(dx[j]) and dq != 0 else np.nan for j in range(3)]
        exj = [abs(sl[j] * dq) < 3 * jit[j] for j in range(3)]
        jR = 1 if np.isfinite(dx[1]) else (2 if np.isfinite(dx[2]) else None)
        if jR is not None and np.isfinite(dx[0]):
            predG = (sl[jR] - sl[0]) * dq; dG = dx[jR] - dx[0]
            rG = dG / predG if predG != 0 else np.nan
            ex1 = abs(predG) < 3 * max(jit[0], jit[jR])             # pre-registered rule 4: single-edge jitter
            exG = abs(predG) < 3 * jit[3]                          # variant: jitter of the gap itself
        else:
            predG = dG = rG = np.nan; ex1 = exG = True
        return dict(dq=dq, dx=dx, rj=rj, exj=exj, predG=predG, dG=dG, rG=rG, ex1=ex1, exG=exG, jR=jR, kA=kA, kB=kB, sh=sh)

    def collect(which, s, sl, rule="ex1"):
        vals, pg, dg, nx, nall = [], [], [], 0, 0
        for g in holds:
            res = r_for(g, s, which, sl)
            if res is None or not np.isfinite(res["rG"]): continue
            nall += 1
            if res[rule]: nx += 1; continue
            vals.append(res["rG"]); pg.append(res["predG"]); dg.append(res["dG"])
        return np.array(vals), np.array(pg), np.array(dg), nx, nall

    def pooled(pg, dg, nb=400):
        if len(pg) < 3: return np.nan, np.nan, np.nan
        rr = np.sum(pg * dg) / np.sum(pg * pg); rng = np.random.default_rng(2); bs = []
        for _ in range(nb):
            b = rng.integers(0, len(pg), len(pg)); bs.append(np.sum(pg[b] * dg[b]) / np.sum(pg[b] ** 2))
        return rr, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print(f"\n(4b) per grasp (shift s = {s_al:+d}, hold-angle calibration). A = end of the encoder's fast step, B5 = start of the first 5-frame "
          "encoder plateau (|dq| < 0.05 deg), Bend = 2 frames before the hold ends. r_gap: Ru (or Rl if Ru is hidden) minus L.")
    print("   ep  s0    A   B5 Bend   q_A   lag  | dq_B5  dG_B5  r_B5  | dq_Bend dG_Bend r_Bend  r_L   r_Ru   r_Rl  | r_S   jit1  excl(B5/Bend)")
    rows = []
    def f(v, w=6, p=2): return f"{v:+{w}.{p}f}" if v is not None and np.isfinite(v) else " " * (w - 3) + "nan"
    for g in holds:
        a = r_for(g, s_al, "B5", slope_hold); b = r_for(g, s_al, "Bend", slope_hold); c = r_for(g, s_al, "S", slope_hold)
        I = g["I"]; j1 = max(g["jit"][0], g["jit"][1]) if g["jit"] is not None else np.nan
        print(f"   {g['ep']:2d} {g['s0']:4d} {g['A']:4d} {g['B5'] if g['B5'] is not None else -1:4d} {g['hend'] - 2:4d} {q[I[g['A']]]:6.2f} {f(g.get('lag', np.nan), 5)} | "
              + (f"{f(a['dq'])} {f(a['dG'])} {f(a['rG'])}" if a else " " * 20) + " | "
              + (f"{f(b['dq'], 7)} {f(b['dG'], 7)} {f(b['rG'])} {f(b['rj'][0])} {f(b['rj'][1])} {f(b['rj'][2])}" if b else " " * 41) + " | "
              + f"{f(c['rG'] if c else np.nan)} {f(j1, 5)}  {'x' if (a is None or a['ex1']) else 'ok'}/{'x' if (b is None or b['ex1']) else 'ok'}")
        rows.append(dict(ep=g["ep"], s0=g["s0"], A=g["A"], B5=g["B5"] if g["B5"] is not None else -1, Bend=g["hend"] - 2, qA=q[I[g["A"]]], lag=g.get("lag", np.nan),
                         dq_B5=a["dq"] if a else np.nan, dG_B5=a["dG"] if a else np.nan, predG_B5=a["predG"] if a else np.nan, r_B5=a["rG"] if a else np.nan,
                         ex_B5=int(a is None or a["ex1"]), dq_Bend=b["dq"] if b else np.nan, dG_Bend=b["dG"] if b else np.nan, predG_Bend=b["predG"] if b else np.nan,
                         r_Bend=b["rG"] if b else np.nan, ex_Bend=int(b is None or b["ex1"]), r_L=b["rj"][0] if b else np.nan, r_Ru=b["rj"][1] if b else np.nan,
                         r_Rl=b["rj"][2] if b else np.nan, r_S=c["rG"] if c else np.nan, jit_single=j1, jump=int(g["jump"])))
    with open(os.path.join(out, f"grasps_{tag}.csv"), "w") as fh:
        keys = list(rows[0].keys()); fh.write(",".join(keys) + "\n")
        for r_ in rows: fh.write(",".join(f"{r_[k]:.4f}" if isinstance(r_[k], float) else str(r_[k]) for k in keys) + "\n")
    print("   per-grasp table written:", f"grasps_{tag}.csv in --out")

    # ---------------------------------------------------------------- (4e) summaries
    print("\n(4e) r of the gap: median (IQR), n kept / n measurable, excluded by rule 4 (|predicted gap change| < 3 x single-edge hold jitter); "
          "pooled = sum(dG*pred)/sum(pred^2) with bootstrap 95 %")
    summ = {}
    for sln in ("hold", "open"):
        for which in ("B5", "Bend", "S"):
            for s in (-2, -1, 0, 1, 2, "g"):
                v, pg, dg, nx, nall = collect(which, s, SL[sln]); summ[(sln, which, s)] = v
                md, lo, hi = qstat(v); pr = pooled(pg, dg)
                print(f"   {sln:4s} {('A->' + which) if which != 'S' else 'A+5->Bend':9s} s {s if s == 'g' else f'{s:+d}':>2s}: r {md:+.3f} ({lo:+.3f}..{hi:+.3f}), n {len(v)}/{nall}, "
                      f"excl {nx}; pooled {pr[0]:+.3f} ({pr[1]:+.3f}..{pr[2]:+.3f})")
    v, pg, dg, nx, nall = collect("B5", s_al, slope_hold, rule="exG"); md, lo, hi = qstat(v)
    print(f"   variant exclusion (gap jitter), hold A->B5 s {s_al:+d}: r {md:+.3f} ({lo:+.3f}..{hi:+.3f}), n {len(v)}/{nall}, excl {nx}")
    print("   fractional shift scan (hold calibration), median r:")
    scan = {}
    for which in ("B5", "Bend", "S"):
        ss = np.arange(-3.0, 1.01, 0.25); md = []
        for s in ss:
            v = collect(which, s, slope_hold)[0]; md.append(np.median(v) if len(v) else np.nan)
        scan[which] = (ss, np.array(md))
        print(f"     {which:4s}: " + " ".join(f"{s:+.2f}:{m:+.2f}" for s, m in zip(ss, md)))
    for which in ("B5", "Bend"):
        v = summ[("hold", which, s_al)]
        print(f"   distribution of per-grasp r (hold, A->{which}, s {s_al:+d}): " + ", ".join(
            f"{nm} {np.sum((v >= lo_) & (v < hi_))}" for nm, lo_, hi_ in (("<0", -np.inf, 0), ("0-0.2", 0, 0.2), ("0.2-0.5", 0.2, 0.5), ("0.5-0.8", 0.5, 0.8), ("0.8-1.2", 0.8, 1.2), (">1.2", 1.2, np.inf))))
        big = [r_["r_" + which] for r_ in rows if not r_["ex_" + which] and abs(r_["dq_" + which]) >= 0.5 and np.isfinite(r_["r_" + which])]
        sm = [r_["r_" + which] for r_ in rows if not r_["ex_" + which] and abs(r_["dq_" + which]) < 0.5 and np.isfinite(r_["r_" + which])]
        print(f"      by encoder advance: |dq| >= 0.5 deg: median {np.median(big) if big else np.nan:+.3f} (n {len(big)}); |dq| < 0.5 deg: median {np.median(sm) if sm else np.nan:+.3f} (n {len(sm)})")
        pos = [r_["r_" + which] for r_ in rows if not r_["ex_" + which] and r_["dq_" + which] > 0 and np.isfinite(r_["r_" + which])]
        neg = [r_["r_" + which] for r_ in rows if not r_["ex_" + which] and r_["dq_" + which] < 0 and np.isfinite(r_["r_" + which])]
        print(f"      by direction: encoder closing further (dq > 0): median {np.median(pos) if pos else np.nan:+.3f} (n {len(pos)}); "
              f"encoder backing off (dq < 0): median {np.median(neg) if neg else np.nan:+.3f} (n {len(neg)})")
    print(f"   per pad (hold calibration, A->Bend, s {s_al:+d}, each pad's own rule-4 exclusion): ", end="")
    for j in range(3):
        v = [res["rj"][j] for g in holds for res in [r_for(g, s_al, "Bend", slope_hold)] if res is not None and np.isfinite(res["rj"][j]) and not res["exj"][j]]
        if len(v): print(f"{SLOT_NAMES[j]} {np.median(v):+.3f} ({np.percentile(v, 25):+.3f}..{np.percentile(v, 75):+.3f}) n {len(v)};  ", end="")
    print()
    for which in ("B5", "Bend"):
        print(f"   per-episode median r (hold, A->{which}, s {s_al:+d}): ", end="")
        epmed = []
        for e in sorted(set(g["ep"] for g in holds)):
            v = [r_["r_" + which] for r_ in rows if r_["ep"] == e and not r_["ex_" + which] and np.isfinite(r_["r_" + which])]
            if v: epmed.append(np.median(v)); print(f"{e}:{np.median(v):+.2f}(n{len(v)}) ", end="")
        print()
        if epmed: print(f"      median of {len(epmed)} episode medians {np.median(epmed):+.3f} (IQR {np.percentile(epmed, 25):+.3f}..{np.percentile(epmed, 75):+.3f})")

    # ---------------------------------------------------------------- (4f) within-hold regression
    print("\n(4f) within-hold regression: per hold, slope of the camera gap (Ru-L) on q over A+5..Bend (camera shifted), / calibration gap slope")
    reg = {}
    for sln in ("hold", "open"):
        sG = SL[sln][1] - SL[sln][0]
        for s in (-2, -1, 0, 1, 2):
            rr = []
            for g in holds:
                I = g["I"]; ks = np.arange(g["A"] + 5, g["hend"] - 1); ks = ks[(ks + s >= 0) & (ks + s < len(I))]
                if len(ks) < 10: continue
                gap = X[I[ks + s], 1, 0] - X[I[ks + s], 0, 0]; qq = q[I[ks]]; m = np.isfinite(gap)
                if m.sum() < 10 or np.ptp(qq[m]) < 0.2: continue
                rr.append(np.polyfit(qq[m], gap[m], 1)[0] / sG)
            reg[(sln, s)] = np.array(rr); md, lo, hi = qstat(rr)
            print(f"   {sln:4s} s {s:+d}: {len(rr)} holds with q span >= 0.2 deg: r median {md:+.3f} (IQR {lo:+.3f}..{hi:+.3f})")

    # ---------------------------------------------------------------- (4c) landing column
    print("\n(4c) landing column: camera landing frame (gap within 1.5 px/frame for 2 frames after closing > 15 px) minus encoder landing A")
    dA = []
    for g in holds:
        I = g["I"]; gap = X[I, 1, 0] - X[I, 0, 0]
        kc = next((u for u in range(g["s0"] - 3, min(g["A"] + 10, len(I) - 2)) if np.isfinite(gap[u]) and np.isfinite(gap[u + 1]) and np.isfinite(gap[u + 2])
                   and gap[u] < np.nanmedian(gap[max(g["s0"] - 8, 0):max(g["s0"] - 3, 1)]) - 15 and abs(gap[u + 1] - gap[u]) < 1.5 and abs(gap[u + 2] - gap[u + 1]) < 1.5), None)
        if kc is not None: dA.append(kc - g["A"])
    dA = np.array(dA)
    if len(dA):
        print(f"   raw (camera index - encoder index), n {len(dA)}: " + ", ".join(f"{v:+d}: {np.sum(dA == v)}" for v in range(int(dA.min()), int(dA.max()) + 1))
              + f"; within +-1 frame: {np.mean(np.abs(dA) <= 1):.2f}")
        da = dA - s_al
        print(f"   after the measured alignment (minus s = {s_al:+d}): " + ", ".join(f"{v:+d}: {np.sum(da == v)}" for v in range(int(da.min()), int(da.max()) + 1))
              + f"; within +-1 frame: {np.mean(np.abs(da) <= 1):.2f}")

    # ---------------------------------------------------------------- (4g) encoder overshoot and advance
    reb = np.array([np.max(q[g["I"]][g["s0"]:g["A"] + 3]) - q[g["I"]][g["A"] + 3] for g in holds])
    print(f"\n(4g) encoder overshoot at landing (peak within the step .. A+2, minus q at A+3; no command is recorded, so this stands in "
          f"for the command overshoot): median {np.median(reb):.2f} deg, > 0.5 deg in {np.sum(reb > 0.5)} of {len(reb)}, max {reb.max():.2f}")
    early = np.array([q[g["I"][g["A"] + 3]] - q[g["I"][g["A"]]] for g in holds]); late = np.array([q[g["I"][g["hend"] - 2]] - q[g["I"][g["A"] + 3]] for g in holds])
    b5 = np.array([q[g["I"][g["B5"]]] - q[g["I"][g["A"]]] for g in holds if g["B5"] is not None]); b5n = np.array([g["B5"] - g["A"] for g in holds if g["B5"] is not None])
    dur = np.array([(g["hend"] - 2 - g["A"]) / 30.0 for g in holds])
    print(f"   encoder advance A->B5 median {np.median(b5):+.2f} deg (IQR {np.percentile(b5, 25):+.2f}..{np.percentile(b5, 75):+.2f}) over a median {np.median(b5n):.0f} frames; "
          f"A->A+3 {np.median(early):+.2f}, A+3->Bend {np.median(late):+.2f} deg over a median {np.median(dur):.1f} s")

    # ---------------------------------------------------------------- (4i) post hoc window scan
    print("\n(4i) POST HOC (not pre-registered): r of the gap over [A+j, Bend] (hold calibration, rule-4 exclusion, Bend - (A+j) >= 10 frames); "
          "median (n) by camera shift. j = 0 is A->Bend; the landing transient is in the first 1-3 frames.")
    wsh = [-3.0, -2.5, -2.25, -2.0, -1.75, -1.5, -1.0, 0.0, 1.0]
    wscan = {}
    print("    j  enc dq (s_al) | " + " ".join(f"{x:+6.2f}  " for x in wsh))
    for j in range(9):
        md, nn, dqs = [], [], []
        for sh in wsh:
            v = []
            for g in holds:
                h = dict(g); h["A"] = g["A"] + j
                if h["hend"] - 2 - h["A"] < 10: continue
                rr = r_for(h, sh, "Bend", slope_hold)
                if rr is None or not np.isfinite(rr["rG"]) or rr["ex1"]: continue
                v.append(rr["rG"])
                if sh == s_al: dqs.append(rr["dq"])
            md.append(np.median(v) if v else np.nan); nn.append(len(v))
        wscan[j] = (np.array(wsh), np.array(md), np.array(nn), np.median(dqs) if dqs else np.nan)
        print(f"   {j:2d}  {wscan[j][3]:+.3f}        | " + " ".join(f"{m:+.2f}({n:2d})" for m, n in zip(md, nn)))
    # timing dependence of each estimator across grasps: rank correlation of per-grasp r with the grasp's own lag
    def rank(x): return np.argsort(np.argsort(x)).astype(float)
    lagdep = {}
    for which in ("B5", "Bend", "S"):
        lg, rv = [], []
        for g in holds:
            rr = r_for(g, s_al, which, slope_hold)
            if rr is None or not np.isfinite(rr["rG"]) or rr["ex1"] or not np.isfinite(g.get("lag", np.nan)): continue
            lg.append(g["lag"]); rv.append(rr["rG"])
        lg, rv = np.array(lg), np.array(rv); lagdep[which] = (lg, rv)
        if len(lg) > 5:
            hi = lg > np.median(lg)
            print(f"   per-grasp lag vs r ({'A->' + which if which != 'S' else 'A+5->Bend'}, s {s_al:+d}): rank corr {np.corrcoef(rank(lg), rank(rv))[0, 1]:+.2f}, n {len(lg)}; "
                  f"median r for the less-early half of grasps {np.median(rv[hi]):+.2f}, more-early half {np.median(rv[~hi]):+.2f}")
    # object motion inside the grasp during A+5..Bend: change of the red fraction in the central box (weak proxy)
    T = []
    for g in holds:
        rr = r_for(g, s_al, "S", slope_hold)
        if rr is None or not np.isfinite(rr["rG"]) or rr["ex1"]: continue
        I = g["I"]; n = len(I)
        def rmed(k):
            c = int(round(k + rr["sh"])); return np.median(red[I[np.clip(np.arange(c - 1, c + 2), 0, n - 1)]])
        T.append((rr["rG"], abs(rmed(rr["kB"]) - rmed(rr["kA"]))))
    T = np.array(T)
    if len(T) > 5:
        print(f"   object-motion proxy (|change of red fraction| A+5 vs Bend; median {np.median(T[:, 1]):.3f}): rank corr with r(A+5->Bend) "
              f"{np.corrcoef(rank(T[:, 0]), rank(T[:, 1]))[0, 1]:+.2f}, n {len(T)}; r for |d red| < 0.03: {np.median(T[T[:, 1] < 0.03, 0]) if np.any(T[:, 1] < 0.03) else np.nan:+.2f} "
              f"(n {np.sum(T[:, 1] < 0.03)}), >= 0.03: {np.median(T[T[:, 1] >= 0.03, 0]) if np.any(T[:, 1] >= 0.03) else np.nan:+.2f} (n {np.sum(T[:, 1] >= 0.03)})")

    # ---------------------------------------------------------------- (4h) k_obj
    print("\n(4h) k_obj = k_g (1 - r) / (4 r r_c^2), k_g 140 Nm/rad (arm A, assumed), r_c 60 / 97 mm")
    def kobj(rv): return [K_G * (1 - rv) / (4 * rv * rc ** 2) / 1000 for rc in R_C] if 0 < rv < 1 else [np.nan, np.nan]
    items = [("cube F5 (s 0) 0.11", 0.11), ("cube F5 aligned (s +1) 0.05", 0.05), ("cube F5 IQR 0.04", 0.04), ("cube F5 IQR 0.52", 0.52)]
    for sln in ("hold", "open"):
        for which in ("B5", "Bend"):
            v = summ[(sln, which, s_al)]
            if len(v): items += [(f"strawberry {sln} A->{which} s {s_al:+d} median", np.median(v)), (f"   IQR low", np.percentile(v, 25)), (f"   IQR high", np.percentile(v, 75))]
    for j in (2, 5):
        sh_, md_, nn_, _ = wscan[j]
        items.append((f"strawberry POST HOC A+{j}->Bend s {s_al:+d} median", md_[list(sh_).index(float(s_al))]))
    for nm, rv in items:
        k1, k2 = kobj(rv)
        print(f"   {nm:40s} r {rv:+.3f}: " + (f"{k2:.1f}..{k1:.1f} N/mm" if np.isfinite(k1) else "outside (0, 1): k_obj undefined"))
    return dict(X=X, G=G, holds=holds, rows=rows, slope_open=slope_open, slope_hold=slope_hold, fit_open=fit_open, fit_hold=fit_hold, cal_pts=cal_pts,
                HP=HP, lags=lags, summ=summ, scan=scan, reg=reg, dA=dA, jit=J, q=q, ep=ep, r_for=r_for, lag_med=lag_med, s_al=s_al,
                wscan=wscan, lagdep=lagdep)


def cube_control(path, slope=-5.14):
    """Control (post hoc): the same window scan on F5's wood cube, from floor/padcal.py's per-frame output (padcal.npz:
    episode, frame, encoder deg, velocity, torque, arm speed, left edge x, right edge x, red flag). Grasps and A are
    padcal.py's (|v| < 40 deg/s and torque > 1.5 Nm after a torque onset); Bend = 3 frames before the torque falls below
    half its plateau or |v| > 40. Gap slope -5.14 px/deg is F5's calibration. F5 found the camera ~1 frame LATE (s = +1)."""
    R = np.load(path)["R"]; e_, k_, th, v, tau, av, xl, xr, rb = R.T
    G = []
    for e in np.unique(e_):
        i = np.nonzero(e_ == e)[0]; t_, v_ = tau[i], v[i]
        hold = (np.abs(v_) < 2) & (t_ > 1.5)
        if hold.sum() < 5: continue
        plateau = np.median(t_[hold])
        on = next((k for k in range(10, len(t_)) if t_[k] > 1.0 and (t_[k - 10:k] < 0.3).sum() >= 7), None)
        if on is None: continue
        A = next((k for k in range(on, min(len(t_), on + 15)) if abs(v_[k]) < 40 and t_[k] > 1.5), None)
        if A is None: continue
        rest = np.nonzero(t_[A:] >= 0.95 * plateau)[0]
        if len(rest) == 0 or rest[0] == 0 or rest[0] > 10: continue
        rel = next((k for k in range(A + rest[0], len(t_)) if t_[k] < 0.5 * plateau or abs(v_[k]) > 40), len(t_))
        G.append((i, A, rel - 3))
    shifts = [-1, 0, 1, 2]; out = {}
    print(f"\n(5) CONTROL, F5 wood cube ({os.path.basename(path)}): {len(G)} grasps; r over [A+j, Bend], gap slope {slope} px/deg; median (n) by camera shift")
    print("    j  enc dq  dtau (s 0) | " + " ".join(f"{x:+d}       " for x in shifts))
    for j in range(9):
        md, nn, dqs, dts = [], [], [], []
        for sh in shifts:
            vals = []
            for i, A, Bend in G:
                kA = A + j
                if Bend - kA < 10: continue
                n = len(i); c = lambda k: min(max(k + sh, 0), n - 1)
                gap = xr[i] - xl[i]; dq = th[i][Bend] - th[i][kA]; pred = slope * dq
                cc = np.clip(np.arange(kA, Bend + 1) + sh, 0, n - 1)
                jit = max(np.std(np.diff(xl[i][cc])), np.std(np.diff(xr[i][cc]))) / np.sqrt(2)
                if abs(pred) < 3 * jit: continue
                vals.append((gap[c(Bend)] - gap[c(kA)]) / pred)
                if sh == 0: dqs.append(dq); dts.append(tau[i][Bend] - tau[i][kA])
            md.append(np.median(vals) if vals else np.nan); nn.append(len(vals))
        out[j] = (np.array(shifts, float), np.array(md), np.array(nn), np.median(dqs) if dqs else np.nan)
        print(f"   {j:2d}  {out[j][3]:+.3f}  {np.median(dts) if dts else np.nan:+.2f}   | " + " ".join(f"{m:+.2f}({n:2d})" for m, n in zip(md, nn)))
    return out


def decode_frames(root, cam, e, ks):
    """Frames ks of episode e (seek a quarter frame early so rounding cannot land on the next frame)."""
    meta = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(root, "meta/episodes/chunk-*/file-*.parquet")))])
    mi = np.nonzero(np.array(meta.column("episode_index").to_pylist()) == e)[0][0]
    seg = {c.split("/")[-1]: meta.column(c)[mi].as_py() for c in meta.column_names if c.startswith(f"videos/{cam}/")}
    vf = os.path.join(root, f"videos/{cam}/chunk-{seg['chunk_index']:03d}/file-{seg['file_index']:03d}.mp4")
    info = json.load(open(os.path.join(root, "meta/info.json"))); H, W = info["features"][cam]["shape"][:2]
    out = []
    for k in ks:
        b = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{seg['from_timestamp'] + (k - 0.25) / 30.0:.4f}", "-i", vf, "-frames:v", "1",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        out.append(np.frombuffer(b, np.uint8).reshape(H, W, 3))
    return out


def montages(res, root, cam, out, tag, D, eps_show, r0, r1):
    """(i) detector overlays at open / landing / hold / release for the first measurable hold of each episode in
    eps_show (camera frames taken at encoder index + s, the measured alignment); (ii) landing -> hold-end crops of the
    grasps with the largest encoder advance, to see whether the object shrinks."""
    from PIL import Image, ImageDraw
    X, q, ep, s_al = res["X"], res["q"], res["ep"], res["s_al"]; ex, nb = D["ex"].astype(float), D["nb"]
    COL = {0: (255, 255, 255), 1: (0, 255, 255), 2: (255, 0, 255), 3: (0, 255, 0), 4: (255, 128, 0)}
    paths = []
    for e in eps_show:
        gs = [g for g in res["holds"] if g["ep"] == e]
        if not gs: continue
        g = next((g for g in gs if np.isfinite(X[g["I"][g["A"] + s_al], 0, 0]) and np.isfinite(X[g["I"][g["A"] + s_al], 1, 0])), gs[0])
        I = g["I"]; ks = [g["s0"] - 3, g["A"] - 2, g["A"], g["A"] + 3, g["B5"] if g["B5"] is not None else g["A"] + 6, g["hend"] - 2,
                          min(g["hend"] + 6, len(I) - 1 - max(s_al, 0))]
        names = ["open", "A-2", "A (landing)", "A+3", "B5", "Bend (hold)", "release"]
        kc = [min(max(k + s_al, 0), len(I) - 1) for k in ks]
        frames = decode_frames(root, cam, e, kc); tiles = []
        for fr, k, c_, nm in zip(frames, ks, kc, names):
            sc = 1.5; im = Image.fromarray(fr.copy()).crop((120, 200, 490, 370)).resize((int(370 * sc), int(170 * sc)), Image.NEAREST)
            d = ImageDraw.Draw(im); f = lambda x, y: ((x - 120) * sc, (y - 200) * sc)
            i = I[c_]
            for s_ in (0, 1):
                for rr in range(0, ex.shape[2], 2):
                    if np.isfinite(ex[i, s_, rr]):
                        x, y = f(ex[i, s_, rr], r0 + rr); d.point([(x, y), (x + 1, y)], fill=COL[int(nb[i, s_, rr])])
            for j in range(3):
                if np.isfinite(X[i, j, 0]):
                    x, y = f(X[i, j, 0], X[i, j, 1]); d.line([x - 6, y, x + 6, y], fill=(255, 0, 0), width=2); d.line([x, y - 6, x, y + 6], fill=(255, 0, 0), width=2)
            d.rectangle([0, 0, im.width, 12], fill=(0, 0, 0))
            d.text((2, 0), f"ep{e} {nm} enc k{k} q{q[I[k]]:.2f} | cam k{c_}: L{X[i, 0, 0]:.1f} Ru{X[i, 1, 0]:.1f} Rl{X[i, 2, 0]:.1f}", fill=(255, 255, 0))
            tiles.append(im)
        nc = 4; nr = (len(tiles) + nc - 1) // nc; W_, H_ = tiles[0].size; M = Image.new("RGB", (nc * W_, nr * H_))
        for t, im in enumerate(tiles): M.paste(im, ((t % nc) * W_, (t // nc) * H_))
        pth = os.path.join(out, f"montage_{tag}_ep{e:03d}.png"); M.save(pth); paths.append(pth)
    # compression montage: aligned landing+1 vs aligned Bend for the grasps with the largest encoder advance
    cand = []
    for g in res["holds"]:
        rb = res["r_for"](g, s_al, "Bend", res["slope_hold"])
        if rb is not None and np.isfinite(rb["rG"]) and not rb["ex1"]: cand.append((rb["dq"], g, rb))
    cand.sort(key=lambda t: -t[0])
    tiles = []
    for dq, g, rb in cand[:8]:
        I = g["I"]; ka, kb = g["A"] + 1, g["hend"] - 2
        fa, fb = decode_frames(root, cam, g["ep"], [ka + s_al, kb + s_al])
        sc = 1.2; box = (150, 200, 450, 360)
        ia = Image.fromarray(fa).crop(box); ib = Image.fromarray(fb).crop(box)
        diff = np.clip(np.abs(fa.astype(int) - fb.astype(int)).sum(2) * 2, 0, 255).astype(np.uint8)
        idf = Image.fromarray(np.stack([diff] * 3, 2)).crop(box)
        row = Image.new("RGB", (3 * (box[2] - box[0]), box[3] - box[1]))
        for t, im in enumerate((ia, ib, idf)): row.paste(im, (t * (box[2] - box[0]), 0))
        d = ImageDraw.Draw(row); d.rectangle([0, 0, row.width, 12], fill=(0, 0, 0))
        d.text((2, 0), f"ep{g['ep']} enc A+1=k{ka} q{q[I[ka]]:.2f} | enc Bend=k{kb} q{q[I[kb]]:.2f} | cam +{s_al} | dq(A..Bend) {rb['dq']:+.2f} deg, "
                       f"dG {rb['dG']:+.2f} px (pred {rb['predG']:+.2f}), r {rb['rG']:+.2f} | |diff| x2", fill=(255, 255, 0))
        tiles.append(row.resize((int(row.width * sc), int(row.height * sc))))
    if tiles:
        W_, H_ = tiles[0].size; M = Image.new("RGB", (W_, H_ * len(tiles)))
        for t, im in enumerate(tiles): M.paste(im, (0, t * H_))
        pth = os.path.join(out, f"compress_{tag}.png"); M.save(pth); paths.append(pth)
    for pth in paths: print("   montage:", os.path.basename(pth), "in --out")
    return paths


def figure(res, path, tag, cube=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    q = res["q"]; cal = res["cal_pts"]; HP = res["HP"]; s_al = res["s_al"]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
    a = ax[0, 0]
    for j, c in zip(range(3), ("C0", "C1", "C2")):
        m = cal[:, 5 + j] >= 10
        a.plot(cal[m, 0], cal[m, 2 + j], "o", color=c, ms=4, label=f"{SLOT_NAMES[j]}: still open clusters")
        a.plot(HP[:, 0], HP[:, 1 + j], ".", color=c, ms=3, alpha=0.6)
        qq = np.linspace(-40, 0, 50)
        a.plot(qq, np.polyval(res["fit_open"][j], qq), "--", color=c, lw=0.8)
        a.plot(qq, np.polyval(res["fit_hold"][j], qq), "-", color=c, lw=0.8)
    a.set_xlabel("gripper encoder q [motor deg, 0 = closed]"); a.set_ylabel("hole x [px]")
    a.set_title("(a) hole x vs encoder. circles: still open clusters (dashed: their line, extended);\n"
                f"dots: each hold at A+5 (camera {s_al:+d}) (solid: their line = hold-angle calibration)", fontsize=9)
    a.legend(fontsize=7)
    a = ax[0, 1]
    pg, dg = [], []
    for g in res["holds"]:
        rb = res["r_for"](g, s_al, "B5", res["slope_hold"])
        if rb is not None and np.isfinite(rb["rG"]) and not rb["ex1"]: pg.append(rb["predG"]); dg.append(rb["dG"])
    pg, dg = np.array(pg), np.array(dg)
    a.plot(pg, dg, "k.", ms=5, label="A -> B5")
    pg2, dg2 = [], []
    for g in res["holds"]:
        rb = res["r_for"](g, s_al, "Bend", res["slope_hold"])
        if rb is not None and np.isfinite(rb["rG"]) and not rb["ex1"]: pg2.append(rb["predG"]); dg2.append(rb["dG"])
    a.plot(pg2, dg2, "x", color="C1", ms=4, label="A -> Bend")
    lim = np.array([-16, 16])
    for rr, st in ((1.0, "-"), (0.5, "--"), (0.2, ":")):
        a.plot(lim, rr * lim, st, color="C3", lw=1, label=f"r = {rr}")
    a.set_xlim(-16, 16); a.set_ylim(-16, 16)
    a.axhline(0, color="0.7", lw=0.5); a.axvline(0, color="0.7", lw=0.5)
    a.set_xlabel("predicted gap change = encoder advance x hold-angle calibration [px]"); a.set_ylabel("camera gap change (Ru - L) [px]")
    a.set_title(f"(b) per grasp, camera shift {s_al:+d}, rule-4 kept: n {len(pg)} (B5), {len(pg2)} (Bend)", fontsize=9); a.legend(fontsize=7)
    a = ax[1, 0]
    for which, c in (("B5", "C0"), ("Bend", "C1"), ("S", "C2")):
        ss, md = res["scan"][which]
        a.plot(ss, md, "-", color=c, lw=1, label=f"{'A -> ' + which if which != 'S' else 'A+5 -> Bend'} (hold cal.), fractional shift")
        si = [-2, -1, 0, 1, 2]; vv = [res["summ"][("hold", which, s)] for s in si]
        a.errorbar(np.array(si) + {"B5": -0.06, "Bend": 0, "S": 0.06}[which], [np.median(v) for v in vv],
                   yerr=[[np.median(v) - np.percentile(v, 25) for v in vv], [np.percentile(v, 75) - np.median(v) for v in vv]], fmt="o", color=c, capsize=3, ms=4)
        vo = [res["summ"][("open", which, s)] for s in si]
        a.plot(np.array(si), [np.median(v) for v in vo], "s", color=c, mfc="none", ms=5)
    for y in (0.2, 0.5): a.axhline(y, color="0.5", ls="--", lw=0.8)
    a.axvline(res["lag_med"], color="k", lw=0.8, label=f"measured alignment ({res['lag_med']:+.2f})")
    a.set_xlabel("camera shift s [frames] (camera index = encoder index + s)"); a.set_ylabel("r (gap), median; bars IQR")
    a.set_title("(c) timing robustness. filled: hold-angle calibration, open squares: open-end calibration", fontsize=9); a.legend(fontsize=7)
    a = ax[1, 1]
    allr = np.concatenate([np.array(res["lags"][j]) for j in range(3) if len(res["lags"][j])])
    a.hist(allr, bins=np.arange(-4, 2.01, 0.25), color="0.4")
    a.set_xlabel("camera crossing time - encoder crossing time at the closing step [frames]"); a.set_ylabel("count (hole x grasp)")
    a.set_title(f"(d) timing: median {np.median(allr):+.2f} frames, n {len(allr)} (negative = image leads the recorded q)", fontsize=9)
    a = ax[0, 2]
    js = sorted(res["wscan"]); lm = res["lag_med"]
    sh = res["wscan"][js[0]][0]; near = np.abs(sh - lm) <= 1.0
    med = np.array([res["wscan"][j][1][list(sh).index(float(s_al))] for j in js])
    lo = np.array([np.nanmin(res["wscan"][j][1][near]) for j in js]); hi = np.array([np.nanmax(res["wscan"][j][1][near]) for j in js])
    a.fill_between(js, lo, hi, color="C3", alpha=0.2, label=f"strawberry ({tag}): range over shifts within 1 frame of {lm:+.2f}")
    a.plot(js, med, "o-", color="C3", ms=4, label=f"strawberry ({tag}), shift {s_al:+d}")
    if cube is not None:
        cj = sorted(cube); csh = cube[cj[0]][0]; cn = np.abs(csh - 1.0) <= 1.0
        a.fill_between(cj, [np.nanmin(cube[j][1][cn]) for j in cj], [np.nanmax(cube[j][1][cn]) for j in cj], color="0.5", alpha=0.25,
                       label="F5 wood cube: range over shifts 0..+2")
        a.plot(cj, [cube[j][1][list(csh).index(1.0)] for j in cj], "s-", color="0.3", ms=4, label="F5 wood cube, shift +1 (its alignment)")
    for y in (0.2, 0.5): a.axhline(y, color="0.5", ls="--", lw=0.8)
    a.set_ylim(-0.3, 1.5); a.set_xlabel("window start: A + j [frames]  (window ends at Bend)"); a.set_ylabel("r (gap), median over grasps")
    a.set_title("(e) POST HOC: r over [A+j, Bend]. j = 0 is the landing window;\nfrom j = 2 the timing no longer matters", fontsize=9); a.legend(fontsize=7)
    a = ax[1, 2]
    for which, c, mk in (("B5", "C0", "."), ("S", "C2", "x")):
        lg, rv = res["lagdep"][which]
        rk = np.corrcoef(np.argsort(np.argsort(lg)), np.argsort(np.argsort(rv)))[0, 1] if len(lg) > 2 else np.nan
        a.plot(lg, np.clip(rv, -2, 3), mk, color=c, ms=5,
               label=f"{'A -> B5 (pre-registered)' if which == 'B5' else 'A+5 -> Bend'}, n {len(lg)}, rank corr {rk:+.2f}")
    a.axvline(lm, color="k", lw=0.8); a.axhline(0.5, color="0.5", ls="--", lw=0.8); a.axhline(0.2, color="0.5", ls="--", lw=0.8)
    a.set_xlabel("the grasp's own camera lag [frames]"); a.set_ylabel(f"r of that grasp at shift {s_al:+d} (clipped to -2..3)")
    a.set_title("(f) per grasp: r against the grasp's own timing (one global shift is applied)", fontsize=9); a.legend(fontsize=7)
    fig.suptitle(f"F7 strawberry, camera vs encoder after landing ({tag})", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=110); print("   figure:", os.path.basename(path))


def main():
    root, cam = sys.argv[1], sys.argv[2]
    e0, e1 = [int(x) for x in opt("--episodes", "0-39").split("-")]
    r0, r1 = [int(x) for x in opt("--rows", "230-335").split("-")]
    dark_frac = opt("--dark", 0.55); col = opt("--col", 14); xc = opt("--xc", 296)
    save, load = opt("--save", ""), opt("--load", ""); out = opt("--out", "."); tag = opt("--tag", "v2")
    EP, FI, S, A, V, E = load_state(root)
    data_facts(S, A, V, E, col)
    if load:
        D = dict(np.load(load))
    else:
        D = extract(root, cam, range(e0, e1 + 1), r0, r1, xc, dark_frac, EP, S, col)
        if save: np.savez_compressed(save, **D)
    res = analyse(D, out, tag, root, cam)
    cube = cube_control(opt("--cube", "")) if "--cube" in sys.argv else None
    fig = opt("--fig", "")
    if fig: figure(res, fig, tag, cube)
    if "--montage" in sys.argv:
        montages(res, root, cam, out, tag, D, [int(x) for x in opt("--montage", "0,19,27,33").split(",")], r0, r1)


if __name__ == "__main__":
    main()
