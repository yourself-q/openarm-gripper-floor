"""Figure for the floor note: what the wrist camera sees while the sensorless OpenArm 2.0 gripper squeezes a wooden
cube, and how two groups of datasets recorded at the same gains (most likely two arms, by name; unconfirmed) differ in
their friction band.

Panels: (a) two frames of one grasp (landing, hold) with the detected inner finger edges; (b) image gap between the
finger edges vs encoder angle on still, unloaded frames (self-calibration, no object between the pads); (c) one landing
sequence: measured gap vs the gap the encoder predicts through the calibration, anchored at landing; (d) all grasps:
predicted vs measured gap change from landing to hold; (e) |torque| on strictly still, free, push-closed frames for the
two dataset groups (band.py's criterion).

usage: python3 -I floor_fig.py --npz padcal.npz --root ROOT_A --rootB ROOT_B [--cam observation.images.cam] [--ep 6]
       [--out floor_fig.png] [--tweet floor_fig_abd.png]   (the second file holds panels a, b, d only)"""
import glob, json, os, subprocess, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec


def opt(k, d):
    return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


NPZ, ROOT, ROOTB = opt("--npz", "padcal.npz"), opt("--root", ""), opt("--rootB", "")
CAM, EPX, OUT, TW = opt("--cam", "observation.images.cam"), opt("--ep", 6), opt("--out", "floor_fig.png"), opt("--tweet", "")
R0, R1 = 280, 320                                      # contact rows used by padcal.py

R = np.load(NPZ)["R"]
e_, k_, th, v, tau, av, xl, xr, rb = R.T
gap = xr - xl

# ---- (b) calibration: still, unloaded, nothing red between the pads
cal = (np.abs(v) < 2) & (np.abs(tau) < 1.3) & (rb < 0.5) & np.isfinite(gap)
bins = np.arange(-56, 1, 2.0); pts = []
for b in bins:
    m = cal & (th >= b) & (th < b + 2)
    if m.sum() >= 15: pts.append((b + 1, np.median(xl[m]), np.median(xr[m]), np.median(gap[m]), m.sum()))
pts = np.array(pts)


def local_slope(col, ang, half=6):
    m = np.abs(pts[:, 0] - ang) <= half
    return np.polyfit(pts[m, 0], pts[m, col], 1)[0] if m.sum() >= 2 else np.nan


# ---- (d) grasps, the rules of padcal.py
G = []; seqs = {}
for e in np.unique(e_):
    i = np.nonzero(e_ == e)[0]
    t_, v_, p_, g_ = tau[i], v[i], th[i], gap[i]
    hold = (np.abs(v_) < 2) & (t_ > 1.5)
    if hold.sum() < 5: continue
    plateau = np.median(t_[hold])
    on = next((k for k in range(10, len(t_)) if t_[k] > 1.0 and (t_[k - 10:k] < 0.3).sum() >= 7), None)
    if on is None: continue
    A = next((k for k in range(on, min(len(t_), on + 15)) if abs(v_[k]) < 40 and t_[k] > 1.5), None)
    if A is None: continue
    rest = np.nonzero(t_[A:] >= 0.95 * plateau)[0]
    if len(rest) == 0 or rest[0] == 0 or rest[0] > 10: continue
    B = A + rest[0]
    if not (np.isfinite(g_[A]) and np.isfinite(g_[B])): continue
    s = local_slope(3, p_[B])
    G.append((e, A, B, p_[B] - p_[A], t_[B] - t_[A], g_[B] - g_[A], s * (p_[B] - p_[A])))
    seqs[int(e)] = (i, A, B)
G = np.array(G)
frac = G[:, 5] / G[:, 6]
# hold: one value per grasp (median over its still, loaded frames), then the median over grasps
hold_m = (np.abs(v) < 2) & (tau > 1.5) & np.isfinite(gap)
hold_eps = [e for e in np.unique(e_[hold_m]) if (hold_m & (e_ == e)).sum() >= 5]
hold_gap_ep = np.array([np.median(gap[hold_m & (e_ == e)]) for e in hold_eps])
hold_th_ep = np.array([np.median(th[hold_m & (e_ == e)]) for e in hold_eps])
hold_th, hold_gap = np.median(hold_th_ep), np.median(hold_gap_ep)
m_fit = (pts[:, 0] >= -29) & (pts[:, 0] <= -21)
pf = np.polyfit(pts[m_fit, 0], pts[m_fit, 3], 1); s_hold = pf[0]; ext_gap = np.polyval(pf, hold_th)
print(f"hold: {len(hold_eps)} grasps, encoder median {hold_th:.1f} deg, image gap median {hold_gap:.0f} px "
      f"(IQR {np.percentile(hold_gap_ep, 25):.0f}..{np.percentile(hold_gap_ep, 75):.0f}, range {hold_gap_ep.min():.0f}..{hold_gap_ep.max():.0f}); "
      f"free-motion line ({s_hold:+.2f} px/deg over -29..-21) extrapolated to the hold angle: {ext_gap:.0f} px")
print(f"{len(G)} grasps; visible fraction median {np.median(frac):+.2f} (IQR {np.percentile(frac, 25):+.2f}..{np.percentile(frac, 75):+.2f}); "
      f"d_theta median {np.median(G[:, 3]):+.2f} deg, d_tau {np.median(G[:, 4]):+.2f} Nm, gap change {np.median(G[:, 5]):+.2f} px vs predicted {np.median(G[:, 6]):+.2f} px")

# ---- (e) still-torque band of two units, band.py's criterion
def band(root):
    tabs = []
    for f in sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True)):
        try: tabs.append(pq.read_table(f, columns=["observation.state", "episode_index"]))
        except Exception: pass
    tab = pa.concat_tables(tabs)
    S = np.array(tab.column("observation.state").to_pylist(), float); ep = np.array(tab.column("episode_index").to_pylist())
    pos, vel, tq = S[:, 21], S[:, 22], S[:, 23]; n = len(pos); W = 3
    still = np.abs(vel) < 1.0; rng = np.zeros(n, bool)
    for i in range(W, n - W):
        seg = pos[i - W:i + W + 1]; rng[i] = (seg.max() - seg.min()) < 0.03 and (ep[i - W] == ep[i + W])
    still &= rng
    lo, hi = np.percentile(pos, 0.5), np.percentile(pos, 99.5); mid = (pos > lo + 3) & (pos < hi - 3)
    hold = {}
    for e in np.unique(ep):
        m = (ep == e) & still & (tq > 1.5)
        if m.sum() > 5: hold[e] = np.median(pos[m])
    wider = np.array([pos[i] < hold.get(ep[i], np.nan) - 6 for i in range(n)])
    cl = still & mid & (tq > 0) & wider; op = still & mid & (tq < 0)
    return np.abs(tq[cl]), np.abs(tq[op]), np.median(list(hold.values()))


bA = band(ROOT) if ROOT else None; bB = band(ROOTB) if ROOTB else None
for lab, b in (("unit A", bA), ("unit B", bB)):
    if b is not None:
        print(f"{lab}: push-closed still n {len(b[0])} p50 {np.median(b[0]):.2f} p90 {np.percentile(b[0], 90):.2f}; "
              f"pull-open n {len(b[1])} p50 {np.median(b[1]):.2f} p90 {np.percentile(b[1], 90):.2f}; hold {b[2]:.1f} deg")

# ---- (a) frames of one grasp
def frames_for(root, cam, e, ks):
    info = json.load(open(os.path.join(root, "meta/info.json")))
    H, W = info["features"][cam]["shape"][:2]
    tab = pa.concat_tables([pq.read_table(f, columns=["episode_index", "timestamp"]) for f in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))])
    EP = np.array(tab.column("episode_index").to_pylist()); TS = np.array(tab.column("timestamp").to_pylist(), float)
    meta = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(root, "meta/episodes/chunk-*/file-*.parquet")))])
    MEP = np.array(meta.column("episode_index").to_pylist()); mi = np.nonzero(MEP == e)[0][0]
    seg = {c.split("/")[-1]: meta.column(c)[mi].as_py() for c in meta.column_names if c.startswith(f"videos/{cam}/")}
    vf = os.path.join(root, f"videos/{cam}/chunk-{seg.get('chunk_index', 0):03d}/file-{seg.get('file_index', 0):03d}.mp4")
    ts = TS[EP == e]; out = []
    for k in ks:
        t = seg["from_timestamp"] + ts[k] - ts[0]
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.4f}", "-i", vf, "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        out.append(np.frombuffer(raw, np.uint8).reshape(H, W, 3))
    return out


# ---- figure
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9})
fig = plt.figure(figsize=(14.5, 8.6), dpi=150)
gs = gridspec.GridSpec(2, 3, width_ratios=[1.25, 1, 1], wspace=0.30, hspace=0.50)

iA, A, B = seqs[EPX]
kk = [A, B]
axs_a = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
if ROOT:
    fr = frames_for(ROOT, CAM, EPX, kk)
    for j, (ax, f, k, lab) in enumerate(zip(axs_a, fr, kk, ("landing (A)", "hold (B)"))):
        crop = f[150:450]
        ax.imshow(crop); ax.set_xticks([]); ax.set_yticks([])
        for x in (xl[iA[k]], xr[iA[k]]):
            ax.axvline(x, color="cyan", lw=1.0)
        ax.axhspan(R0 - 150, R1 - 150, color="yellow", alpha=0.12, lw=0)
        ax.set_title(f"(a) ep {EPX}, {lab}\nencoder {th[iA[k]]:.1f} deg, torque {tau[iA[k]]:.2f} Nm, image gap {gap[iA[k]]:.1f} px")
    fig.axes[0].text(0.01, 0.03, "wrist camera, 640x480, 30 fps", color="w", fontsize=7.5,
                     transform=fig.axes[0].transAxes)

# (b)
axb = ax = fig.add_subplot(gs[0, 1])
ax.plot(th[cal], gap[cal], ".", ms=2, color="0.75", label=f"still, unloaded frames (n={cal.sum()})")
brk = np.nonzero(np.diff(pts[:, 0]) > 4)[0]
for seg in np.split(np.arange(len(pts)), brk + 1):
    ax.plot(pts[seg, 0], pts[seg, 3], "o-", ms=3.5, color="C0", lw=1.2, label="2-deg bin medians" if seg[0] == 0 else None)
xx = np.array([pts[m_fit, 0].max(), hold_th])
ax.plot(xx, np.polyval(pf, xx), ":", color="C3", lw=1.0, label=f"extrapolated to {hold_th:.1f} deg: {ext_gap:.0f} px")
ax.plot([hold_th], [ext_gap], "s", mfc="none", color="C3", ms=6)
ax.plot([hold_th], [hold_gap], "s", color="C3", ms=6,
        label=f"hold on the cube: {hold_gap:.0f} px (median of {len(hold_eps)} grasps)")
ax.set_xlabel("gripper encoder angle (deg; 0 = closed)"); ax.set_ylabel("image gap between finger edges (px)")
ax.set_title(f"(b) self-calibration: still frames, nothing between the pads\n{s_hold:+.2f} px/deg over -29..-21 deg\nthe hold angle lies 3 deg beyond the last bin")
ax.legend(fontsize=7.5, loc="lower left"); ax.grid(alpha=0.3)

# (c)
ax = fig.add_subplot(gs[0, 2])
w = np.arange(max(0, A - 4), min(len(iA), B + 11)); rel = w - A
g_meas = gap[iA[w]]; s_loc = local_slope(3, th[iA[B]])
g_pred = g_meas[rel == 0][0] + s_loc * (th[iA[w]] - th[iA[A]])
ax.plot(rel, g_meas, "o-", color="C0", ms=3.5, label="image gap (measured)")
ax.plot(rel[rel >= 0], g_pred[rel >= 0], "--", color="C3", label=f"gap the encoder predicts ({s_loc:+.1f} px/deg)")
ax.set_xlabel("frames after landing A (30 fps)"); ax.set_ylabel("image gap (px)")
ax2 = ax.twinx(); ax2.plot(rel, tau[iA[w]], "s-", color="0.4", ms=2.5, lw=0.8, label="motor torque (Nm)"); ax2.set_ylabel("motor torque (Nm)")
ax.axvline(0, color="k", lw=0.6, alpha=0.5); ax.axvline(B - A, color="k", lw=0.6, alpha=0.5)
ax.set_title(f"(c) ep {EPX} landing: encoder {th[iA[A]]:.1f} -> {th[iA[B]]:.1f} deg from A to B\nthe image gap stays within 0.3 px")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.55))
ax.set_ylim(min(g_meas.min(), g_pred.min()) - 6, g_meas.max() + 10); ax.grid(alpha=0.3)

# (d)
axd = ax = fig.add_subplot(gs[1, 1])
ax.plot(G[:, 6], G[:, 5], "o", color="C0", ms=4)
lim = (min(G[:, 6].min(), -1) - 1, 1)
ax.plot(lim, lim, "--", color="C3", lw=1, label="pads move as the encoder says (y = x)")
ax.axhline(0, color="k", lw=0.6); ax.set_xlim(lim); ax.set_ylim(-4, 3)
ax.set_xlabel("gap change the encoder predicts, A -> B (px)"); ax.set_ylabel("gap change measured in the image (px)")
ax.set_title(f"(d) {len(G)} grasps, landing -> hold\nmedian visible fraction {np.median(frac):.2f} by gap (0.11 per edge)\nhold jitter 0.5-0.7 px per edge")
ax.legend(fontsize=7.5, loc="lower left"); ax.grid(alpha=0.3)

# (e)
ax = fig.add_subplot(gs[1, 2])
edges = np.linspace(0, 1.4, 36)
for lab, b, c in (("unit A: pick-place-100ep-v2", bA, "C0"), ("unit B: pick-place-newarm", bB, "C1")):
    if b is None: continue
    ax.hist(b[0], bins=edges, density=True, alpha=0.5, color=c,
            label=f"{lab}: p50 {np.median(b[0]):.2f}, p90 {np.percentile(b[0], 90):.2f} Nm (n={len(b[0])})")
ax.set_xlabel("|motor torque| (Nm)"); ax.set_ylabel("density"); ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.set_title("(e) two dataset groups at the same kp 16\n(most likely two arms, by name; unconfirmed)\n|torque| strictly still, pushing closed, jaws empty")
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

st = fig.suptitle("OpenArm 2.0 sensorless gripper (DM4310, MIT mode kp 16): what the encoder, the torque and the wrist camera each see", fontsize=11, y=0.995)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)

if TW:                                                 # panels a, b, d only, for a post
    from matplotlib.transforms import Bbox
    keep = axs_a + [axb, axd]
    for a in fig.axes:
        a.set_visible(a in keep)
    st.set_visible(False)
    r = fig.canvas.get_renderer()
    bb = Bbox.union([a.get_tightbbox(r) for a in keep]).transformed(fig.dpi_scale_trans.inverted()).expanded(1.03, 1.03)
    fig.savefig(TW, bbox_inches=bb)
    print("wrote", TW)
