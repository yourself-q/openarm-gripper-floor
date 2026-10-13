"""Self-calibrated pad motion from the wrist camera. The camera is fixed to the gripper body, so the image position
of each finger's inner edge depends on the finger angle only, not on the arm. Decode every frame of each episode
with a local video, find the inner edge of the left and right finger (scan outward from the image centre along the
contact rows until the pixels go dark for 3 px; sub-pixel by linear interpolation of the brightness crossing), and
record it with the encoder angle, velocity and torque. Then: (a) calibration px/deg of each edge from still, unloaded
frames across angles; (b) for each grasp, the edge travel from landing (A) to full hold (B) against the encoder's
travel, as a fraction of what the calibration predicts; (c) edge jitter during the constant hold (noise floor).
usage: python3 -I padcal.py ROOT CAMKEY [--episodes 0-47] [--rows 280-320] [--dark 70] [--save padcal.npz]"""
import glob, json, os, subprocess, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa

root, cam = sys.argv[1], sys.argv[2]
def opt(k, d):
    return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
e0, e1 = [int(x) for x in opt("--episodes", "0-47").split("-")]
r0, r1 = [int(x) for x in opt("--rows", "280-320").split("-")]
DARK = opt("--dark", 70.0); save = opt("--save", "padcal.npz")

info = json.load(open(os.path.join(root, "meta/info.json")))
tab = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))])
EP = np.array(tab.column("episode_index").to_pylist()); TS = np.array(tab.column("timestamp").to_pylist(), float)
S = np.array(tab.column("observation.state").to_pylist(), float)
POS, VEL, TAU = S[:, 21], S[:, 22], S[:, 23]; ARMV = np.abs(S[:, 1:21:3]).max(1)
meta = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(root, "meta/episodes/chunk-*/file-*.parquet")))])
MEP = np.array(meta.column("episode_index").to_pylist())
H, W = info["features"][cam]["shape"][:2]; XC = W // 2

def edges(frame):
    """Inner edge x of the left and right finger on the contact rows (median over rows), and whether red lies between."""
    band = frame[r0:r1 + 1].astype(float)
    bright = band.max(2)                                    # max(R,G,B)
    red = (band[:, :, 0] - 0.5 * (band[:, :, 1] + band[:, :, 2]) > 40)
    xl, xr = [], []
    for row in range(bright.shape[0]):
        b = bright[row]
        dark = b < DARK
        # left: first x < XC where dark for 3 consecutive px (x, x-1, x-2)
        d3 = dark[2:XC + 1] & dark[1:XC] & dark[0:XC - 1]      # index i -> x = i + 2 is the innermost of the three
        ii = np.nonzero(d3)[0]
        if len(ii):
            x = ii[-1] + 2
            # crossing between x (dark) and x+1 (bright): interpolate where b == DARK
            if x + 1 < W and b[x + 1] != b[x]: xl.append(x + (DARK - b[x]) / (b[x + 1] - b[x]))
            else: xl.append(float(x))
        d3r = dark[XC:W - 2] & dark[XC + 1:W - 1] & dark[XC + 2:W]
        jj = np.nonzero(d3r)[0]
        if len(jj):
            x = jj[0] + XC
            if x - 1 >= 0 and b[x - 1] != b[x]: xr.append(x - (DARK - b[x]) / (b[x - 1] - b[x]))
            else: xr.append(float(x))
    red_between = bool(red[:, XC - 40:XC + 40].mean() > 0.5)
    return (np.median(xl) if len(xl) >= 10 else np.nan), (np.median(xr) if len(xr) >= 10 else np.nan), red_between

rows = []
for e in range(e0, e1 + 1):
    idx = np.nonzero(EP == e)[0]
    if len(idx) == 0: continue
    mi = np.nonzero(MEP == e)[0]
    if len(mi) == 0: continue
    seg = {c.split("/")[-1]: meta.column(c)[mi[0]].as_py() for c in meta.column_names if c.startswith(f"videos/{cam}/")}
    vf = os.path.join(root, f"videos/{cam}/chunk-{seg.get('chunk_index', 0):03d}/file-{seg.get('file_index', 0):03d}.mp4")
    if not os.path.exists(vf): continue
    n = len(idx); dur = n / 30.0
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-ss", f"{seg['from_timestamp']:.4f}", "-i", vf, "-t", f"{dur + 0.02:.4f}", "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    k = 0; nb = H * W * 3
    while k < n:
        buf = p.stdout.read(nb)
        if len(buf) < nb: break
        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        xl, xr, rb = edges(fr)
        rows.append((e, k, POS[idx[k]], VEL[idx[k]], TAU[idx[k]], ARMV[idx[k]], xl, xr, rb))
        k += 1
    p.stdout.close(); p.wait()
    print(f"  ep {e}: {k}/{n} frames decoded", flush=True)
R = np.array(rows, float); np.savez(save, R=R)
e_, k_, th, v, tau, av, xl, xr, rb = R.T
print(f"{len(R)} frames; edge found L {np.isfinite(xl).mean():.2f} R {np.isfinite(xr).mean():.2f}")

# (a) calibration: still, unloaded (|tau| < 1.3 i.e. inside the friction band, no red between pads)
cal = (np.abs(v) < 2) & (np.abs(tau) < 1.3) & (rb < 0.5) & np.isfinite(xl) & np.isfinite(xr)
print("\n(a) calibration, still unloaded frames: angle bin, n, x_left median, x_right median, gap")
bins = np.arange(-56, 1, 2.0); cal_pts = []
for b in bins:
    m = cal & (th >= b) & (th < b + 2)
    if m.sum() >= 15:
        print(f"   [{b:6.1f},{b+2:6.1f})  {m.sum():5d}  {np.median(xl[m]):7.2f}  {np.median(xr[m]):7.2f}  {np.median(xr[m]-xl[m]):7.2f}")
        cal_pts.append((b + 1, np.median(xl[m]), np.median(xr[m])))
cal_pts = np.array(cal_pts)
def local_slope(pts, col, ang, half=6):
    m = np.abs(pts[:, 0] - ang) <= half
    if m.sum() < 2: return np.nan
    return np.polyfit(pts[m, 0], pts[m, col], 1)[0]
# (b) grasps
print("\n(b) grasps: ep, A..B, d_theta, d_tau, dxL, dxR, predicted dxL/dxR from calibration at the hold angle, visible fraction L/R, arm|v|max")
fr = []
for e in np.unique(e_):
    m = e_ == e; i = np.nonzero(m)[0]
    t_, v_, p_, x_l, x_r, a_ = tau[i], v[i], th[i], xl[i], xr[i], av[i]
    hold = (np.abs(v_) < 2) & (t_ > 1.5)
    if hold.sum() < 5: continue
    plateau = np.median(t_[hold])
    on = next((k for k in range(10, len(t_)) if t_[k] > 1.0 and (t_[k-10:k] < 0.3).sum() >= 7), None)
    if on is None: continue
    A = next((k for k in range(on, min(len(t_), on + 15)) if abs(v_[k]) < 40 and t_[k] > 1.5), None)
    if A is None: continue
    rest = np.nonzero(t_[A:] >= 0.95 * plateau)[0]
    if len(rest) == 0 or rest[0] == 0 or rest[0] > 10: continue
    B = A + rest[0]
    if not (np.isfinite(x_l[A]) and np.isfinite(x_l[B]) and np.isfinite(x_r[A]) and np.isfinite(x_r[B])): continue
    dth = p_[B] - p_[A]; dxl = x_l[B] - x_l[A]; dxr = x_r[B] - x_r[A]
    sl, sr = local_slope(cal_pts, 1, p_[B]), local_slope(cal_pts, 2, p_[B])
    pl, pr = sl * dth, sr * dth
    fl, frr = (dxl / pl if pl else np.nan), (dxr / pr if pr else np.nan)
    armm = a_[A:B + 1].max()
    print(f"   {int(e):3d} {A:4d}..{B:<4d} {dth:+6.3f} {t_[B]-t_[A]:5.2f}  {dxl:+6.2f} {dxr:+6.2f}   pred {pl:+6.2f} {pr:+6.2f}   frac {fl:+5.2f} {frr:+5.2f}   {armm:5.1f}")
    fr.append((e, dth, dxl, dxr, pl, pr, fl, frr, armm))
fr = np.array(fr)
if len(fr):
    print(f"   {len(fr)} grasps: visible fraction median L {np.nanmedian(fr[:,6]):+.2f} R {np.nanmedian(fr[:,7]):+.2f}; "
          f"dxL median {np.median(fr[:,2]):+.2f} px, dxR {np.median(fr[:,3]):+.2f} px for d_theta median {np.median(fr[:,1]):+.3f} deg")
    q = fr[fr[:, 8] < 5]
    if len(q): print(f"   arm-still (<5 deg/s) {len(q)}: frac L {np.nanmedian(q[:,6]):+.2f} R {np.nanmedian(q[:,7]):+.2f}")
# (c) hold jitter
hm = (np.abs(v) < 2) & (tau > 1.5) & np.isfinite(xl) & np.isfinite(xr)
jl, jr = [], []
for e in np.unique(e_):
    m = hm & (e_ == e)
    if m.sum() >= 20: jl.append(np.std(xl[m])); jr.append(np.std(xr[m]))
print(f"\n(c) edge jitter during holds (std per episode, median over episodes): L {np.median(jl):.2f} px, R {np.median(jr):.2f} px")
