"""Does the wrist camera see the squeeze, and where does the compliance live? For each grasp in a torque-logged
OpenArm 2.0 dataset with local video: onset = first frame with torque > 1 Nm after the jaw was free; A = first frame
after onset where the jaw has landed (|v| < 40 deg/s, torque > 1.5); B = first frame after A at 95 % of the hold
plateau. Between A and B the motor encoder turns a further d_theta while the torque rises d_tau. Once pinched, cube
and pads are rigid with the camera, so the visible red extent of the cube between the two pads (soft red mask summed
over the contact rows, sub-pixel through codec edge blur) can only shrink by the pads' inward travel. We print, per
grasp: d_theta, d_tau, red width at A and B, the slope px/Nm and px/deg, plus phase-correlation shifts of a left-pad,
right-pad and cube crop for a cross-check. Encoder-vs-camera ratio tells whether the compliance is downstream of
the finger (visible) or inside the transmission (invisible).
usage: python3 -I squeeze2.py ROOT CAMKEY [--episodes 0-60] [--rows 265-325] [--xwin 150-500]"""
import glob, json, os, subprocess, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa

root, cam = sys.argv[1], sys.argv[2]
def opt(k, d):
    return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
e0, e1 = [int(x) for x in opt("--episodes", "0-60").split("-")]
r0, r1 = [int(x) for x in opt("--rows", "265-325").split("-")]
x0, x1 = [int(x) for x in opt("--xwin", "150-500").split("-")]
PRE = 3
CROPS = {"Lpad": (200, 230, 100, 100), "Rpad": (410, 230, 100, 100), "cube": (300, 215, 110, 120)}

info = json.load(open(os.path.join(root, "meta/info.json")))
tab = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))])
EP = np.array(tab.column("episode_index").to_pylist()); TS = np.array(tab.column("timestamp").to_pylist(), float)
S = np.array(tab.column("observation.state").to_pylist(), float)
POS, VEL, TAU = S[:, 21], S[:, 22], S[:, 23]; ARMV = S[:, 1:21:3]
meta = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(root, "meta/episodes/chunk-*/file-*.parquet")))])
MEP = np.array(meta.column("episode_index").to_pylist())
H, W = info["features"][cam]["shape"][:2]

def frame_rgb(vf, t):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.4f}", "-i", vf, "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(H, W, 3).astype(float)

def redness(img):
    return img[:, :, 0] - 0.5 * (img[:, :, 1] + img[:, :, 2])

def red_width(img):
    """Soft count of red pixels per contact row inside the x window, averaged over rows; edges count fractionally."""
    r = redness(img)[r0:r1 + 1, x0:x1 + 1]
    soft = np.clip((r - 40.0) / 60.0, 0, 1)          # 0 below redness 40, 1 above 100 (cube face ~ 150+)
    return soft.sum(1).mean(), soft.sum(1).std()

def shift_upsampled(a, b, up=20):
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A = np.fft.fft2((a - a.mean()) * win); B = np.fft.fft2((b - b.mean()) * win)
    R = A * np.conj(B); R /= np.abs(R) + 1e-9
    cc = np.fft.ifft2(R); pk = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    sh = np.array(pk, float); big = sh > np.array(a.shape) / 2; sh[big] -= np.array(a.shape)[big]
    ny, nx = a.shape; ky = np.fft.fftfreq(ny); kx = np.fft.fftfreq(nx)
    off = np.arange(-1.5, 1.5 + 1e-9, 1.0 / up)
    Ey = np.exp(2j * np.pi * np.outer(sh[0] + off, ky)); Ex = np.exp(2j * np.pi * np.outer(kx, sh[1] + off))
    C = np.abs(Ey @ R @ Ex); p = np.unravel_index(np.argmax(C), C.shape)
    return -(sh[0] + off[p[0]]), -(sh[1] + off[p[1]])

print(f"{os.path.basename(root)}: {H}x{W}, contact rows {r0}-{r1}, x window {x0}-{x1}")
print("  ep   A..B  d_theta[deg] d_tau[Nm]  width_A  width_B  dW[px]  px/Nm  px/deg  arm|v|max  Lpad dx  Rpad dx  cube dx  (phase corr, px, B rel A)")
res = []
for e in range(e0, e1 + 1):
    idx = np.nonzero(EP == e)[0]
    if len(idx) == 0: continue
    mi = np.nonzero(MEP == e)[0]
    if len(mi) == 0: continue
    seg = {c.split("/")[-1]: meta.column(c)[mi[0]].as_py() for c in meta.column_names if c.startswith(f"videos/{cam}/")}
    vf = os.path.join(root, f"videos/{cam}/chunk-{seg.get('chunk_index', 0):03d}/file-{seg.get('file_index', 0):03d}.mp4")
    if not os.path.exists(vf): continue
    pos, vel, tau, armv, ts = POS[idx], VEL[idx], TAU[idx], ARMV[idx], TS[idx]
    hold_m = (np.abs(vel) < 2.0) & (tau > 1.5)
    if hold_m.sum() < 5: continue
    plateau = np.median(tau[hold_m])
    on = None
    for k in range(10, len(tau)):
        if tau[k] > 1.0 and (tau[k-10:k] < 0.3).sum() >= 7: on = k; break
    if on is None: print(f"  {e:3d}  no onset"); continue
    A = next((k for k in range(on, min(len(tau), on + 15)) if abs(vel[k]) < 40 and tau[k] > 1.5), None)
    if A is None: print(f"  {e:3d}  jaw never landed"); continue
    rest = np.nonzero(tau[A:] >= 0.95 * plateau)[0]
    if len(rest) == 0 or rest[0] > 10 or rest[0] == 0: print(f"  {e:3d}  plateau {'at A already' if len(rest) and rest[0]==0 else 'not reached'}"); continue
    B = A + rest[0]
    arm_max = np.abs(armv[A:B + 1]).max()
    frames = {k: frame_rgb(vf, seg["from_timestamp"] + ts[k] - ts[0]) for k in range(A, B + 1)}
    widths = {k: red_width(frames[k])[0] for k in frames}
    dth = pos[B] - pos[A]; dtau = tau[B] - tau[A]; dW = widths[B] - widths[A]
    # regression of width on tau over A..B
    t_ = np.array([tau[k] for k in frames]); w_ = np.array([widths[k] for k in frames])
    slope = np.polyfit(t_, w_, 1)[0] if len(t_) >= 3 else dW / dtau
    ga = frames[A].mean(2); gb = frames[B].mean(2)
    sh = {n: shift_upsampled(ga[y:y + h, x:x + w], gb[y:y + h, x:x + w]) for n, (x, y, w, h) in CROPS.items()}
    print(f"  {e:3d} {A:4d}..{B:<4d} {dth:+9.3f}   {dtau:6.2f}   {widths[A]:7.2f}  {widths[B]:7.2f}  {dW:+6.2f}  {slope:+5.2f}  "
          f"{dW/dth if abs(dth) > 0.05 else float('nan'):+6.2f}   {arm_max:6.1f}    {sh['Lpad'][1]:+5.2f}    {sh['Rpad'][1]:+5.2f}    {sh['cube'][1]:+5.2f}")
    res.append((e, dth, dtau, dW, slope, arm_max, sh["Lpad"][1], sh["Rpad"][1], sh["cube"][1]))
if res:
    r = np.array([x[1:] for x in res])
    print(f"\n{len(res)} grasps. medians: d_theta {np.median(r[:,0]):+.3f} deg, d_tau {np.median(r[:,1]):.2f} Nm, dW {np.median(r[:,2]):+.2f} px, "
          f"px/Nm {np.median(r[:,3]):+.2f}, px/deg {np.median(r[:,2]/r[:,0]):+.2f}; Lpad dx {np.median(r[:,5]):+.2f}, Rpad dx {np.median(r[:,6]):+.2f}, cube dx {np.median(r[:,7]):+.2f}")
    still = r[r[:, 4] <= 3.0]
    if len(still): print(f"arm-still grasps ({len(still)}): dW median {np.median(still[:,2]):+.2f} px, px/deg {np.median(still[:,2]/still[:,0]):+.2f}, Lpad {np.median(still[:,5]):+.2f}, Rpad {np.median(still[:,6]):+.2f}, cube {np.median(still[:,7]):+.2f}")
