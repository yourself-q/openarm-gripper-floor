"""One episode of a LeRobot v3 dataset: print the gripper trace (pos, action, vel, torque) every K frames, then cut
video frames on a time grid from the episode's video segment and tile them into one montage PNG.
usage: python3 -I ep_montage.py ROOT CAMKEY GRIPPER_SUBSTR EPISODE [--step-s 1.0] [--every 10] [--out montage.png]"""
import glob, json, os, subprocess, sys
import numpy as np, pyarrow.parquet as pq, pyarrow as pa

root, cam, want, e = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def opt(k, d):
    return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
step_s, every, out = opt("--step-s", 1.0), opt("--every", 10), opt("--out", f"frames/montage_{e}.png")
info = json.load(open(os.path.join(root, "meta/info.json"))); feats = info["features"]
tab = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))])
ep = np.array(tab.column("episode_index").to_pylist())
rows = np.nonzero(ep == e)[0]; tab = tab.take(pa.array(rows))
ts = np.array(tab.column("timestamp").to_pylist(), float)
def col(name): return np.array(tab.column(name).to_pylist(), float)
loc = {}
for fname, fd in feats.items():
    for i, n in enumerate(fd.get("names") or []):
        n = str(n).lower()
        if want in n:
            if fname == "action" and n.endswith(".pos"): loc["act"] = (fname, i)
            elif fname.startswith("observation"):
                for key, tags in (("pos", (".pos",)), ("vel", (".vel",)), ("tau", (".torque", ".tau"))):
                    if any(n.endswith(t) for t in tags): loc[key] = (fname, i)
cache = {}
def get(k):
    f, i = loc[k]
    if f not in cache: cache[f] = col(f)
    return cache[f][:, i]
pos, act, vel, tau = get("pos"), get("act"), get("vel"), get("tau")
print(f"episode {e}: {len(ts)} frames, t {ts[0]:.2f}..{ts[-1]:.2f} s")
print("     t     pos     act     vel      tau")
for i in range(0, len(ts), every):
    print(f"{ts[i]:6.2f} {pos[i]:7.2f} {act[i]:7.2f} {vel[i]:7.1f} {tau[i]:8.3f}")
i = int(np.argmax(np.abs(tau))); print(f"max |tau| {tau[i]:.3f} at t {ts[i]:.2f}, pos {pos[i]:.2f}, act {act[i]:.2f}")
# video segment
meta = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(os.path.join(root, "meta/episodes/chunk-*/file-*.parquet")))])
mi = np.nonzero(np.array(meta.column("episode_index").to_pylist()) == e)[0][0]
cols = [c for c in meta.column_names if c.startswith(f"videos/{cam}/")]
seg = {c.split("/")[-1]: meta.column(c)[mi].as_py() for c in cols}
print("video segment:", seg)
vf = os.path.join(root, f"videos/{cam}/chunk-{seg.get('chunk_index', 0):03d}/file-{seg.get('file_index', 0):03d}.mp4")
if not os.path.exists(vf): sys.exit(f"video file not pulled: {vf}")
t0 = seg["from_timestamp"]; grid = np.arange(0, ts[-1] - ts[0] + 1e-6, step_s)
os.makedirs("frames/_tmp", exist_ok=True)
for f in glob.glob("frames/_tmp/*.png"): os.unlink(f)
for k, g in enumerate(grid):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t0 + g:.3f}", "-i", vf, "-frames:v", "1",
                    "-vf", "scale=320:240", f"frames/_tmp/f_{k:03d}.png"], check=True)
n = len(grid); cols_n = 6 if n > 12 else 4; rows_n = int(np.ceil(n / cols_n))
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "1", "-i", "frames/_tmp/f_%03d.png",
                "-vf", f"tile={cols_n}x{rows_n}", "-frames:v", "1", out], check=True)
print(f"montage {out}: {n} frames at {step_s} s steps, {cols_n} per row, row-major from t = 0")
