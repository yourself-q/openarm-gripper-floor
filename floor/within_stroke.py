"""Within-episode moving-friction scatter without the small-group shrinkage: closing 40-100 deg/s steady frames, residual about
the per-5-deg-bin median, then (a) DOF-corrected pooled within-episode std sqrt(sum (r - mean_g)^2 / (N - G)) over episodes with
>= 2 frames, (b) robust: median |r_i - r_j| over within-episode pairs / 0.954 (Gaussian-consistent), (c) same restricted to
episodes with >= 3 frames. Also frames per episode. usage: python3 -I within_stroke.py ROOT..."""
import glob, os, sys, itertools, numpy as np, pyarrow.parquet as pq, pyarrow as pa
for root in sys.argv[1:]:
    tab = pa.concat_tables([pq.read_table(f, columns=["observation.state", "episode_index"]) for f in sorted(glob.glob(os.path.join(root, "data/chunk-*/file-*.parquet")))])
    S = np.array(tab.column("observation.state").to_pylist(), float); ep = np.array(tab.column("episode_index").to_pylist())
    pos, vel, tau = S[:, 21], S[:, 22], S[:, 23]
    lo, hi = np.percentile(pos, 0.5) + 3, np.percentile(pos, 99.5) - 3
    acc = np.zeros_like(vel); acc[1:-1] = (vel[2:] - vel[:-2]) / 2 * 30
    same = np.zeros(len(ep), bool); same[1:-1] = (ep[2:] == ep[1:-1]) & (ep[:-2] == ep[1:-1])
    m = same & (vel > 40) & (vel <= 100) & (np.abs(acc) < 150) & (pos > lo) & (pos < hi)
    b = np.floor(pos[m] / 5) * 5; t = np.abs(tau[m]); e = ep[m]
    res = t - np.array([np.median(t[b == bb]) for bb in b])
    eps, cnt = np.unique(e, return_counts=True)
    out = [f"{os.path.basename(root)}: n {m.sum()}, episodes {len(eps)}, frames/episode p50 {np.median(cnt):.0f} max {cnt.max()}, "
           f"episodes with 1/2/3+ frames {np.sum(cnt==1)}/{np.sum(cnt==2)}/{np.sum(cnt>=3)}"]
    for kmin in (2, 3):
        ss = 0.0; dof = 0; pairs = []
        for ee, c in zip(eps, cnt):
            if c < kmin: continue
            r = res[e == ee]; ss += np.sum((r - r.mean()) ** 2); dof += len(r) - 1
            pairs += [abs(x - y) for x, y in itertools.combinations(r, 2)]
        pooled = np.sqrt(ss / dof) if dof else float("nan")
        robust = np.median(pairs) / 0.954 if pairs else float("nan")
        out.append(f"  episodes with >= {kmin} frames: pooled within-episode std {pooled:.4f} (dof {dof}); robust pairwise sigma {robust:.4f} (pairs {len(pairs)})")
    half = lambda r: 0.5 * (np.percentile(r, 84) - np.percentile(r, 16))
    out.append(f"  for reference: about-bin half {half(res):.4f} std {res.std():.4f}")
    print("\n".join(out))
