"""List a LeRobot dataset's files on the Hub and download its data parquets (all, if the total is small) plus the
first video file per camera when it is under a size cap. Writes into OUTDIR/<owner>__<name>/ keeping the repo paths.
usage: python3 -I hf_pull.py OUTDIR DATASET_ID [--video-cap-mb 200] [--data-cap-mb 300] [--no-video]"""
import json, os, sys, urllib.request

out, did = sys.argv[1], sys.argv[2]
vcap = float(sys.argv[sys.argv.index("--video-cap-mb") + 1]) if "--video-cap-mb" in sys.argv else 200
dcap = float(sys.argv[sys.argv.index("--data-cap-mb") + 1]) if "--data-cap-mb" in sys.argv else 300
novid = "--no-video" in sys.argv
root = os.path.join(out, did.replace("/", "__")); os.makedirs(root, exist_ok=True)

def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "pull/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

tree = json.loads(get(f"https://huggingface.co/api/datasets/{did}/tree/main?recursive=true"))
files = [(e["path"], e.get("size", 0)) for e in tree if e.get("type") == "file"]
json.dump(files, open(os.path.join(root, "_files.json"), "w"))
print(did, len(files), "files")
for p, s in files:
    if not p.startswith("videos/") or p.endswith(("file-000.mp4", "episode_000000.mp4")):
        print(f"  {s/1e6:9.2f} MB  {p}")
def pull(p):
    dst = os.path.join(root, p); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst): return
    open(dst, "wb").write(get(f"https://huggingface.co/datasets/{did}/resolve/main/{p}", timeout=600))
    print("  pulled", p)
for p, s in files:
    if p.startswith("meta/"): pull(p)
data = [(p, s) for p, s in files if p.startswith("data/") and p.endswith(".parquet")]
if sum(s for _, s in data) / 1e6 <= dcap:
    for p, _ in data: pull(p)
else:
    print("  data too large:", sum(s for _, s in data) / 1e6, "MB; pulling the first file only"); pull(data[0][0])
if not novid:
    seen = set()
    for p, s in sorted(files):
        if p.startswith("videos/") and p.endswith(".mp4"):
            cam = p.split("/")[1]
            if cam in seen: continue
            if s / 1e6 <= vcap: pull(p); seen.add(cam)
            else: print(f"  video {p} is {s/1e6:.0f} MB > cap, skipped"); seen.add(cam)
