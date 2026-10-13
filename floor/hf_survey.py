"""Survey Hugging Face datasets whose id matches a search term: fetch meta/info.json (and tasks) for each and print
robot type, fps, episodes, cameras, whether torque is in the state, and the task strings.
usage: python3 -I hf_survey.py OUTDIR [search]"""
import json, os, sys, urllib.request, urllib.parse

out = sys.argv[1]; term = sys.argv[2] if len(sys.argv) > 2 else "openarm"
os.makedirs(out, exist_ok=True)

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "survey/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

ids = []
for off in (0, 100, 200):
    lst = json.loads(get(f"https://huggingface.co/api/datasets?search={term}&limit=100&offset={off}"))
    ids += [d["id"] for d in lst]
    if len(lst) < 100: break
ids = sorted(set(ids))
print(len(ids), "datasets match", term)
rows = []
for i, did in enumerate(ids):
    base = f"https://huggingface.co/datasets/{did}/resolve/main/meta/"
    try:
        info = json.loads(get(base + "info.json"))
    except Exception as e:
        print(f"{did}: no info.json ({type(e).__name__})"); continue
    json.dump(info, open(os.path.join(out, did.replace("/", "__") + ".json"), "w"))
    feats = info.get("features", {})
    cams = {k: v.get("shape") for k, v in feats.items() if v.get("dtype") in ("video", "image")}
    names = feats.get("observation.state", {}).get("names") or []
    if isinstance(names, dict): names = sum((list(v) for v in names.values()), [])
    torque = any("torque" in str(n).lower() or "effort" in str(n).lower() for n in names)
    sdim = feats.get("observation.state", {}).get("shape")
    tasks = ""
    for tf in ("tasks.jsonl", "tasks.parquet"):
        try:
            b = get(base + tf)
        except Exception:
            continue
        if tf.endswith("jsonl"):
            tasks = " | ".join(json.loads(l)["task"] for l in b.decode().splitlines() if l.strip())
        else:
            p = os.path.join(out, did.replace("/", "__") + "_tasks.parquet"); open(p, "wb").write(b)
            try:
                import pyarrow.parquet as pq
                t = pq.read_table(p).to_pydict()
                col = t.get("task") or t.get("__index_level_0__") or []
                tasks = " | ".join(str(x) for x in col)
            except Exception as e:
                tasks = f"(parquet {type(e).__name__})"
        break
    rows.append((did, info.get("robot_type"), info.get("fps"), info.get("total_episodes"), info.get("total_frames"),
                 info.get("codebase_version"), sdim, torque, cams, tasks[:160]))
    print(f"{did} | {info.get('robot_type')} | fps {info.get('fps')} | ep {info.get('total_episodes')} | "
          f"fr {info.get('total_frames')} | {info.get('codebase_version')} | state {sdim} | torque {torque} | "
          f"cams {cams} | {tasks[:160]}")
json.dump(rows, open(os.path.join(out, "_rows.json"), "w"))
