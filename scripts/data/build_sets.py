"""Build demo jsonl sets from COCO train2017 images on disk.
warmup: N_WARM (image, caption) pairs, distinct images, one random caption each.
opd: N_OPD LLaVA-Instruct-665K entries with image coco/train2017/*, first human turn, '<image>' stripped.
Only images that exist on disk are used. Usage: python scripts/data/build_sets.py [N_WARM N_OPD]"""
import glob, json, os, random, sys
import pyarrow.parquet as pq
N_WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
N_OPD = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
root = "data/coco/train2017"; os.makedirs("data/demo", exist_ok=True)
have = set(os.listdir(root)); print("images on disk", len(have))
t = pq.read_table(glob.glob("data/coco/raw/captions/data/train-*.parquet")[0]).to_pylist()
print("caption rows", len(t), "keys", list(t[0].keys()))
fk = next(k for k in t[0] if "file" in k); ck = next(k for k in t[0] if "caption" in k)
rng = random.Random(0)
caps = {}
for r in t:
    fn = os.path.basename(r[fk]); c = r[ck]
    c = c if isinstance(c, list) else [c]
    caps.setdefault(fn, []).extend(c)
fns = sorted(f for f in caps if f in have); rng.shuffle(fns)
w = [{"image": f, "caption": rng.choice(caps[f]).strip()} for f in fns[:N_WARM]]
with open("data/demo/warmup_coco50k.jsonl", "w") as f:
    for r in w: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("warmup written", len(w), "(caption images available:", len(fns), ")")
js = json.load(open("data/instruct/llava_v1_5_mix665k.json"))
coco = [e for e in js if str(e.get("image", "")).startswith("coco/train2017/")]
print("llava entries", len(js), "coco/train2017 entries", len(coco))
rng2 = random.Random(0); rng2.shuffle(coco)
o, miss = [], 0
for e in coco:
    fn = e["image"].split("/")[-1]
    if fn not in have: miss += 1; continue
    q = e["conversations"][0]["value"].replace("<image>", "").strip()
    o.append({"id": e.get("id"), "image": fn, "question": q})
    if len(o) >= N_OPD: break
with open("data/demo/opd_llava_coco10k.jsonl", "w") as f:
    for r in o: f.write(json.dumps(r, ensure_ascii=False) + "\n")
allres = sum(e["image"].split("/")[-1] in have for e in coco)
print("opd written", len(o), "| coco entries resolvable now:", allres, "/", len(coco))
