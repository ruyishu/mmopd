"""MMBench-EN dev eval of a vLLM-loadable VLM (teacher). Usage: python eval_mmbench.py MODEL OUT_DIR [LIMIT]
The HF dev split already contains the circular rotations (rows sharing index % 1e6). Each row is asked once.
single_pass = accuracy on original rows (index < 1e6); circular = question correct iff ALL its rows correct; unparseable = wrong."""
import base64, glob, json, os, re, sys, time
import pandas as pd
from vllm import LLM, SamplingParams
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmb_common import load_dev, build_prompt, extract, score

model, out_dir = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out_dir, exist_ok=True)
df = load_dev(limit)
def img_url(im):
    b = im["bytes"] if isinstance(im, dict) else im
    return "data:image/png;base64," + base64.b64encode(b).decode()
llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")), seed=0, limit_mm_per_prompt={"image": 1})
sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=64, seed=0)
msgs, metas = [], []
for _, r in df.iterrows():
    p, opts = build_prompt(r)
    msgs.append([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_url(r["image"])}}, {"type": "text", "text": p}]}])
    metas.append(opts)
t0 = time.time()
outs = llm.chat(msgs, sp, use_tqdm=True)
recs = []
for (_, r), o, opts in zip(df.iterrows(), outs, metas):
    txt = o.outputs[0].text; pred, method = extract(txt, opts)
    recs.append({"index": int(r["index"]), "gold": r["answer"], "pred": pred, "method": method, "correct": pred == str(r["answer"]).strip(), "output": txt})
res = score(recs); res.update({"model": model, "seconds": round(time.time() - t0, 1),
           "config": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 64, "seed": 0, "circular": True, "circular_source": "dataset rotations grouped by index % 1e6"}})
with open(f"{out_dir}/mmbench_outputs.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(res, open(f"{out_dir}/mmbench_result.json", "w"), indent=2, ensure_ascii=False)
print("RESULT", json.dumps(res, ensure_ascii=False))
