"""GSM8K eval with vLLM (chat template, non-thinking, greedy). Usage: python eval_gsm8k.py MODEL OUT_DIR [LIMIT]"""
import glob, json, os, re, sys, time
import pandas as pd
from vllm import LLM, SamplingParams

model, out_dir = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out_dir, exist_ok=True)
df = pd.read_parquet(sorted(glob.glob("data/eval/gsm8k/main/test-*.parquet"))[0])
if limit: df = df.head(limit)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

def gold(a): return a.split("####")[-1].strip().replace(",", "")
def norm(x):
    x = x.replace(",", "").replace("$", "").strip().rstrip(".")
    try: return float(x)
    except ValueError: return None
def extract(t):
    m = re.findall(r"\\boxed\{([^}]*)\}", t)
    if m:
        n = re.findall(r"-?[\d,]*\.?\d+", m[-1])
        if n: return n[-1], "boxed"
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", t)
    if m: return m[-1], "####"
    n = re.findall(r"-?[\d,]*\.?\d+", t)
    return (n[-1], "last_number") if n else (None, "none")

llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")), seed=0)
sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=1024, seed=0)
msgs = [[{"role": "user", "content": q + SUFFIX}] for q in df["question"]]
t0 = time.time()
outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=True)
correct, how = 0, {}
with open(f"{out_dir}/gsm8k_outputs.jsonl", "w") as f:
    for (_, r), o in zip(df.iterrows(), outs):
        txt = o.outputs[0].text
        pred, method = extract(txt)
        ok = pred is not None and norm(pred) is not None and abs(norm(pred) - norm(gold(r["answer"]))) < 1e-6
        correct += ok; how[method] = how.get(method, 0) + 1
        f.write(json.dumps({"question": r["question"], "gold": gold(r["answer"]), "pred": pred, "method": method, "correct": bool(ok), "output": txt}, ensure_ascii=False) + "\n")
res = {"model": model, "bench": "gsm8k_test", "n": len(df), "acc": round(100 * correct / len(df), 2), "extract_methods": how, "seconds": round(time.time() - t0, 1),
       "config": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 1024, "seed": 0, "enable_thinking": False, "prompt_suffix": SUFFIX}}
json.dump(res, open(f"{out_dir}/gsm8k_result.json", "w"), indent=2, ensure_ascii=False)
print("RESULT", json.dumps(res, ensure_ascii=False))
