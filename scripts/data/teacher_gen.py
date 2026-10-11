"""Teacher responses for the OPD question set (for SFT / offline-KD baselines).
python scripts/data/teacher_gen.py [IN=data/demo/opd_llava_coco10k.jsonl OUT=data/demo/teacher_resp_10k.jsonl]
Qwen3-VL-4B-Instruct via vLLM, temp .7 top_p .8 top_k 20, max 512, seed 0, original image."""
import base64, json, os, sys, time
from vllm import LLM, SamplingParams
IN = sys.argv[1] if len(sys.argv) > 1 else "data/demo/opd_llava_coco10k.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/demo/teacher_resp_10k.jsonl"
rows = [json.loads(l) for l in open(IN)]
llm = LLM(model="models/Qwen3-VL-4B-Instruct", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.25")), seed=0, limit_mm_per_prompt={"image": 1})
sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=512, seed=0)
def url(p): return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()
t0 = time.time(); B = 1000
with open(OUT, "w") as f:
    for i in range(0, len(rows), B):
        ch = rows[i:i + B]
        msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(f"data/coco/train2017/{r['image']}")}}, {"type": "text", "text": r["question"]}]}] for r in ch]
        outs = llm.chat(msgs, sp, use_tqdm=False)
        for r, o in zip(ch, outs):
            t = o.outputs[0]
            f.write(json.dumps({**r, "response": t.text, "n_tokens": len(t.token_ids), "finish": t.finish_reason}, ensure_ascii=False) + "\n")
        f.flush(); print("done", i + len(ch), round(time.time() - t0), "s", flush=True)
print("TEACHER_GEN_DONE", len(rows), round(time.time() - t0), "s")
