"""Generate OPD-style vLLM rollouts from a ckpt and measure truncation / repetition.
python scripts/eval/rollout_check.py CONFIG CKPT OUT_DIR [N=64]"""
import json, os, sys, collections
import torch
from mm_opd.imgdata import JsonlImages, VisionPrep
from mm_opd.model.student import Student
from mm_opd.train.common import load_cfg, seed_all
cfg_p, ck, out = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 64
cfg = load_cfg(cfg_p); O = cfg["opd"]; seed_all(0); os.makedirs(out, exist_ok=True)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
eng = LLM(model=cfg["student"]["path"], dtype="bfloat16", enable_lora=True, max_lora_rank=cfg["student"]["lora"]["r"], max_loras=1,
          enable_prompt_embeds=True, gpu_memory_utilization=O.get("vllm_gpu_util", 0.12), max_model_len=O["total_max_len"], seed=0)
dev = cfg["student"]["device"]
st = Student(cfg, dev); st.projector.load_state_dict(torch.load(f"{ck}/projector.pt"))
ds = JsonlImages(cfg["data"]["opd_jsonl"], cfg["data"]["coco_root"], N); prep = VisionPrep(cfg); tok = st.tok
prompts = []
for r in ds:
    ids = tok(st.prompt_text(r["question"]), add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prompts.append({"prompt_embeds": st.embed(ids, prep([r["pil"]]))[0].to(torch.bfloat16).cpu()})
sp = SamplingParams(temperature=O["rollout_temperature"], top_p=O["rollout_top_p"], max_tokens=O["max_new_tokens"], seed=0)
outs = eng.generate(prompts, sp, lora_request=LoRARequest("ck", 1, f"{ck}/lora"))
def rep4(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)
rows = []
for r, o in zip(ds.rows, outs):
    t = o.outputs[0]; ids = list(t.token_ids)
    rows.append({"image": r["image"], "question": r["question"], "n_tokens": len(ids), "finish": t.finish_reason, "rep4": round(rep4(ids), 3), "text": t.text})
with open(f"{out}/rollouts.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
cap = [r for r in rows if r["finish"] == "length"]
summ = {"n": len(rows), "hit_max_frac": round(len(cap) / len(rows), 3), "mean_tokens": round(sum(r["n_tokens"] for r in rows) / len(rows), 1),
        "mean_rep4_all": round(sum(r["rep4"] for r in rows) / len(rows), 3),
        "mean_rep4_capped": round(sum(r["rep4"] for r in cap) / max(len(cap), 1), 3),
        "capped_rep4_gt_0.5": sum(r["rep4"] > 0.5 for r in cap), "noncapped_rep4_gt_0.5": sum(r["rep4"] > 0.5 for r in rows if r["finish"] != "length")}
json.dump(summ, open(f"{out}/summary.json", "w"), indent=2); print("SUMMARY", json.dumps(summ))
for r in sorted(cap, key=lambda x: -x["rep4"])[:3]:
    print("CAPPED rep4=%s Q=%r\n  HEAD=%r\n  TAIL=%r" % (r["rep4"], r["question"][:100], r["text"][:300], r["text"][-300:]))
