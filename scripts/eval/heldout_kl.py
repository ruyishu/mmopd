"""Held-out long-answer eval: student answers 200 held-out LLaVA-Instruct COCO questions (images not in the
10k OPD/SFT/KD set), greedy, max 512, non-thinking (vLLM prompt_embeds + LoRA). Teacher (Qwen3-VL-4B) scores
the student's answer tokens (own template + processor, original image): mean per-token reverse
KL(student||teacher), teacher NLL of student tokens, answer length, repetition (repeated 4-gram fraction).
python scripts/eval/heldout_kl.py CONFIG CKPT OUT_DIR [N=200]"""
import json, os, random, sys, time
import torch
from mm_opd.imgdata import JsonlImages, VisionPrep
from mm_opd.model.student import Student
from mm_opd.train.common import load_cfg, seed_all
from mm_opd.train.opd import Teacher, chunked_reverse_kl
cfg_p, ck, out = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
cfg = load_cfg(cfg_p); O = cfg["opd"]; seed_all(0); os.makedirs(out, exist_ok=True)
HO = "data/demo/heldout_200.jsonl"
if not os.path.exists(HO):
    train_imgs = {json.loads(l)["image"] for l in open(cfg["data"]["opd_jsonl"])}
    have = set(os.listdir(cfg["data"]["coco_root"]))
    js = [e for e in json.load(open("data/instruct/llava_v1_5_mix665k.json")) if str(e.get("image", "")).startswith("coco/train2017/")]
    random.Random(1).shuffle(js); rows, used = [], set()
    for e in js:
        fn = e["image"].split("/")[-1]
        if fn in train_imgs or fn in used or fn not in have: continue
        rows.append({"id": e.get("id"), "image": fn, "question": e["conversations"][0]["value"].replace("<image>", "").strip()}); used.add(fn)
        if len(rows) >= 200: break
    with open(HO, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
has_lora = os.path.isdir(f"{ck}/lora")
eng = LLM(model=cfg["student"]["path"], dtype="bfloat16", enable_lora=has_lora, max_lora_rank=cfg["student"]["lora"]["r"], max_loras=1,
          enable_prompt_embeds=True, gpu_memory_utilization=O.get("vllm_gpu_util", 0.12), max_model_len=O["total_max_len"], seed=0)
sdev, tdev = cfg["student"]["device"], cfg["teacher"]["device"]
st = Student(cfg, sdev); st.projector.load_state_dict(torch.load(f"{ck}/projector.pt"))
ds = JsonlImages(HO, cfg["data"]["coco_root"], N); prep = VisionPrep(cfg); tok = st.tok
prompts, pids = [], []
for r in ds:
    ids = tok(st.prompt_text(r["question"]), add_special_tokens=False, return_tensors="pt")["input_ids"].to(sdev); pids.append(ids)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prompts.append({"prompt_embeds": st.embed(ids, prep([r["pil"]]))[0].to(torch.bfloat16).cpu()})
t0 = time.time()
outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=O["max_new_tokens"]), lora_request=LoRARequest("ck", 1, f"{ck}/lora") if has_lora else None)
gen_sec = time.time() - t0
del eng; torch.cuda.empty_cache()
if has_lora:
    from peft import PeftModel
    st.llm = PeftModel.from_pretrained(st.llm, f"{ck}/lora").merge_and_unload()
st.llm.eval(); teacher = Teacher(cfg["teacher"]["path"], tdev)
def rep4(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
recs, kl_sum, nll_sum, ntok = [], 0.0, 0.0, 0
for r, ids, o in zip(ds, pids, outs):
    a = list(o.outputs[0].token_ids)
    if not a: continue
    with torch.no_grad():
        T = teacher.answer_logits(r["pil"], r["question"], a).to(sdev).float()
        full = torch.cat([ids, torch.tensor([a], device=sdev)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            S = st.llm(inputs_embeds=st.embed(full, prep([r["pil"]])), logits_to_keep=len(a) + 1).logits[0, :-1].float()
        kl = chunked_reverse_kl(S, T, 1.0, 256)
        nll = -torch.log_softmax(T, -1).gather(1, torch.tensor(a, device=sdev)[:, None]).squeeze(1)
    kl_sum += kl.sum().item(); nll_sum += nll.sum().item(); ntok += len(a)
    recs.append({"image": r["image"], "question": r["question"], "n_tokens": len(a), "finish": o.outputs[0].finish_reason, "rep4": round(rep4(a), 3),
                 "rev_kl": round(kl.mean().item(), 4), "teacher_nll": round(nll.mean().item(), 4), "text": o.outputs[0].text})
with open(f"{out}/heldout_outputs.jsonl", "w") as f:
    for x in recs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
n = len(recs)
res = {"ckpt": ck, "n": n, "rev_kl_per_token": round(kl_sum / ntok, 4), "teacher_nll_per_token": round(nll_sum / ntok, 4),
       "rev_kl_mean_of_samples": round(sum(x["rev_kl"] for x in recs) / n, 4), "mean_ans_len": round(ntok / n, 1),
       "hit_max_frac": round(sum(x["finish"] == "length" for x in recs) / n, 3), "mean_rep4": round(sum(x["rep4"] for x in recs) / n, 3),
       "rep4_gt_0.5_frac": round(sum(x["rep4"] > 0.5 for x in recs) / n, 3), "gen_sec": round(gen_sec, 1), "decoding": "greedy, max 512, non-thinking"}
json.dump(res, open(f"{out}/heldout_result.json", "w"), indent=2); print("RESULT", json.dumps(res))
